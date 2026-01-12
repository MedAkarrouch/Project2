#!/usr/bin/env python3
"""
04_index_models.py

Compute (LFD + DepthBuffer) descriptors for the INDEX split and store them in MongoDB.

Updates requested (for DepthBuffer improvements):
- image_size default -> 256 (you’ll run with --image-size 256 anyway)
- add CLI option for depth rotation set (stored in metadata for traceability)
- add L2-normalization option (per-view, row-wise) applied BEFORE storing features
  (recommended: store normalized features in MongoDB so retrieval is faster/consistent)

Behavior (Option A):
- Skip a model if it already has BOTH:
    lfd.features AND depth.features
  unless --recompute is provided.

Run:
    python scripts/04_index_models.py

Useful dev runs:
    python scripts/04_index_models.py --image-size 256 --depth-rotation-set grid24 --l2-normalize
    python scripts/04_index_models.py --recompute --image-size 256 --depth-rotation-set grid24 --l2-normalize --limit 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError

# -------------------------
# Core imports (your code)
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.mesh_loader import MeshLoader, Mesh
from core.mesh_normalizer import MeshNormalizer
from core.renderer import Renderer
from core.lfd_descriptor import LFDDescriptor
from core.depth_buffer_descriptor import DepthBufferDescriptor


# -------------------------
# Paths / Defaults
# -------------------------
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB = "objectlens"
DEFAULT_COLLECTION = "models"

DEFAULT_IMAGE_SIZE = 256  # updated default as requested

# Your dataset root:
#   data/raw/3D Models/<ClassName>/<filename>.obj
OBJ_ROOT_REL = Path("data") / "raw" / "3D Models"
INDEX_CSV_REL = Path("data") / "splits" / "index.csv"


# -------------------------
# Helpers
# -------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_mongo(uri: str, timeout_ms: int = 4000) -> MongoClient:
    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        raise RuntimeError(
            "Could not connect to MongoDB.\n"
            f"URI: {uri}\n"
            "Make sure MongoDB is running locally and listening on 27017."
        ) from e
    return client


def read_index_csv(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"index.csv not found: {path}")
    out: List[Tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames or "class" not in reader.fieldnames:
            raise ValueError(f"index.csv must contain headers: filename,class. Got: {reader.fieldnames}")
        for r in reader:
            fn = (r.get("filename") or "").strip()
            cl = (r.get("class") or "").strip()
            if fn and cl:
                out.append((fn, cl))
    if not out:
        raise RuntimeError("index.csv is empty.")
    return out


def obj_path_for(filename: str, cls: str, obj_root: Path, fallback_map: Optional[Dict[str, Path]] = None) -> Path:
    """
    Primary: data/raw/3D Models/<class>/<filename>
    Fallback: search by filename under obj_root (cached map).
    """
    primary = obj_root / cls / filename
    if primary.exists():
        return primary

    if fallback_map is not None:
        hit = fallback_map.get(filename)
        if hit is not None and hit.exists():
            return hit

    raise FileNotFoundError(f"OBJ not found for {filename} (class={cls}). Tried: {primary}")


def build_fallback_map(obj_root: Path) -> Dict[str, Path]:
    """
    Build filename -> path map for fallback resolution.
    If multiple matches:
      - prefer paths that do NOT contain 'All Models'
      - then choose shortest path
    """
    hits: Dict[str, List[Path]] = {}
    for p in obj_root.rglob("*.obj"):
        hits.setdefault(p.name, []).append(p)

    chosen: Dict[str, Path] = {}
    for name, paths in hits.items():
        paths_sorted = sorted(
            paths,
            key=lambda x: (("All Models" in x.parts) or ("All Models" in str(x)), len(str(x))),
        )
        chosen[name] = paths_sorted[0]
    return chosen


def to_builtin(obj):
    """
    Convert dataclasses + numpy to MongoDB-friendly builtins.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if is_dataclass(obj):
        return {k: to_builtin(v) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]

    return obj


def has_both_descriptors(doc: Optional[dict]) -> bool:
    if not doc:
        return False
    try:
        lfd_ok = doc.get("lfd", {}).get("features", None) is not None
        dep_ok = doc.get("depth", {}).get("features", None) is not None
        if isinstance(doc.get("lfd", {}).get("features", None), list) and len(doc["lfd"]["features"]) == 0:
            lfd_ok = False
        if isinstance(doc.get("depth", {}).get("features", None), list) and len(doc["depth"]["features"]) == 0:
            dep_ok = False
        return bool(lfd_ok and dep_ok)
    except Exception:
        return False


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization: each view feature vector gets norm=1 (unless it's all zeros).
    """
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return (X / n).astype(np.float32)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Index models (LFD + Depth) into MongoDB.")
    ap.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)

    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)

    # NEW: record the depth rotation set you plan to use later in retrieval (traceability)
    ap.add_argument(
        "--depth-rotation-set",
        choices=["yaw12", "yaw24", "grid12", "grid24"],
        default="grid24",
        help="Rotation set used later by SimilarityEngine for DEPTH. Stored as metadata for traceability.",
    )

    # NEW: store normalized features
    ap.add_argument(
        "--l2-normalize",
        action="store_true",
        help="Apply row-wise L2-normalization to per-view features before storing.",
    )

    ap.add_argument("--limit", type=int, default=None, help="Index only first N models (debug).")
    ap.add_argument("--recompute", action="store_true", help="Recompute even if already stored.")
    ap.add_argument("--no-fallback-scan", action="store_true", help="Disable fallback filename scan.")
    args = ap.parse_args()

    root = project_root()
    obj_root = root / OBJ_ROOT_REL
    index_csv = root / INDEX_CSV_REL

    if not obj_root.exists():
        raise FileNotFoundError(f"OBJ root folder not found: {obj_root}")

    items = read_index_csv(index_csv)
    if args.limit is not None:
        items = items[: max(args.limit, 0)]

    print(f"[INFO] OBJ root : {obj_root}")
    print(f"[INFO] Index CSV: {index_csv}")
    print(f"[INFO] Models   : {len(items)}")
    print(f"[INFO] Mongo    : {args.mongo_uri} / {args.db}.{args.collection}")
    print(f"[INFO] Image    : {args.image_size}x{args.image_size}")
    print(f"[INFO] DepthRot : {args.depth_rotation_set}")
    print(f"[INFO] L2-norm  : {'ON' if args.l2_normalize else 'OFF'}")
    print(f"[INFO] Mode     : {'RECOMPUTE' if args.recompute else 'SKIP-IF-EXISTS'}")

    # Mongo connect
    client = connect_mongo(args.mongo_uri)
    coll: Collection = client[args.db][args.collection]

    # Optional fallback map for missing files
    fallback_map: Optional[Dict[str, Path]] = None
    if not args.no_fallback_scan:
        print("[INFO] Building fallback filename->path map (one-time scan)...")
        t0 = time.time()
        fallback_map = build_fallback_map(obj_root)
        print(f"[OK] Fallback map ready ({len(fallback_map)} filenames) in {time.time() - t0:.2f}s")

    # Init GL renderer ONCE
    renderer = Renderer(width=args.image_size, height=args.image_size)

    # Init descriptor computers ONCE
    lfd = LFDDescriptor(renderer=renderer, image_size=args.image_size)
    depth = DepthBufferDescriptor(renderer=renderer, image_size=args.image_size)

    ok = 0
    skipped = 0
    failed = 0
    start_all = time.time()

    total = len(items)

    try:
        for i, (filename, cls) in enumerate(items, start=1):
            t_model = time.time()
            status = "OK"
            msg = ""

            existing = coll.find_one({"_id": filename}, {"lfd.features": 1, "depth.features": 1})
            if (not args.recompute) and has_both_descriptors(existing):
                skipped += 1
                dt = time.time() - t_model
                print(f"[{i:>4}/{total}] SKIP  {filename}  (class={cls})  {dt:.2f}s")
                continue

            try:
                obj_path = obj_path_for(filename, cls, obj_root, fallback_map=fallback_map)

                mesh: Mesh = MeshLoader.load(obj_path)
                mesh = MeshNormalizer.normalize(mesh)

                lfd_desc = lfd.compute(mesh)
                dep_desc = depth.compute(mesh)

                # NEW: L2-normalize per-view features before storing (recommended)
                lfd_feats = np.asarray(lfd_desc.features, dtype=np.float32)
                dep_feats = np.asarray(dep_desc.features, dtype=np.float32)
                if args.l2_normalize:
                    lfd_feats = l2_normalize_rows(lfd_feats)
                    dep_feats = l2_normalize_rows(dep_feats)

                doc = {
                    "_id": filename,
                    "class": cls,
                    "lfd": {
                        "preset": lfd_desc.meta.preset,
                        "image_size": int(getattr(lfd_desc.meta, "image_size", args.image_size)),
                        "ring_start": int(getattr(lfd_desc.meta, "ring_start", 2)),
                        "ring_size": int(getattr(lfd_desc.meta, "ring_size", 8)),
                        "num_views": int(getattr(lfd_desc.meta, "num_views", len(lfd_feats))),
                        "feature_dim": int(
                            getattr(
                                lfd_desc.meta,
                                "feature_dim",
                                int(lfd_feats.shape[1]) if lfd_feats.ndim == 2 else 0,
                            )
                        ),
                        "directions": to_builtin(getattr(lfd_desc.meta, "directions", None)),
                        "features": to_builtin(lfd_feats),
                        # NEW: provenance flags
                        "l2_normalized": bool(args.l2_normalize),
                    },
                    "depth": {
                        "preset": dep_desc.meta.preset,
                        "image_size": int(getattr(dep_desc.meta, "image_size", args.image_size)),
                        "num_views": int(getattr(dep_desc.meta, "num_views", len(dep_feats))),
                        "feature_dim": int(
                            getattr(
                                dep_desc.meta,
                                "feature_dim",
                                int(dep_feats.shape[1]) if dep_feats.ndim == 2 else 0,
                            )
                        ),
                        "linearized_depth": bool(getattr(dep_desc.meta, "linearized_depth", True)),
                        "directions": to_builtin(getattr(dep_desc.meta, "directions", None)),
                        "features": to_builtin(dep_feats),
                        # NEW: provenance flags
                        "l2_normalized": bool(args.l2_normalize),
                        "rotation_set": str(args.depth_rotation_set),
                    },
                    "updated_at": utc_now_iso(),
                }

                coll.update_one(
                    {"_id": filename},
                    {"$set": doc, "$unset": {"index_error": ""}},
                    upsert=True,
                )

                ok += 1

            except Exception as e:
                failed += 1
                status = "FAIL"
                msg = str(e)

                coll.update_one(
                    {"_id": filename},
                    {
                        "$set": {
                            "_id": filename,
                            "class": cls,
                            "index_error": {
                                "message": msg,
                                "time": utc_now_iso(),
                            },
                            "updated_at": utc_now_iso(),
                        }
                    },
                    upsert=True,
                )

            dt = time.time() - t_model
            done = ok + skipped + failed
            elapsed = time.time() - start_all
            rate = done / elapsed if elapsed > 1e-9 else 0.0
            eta = (total - done) / rate if rate > 1e-9 else 0.0

            if status == "OK":
                print(f"[{i:>4}/{total}] OK    {filename}  (class={cls})  {dt:.2f}s  ETA {eta/60:.1f}m")
            else:
                print(f"[{i:>4}/{total}] FAIL  {filename}  (class={cls})  {dt:.2f}s  err={msg}")

    finally:
        try:
            renderer.close()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass

    total_time = time.time() - start_all
    print("\n[FINISHED ✅]")
    print(f"  OK     : {ok}")
    print(f"  SKIP   : {skipped}")
    print(f"  FAIL   : {failed}")
    print(f"  TOTAL  : {ok + skipped + failed} / {total}")
    print(f"  TIME   : {total_time:.2f}s ({total_time/60:.2f} min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
