#!/usr/bin/env python3
"""
05_run_retrieval_eval.py

Evaluate 3D retrieval on the QUERY split against the INDEX split stored in MongoDB.

Two independent methods (NO fusion):
  - LFD (preset LFD_10)
  - Depth Buffer (preset DEPTH_42)

Primary metrics (robust to class imbalance + small classes):
  - Per-class mAP@20
  - Global MACRO computed over successful queries only (equal weight per class)

IMPORTANT (small classes / R < 20):
  - For any query/class with R = #relevant items in the index:
      cap_k = min(20, R)
    We ALWAYS compute the “@20” metrics using cap_k when R < 20, but we keep the
    headers as @20 (as requested). The effective k is reported in cap_k columns.

Also reported:
  - Precision@1,5,10,20
  - Recall@20 (vs total relevant R in index for that class)

Robustness:
  - If a query fails to compute a descriptor (e.g. LFD: "No contour found in silhouette."),
    we SKIP it, log it, and continue evaluation.

Outputs (out/eval by default):
  - lfd_metrics.json, depth_metrics.json
  - per_class_metrics.csv
  - per_query_metrics_lfd.csv, per_query_metrics_depth.csv   (successful queries only)
  - failed_queries_lfd.csv, failed_queries_depth.csv
  - raw_numbers.json
  - figure_tables.png
  - figure_graphs.png
  - retrieval_report.pdf

Updates requested:
  - Default image size = 256
  - Default depth rotation set = grid24
  - Macro-only reporting (already macro-only; ensured no micro remnants)
  - Clean “@k” naming everywhere; cap_k is its own column (no “cap” wording in @20 headers)
  - OPTIONAL: apply L2 normalization to query descriptors at eval time (recommended when you store L2-normalized
             features in Mongo; enabled with --l2-normalize to match indexing)

Critical fix included:
  - DepthMetadata now requires rotation_set + l2_normalized.
    Old Mongo docs may not have them, so we provide safe defaults when loading index descriptors:
        rotation_set = doc["depth"].get("rotation_set", "UNKNOWN")
        l2_normalized = doc["depth"].get("l2_normalized", False)

Run:
  python scripts/05_run_retrieval_eval.py --both --image-size 256 --depth-rotation-set grid24 --l2-normalize
  python scripts/05_run_retrieval_eval.py --method depth --image-size 256 --depth-rotation-set grid24 --l2-normalize
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Seaborn is optional (nicer visuals). If missing, we fall back to matplotlib defaults.
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    sns = None  # type: ignore
    _HAS_SEABORN = False

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError

# -------------------------
# Project imports
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.mesh_loader import MeshLoader, Mesh
from core.mesh_normalizer import MeshNormalizer
from core.renderer import Renderer
from core.lfd_descriptor import LFDDescriptor, LFDModelDescriptor, LFDMetadata
from core.depth_buffer_descriptor import DepthBufferDescriptor, DepthModelDescriptor, DepthMetadata
from core.similarity_engine import SimilarityEngine

# -------------------------
# Defaults / Paths
# -------------------------
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB = "objectlens"
DEFAULT_COLLECTION = "models"

OBJ_ROOT_REL = Path("data") / "raw" / "3D Models"
INDEX_CSV_REL = Path("data") / "splits" / "index.csv"
QUERY_CSV_REL = Path("data") / "splits" / "query.csv"

DEFAULT_IMAGE_SIZE = 256  # UPDATED
K_PRIMARY = 20
K_LIST = [1, 5, 10, 20]


# -------------------------
# Utils
# -------------------------
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


def read_split_csv(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    out: List[Tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames or "class" not in reader.fieldnames:
            raise ValueError(f"CSV must contain headers: filename,class. Got: {reader.fieldnames}")
        for r in reader:
            fn = (r.get("filename") or "").strip()
            cl = (r.get("class") or "").strip()
            if fn and cl:
                out.append((fn, cl))
    if not out:
        raise RuntimeError(f"{path.name} is empty.")
    return out


def obj_path_for(filename: str, cls: str, obj_root: Path) -> Path:
    p = obj_root / cls / filename
    if not p.exists():
        raise FileNotFoundError(f"OBJ not found: {p}")
    return p


def _set_plot_style() -> None:
    if _HAS_SEABORN:
        try:
            sns.set_theme(style="whitegrid", context="paper")
        except Exception:
            pass
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization: each view feature vector gets norm=1 (unless it's all zeros).
    This should match what you store in Mongo when indexing with --l2-normalize.
    """
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return (X / n).astype(np.float32)


# -------------------------
# Metrics
# -------------------------
def average_precision_at_k(relevance: List[int], k: int, total_relevant: int) -> float:
    """
    Truncated AP@k:
      AP@k = (1 / min(R, k)) * sum_{i=1..k} P(i) * rel(i)
    """
    if k <= 0:
        return 0.0
    R = int(total_relevant)
    if R <= 0:
        return 0.0

    rel = np.asarray(relevance[:k], dtype=np.int32)
    if rel.size == 0:
        return 0.0

    cumsum = np.cumsum(rel)
    idx = np.arange(1, rel.size + 1, dtype=np.float32)
    precision_at_i = cumsum / idx

    ap_num = float(np.sum(precision_at_i * rel))
    denom = float(min(R, k))
    return ap_num / denom


def precision_at_k(relevance: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = np.asarray(relevance[:k], dtype=np.int32)
    if rel.size == 0:
        return 0.0
    return float(rel.mean())


def recall_at_k(relevance: List[int], k: int, total_relevant: int) -> float:
    if k <= 0:
        return 0.0
    R = int(total_relevant)
    if R <= 0:
        return 0.0
    rel = int(np.sum(np.asarray(relevance[:k], dtype=np.int32)))
    return float(rel / R)


# -------------------------
# Loading index descriptors from MongoDB
# -------------------------
def load_index_descriptors(coll: Collection, method: str) -> Tuple[List[str], List[str], List[Any]]:
    """
    Returns:
      index_ids:    list of filenames
      index_class:  list of class labels
      index_desc:   list of ModelDescriptor (LFDModelDescriptor or DepthModelDescriptor)
    """
    if method == "lfd":
        proj = {
            "class": 1,
            "lfd.features": 1,
            "lfd.ring_start": 1,
            "lfd.ring_size": 1,
            "lfd.directions": 1,
            "lfd.image_size": 1,
            "lfd.l2_normalized": 1,
        }
        cursor = coll.find({"lfd.features": {"$exists": True, "$ne": []}}, proj)

        index_ids: List[str] = []
        index_cls: List[str] = []
        index_desc: List[Any] = []

        for doc in cursor:
            fn = doc["_id"]
            cl = doc.get("class", "UNKNOWN")

            feats = np.asarray(doc["lfd"]["features"], dtype=np.float32)
            ring_start = int(doc["lfd"].get("ring_start", 2))
            ring_size = int(doc["lfd"].get("ring_size", 8))
            image_size = int(doc["lfd"].get("image_size", 128))

            directions = doc["lfd"].get("directions", None)
            if directions is None:
                directions_np = np.zeros((int(feats.shape[0]), 3), dtype=np.float32)
            else:
                directions_np = np.asarray(directions, dtype=np.float32)

            meta = LFDMetadata(
                preset="LFD_10",
                image_size=image_size,
                ring_start=ring_start,
                ring_size=ring_size,
                directions=directions_np,
            )
            desc = LFDModelDescriptor(features=feats, meta=meta)

            index_ids.append(fn)
            index_cls.append(cl)
            index_desc.append(desc)

        return index_ids, index_cls, index_desc

    if method == "depth":
        proj = {
            "class": 1,
            "depth.features": 1,
            "depth.directions": 1,
            "depth.image_size": 1,
            "depth.linearized_depth": 1,
            "depth.l2_normalized": 1,
            "depth.rotation_set": 1,
        }
        cursor = coll.find({"depth.features": {"$exists": True, "$ne": []}}, proj)

        index_ids: List[str] = []
        index_cls: List[str] = []
        index_desc: List[Any] = []

        for doc in cursor:
            fn = doc["_id"]
            cl = doc.get("class", "UNKNOWN")

            feats = np.asarray(doc["depth"]["features"], dtype=np.float32)
            directions = np.asarray(doc["depth"]["directions"], dtype=np.float32)
            image_size = int(doc["depth"].get("image_size", 128))
            linearized_depth = bool(doc["depth"].get("linearized_depth", True))

            # ---- critical fix: provide defaults for new required metadata fields ----
            rotation_set = str(doc["depth"].get("rotation_set", "UNKNOWN"))
            l2_normalized = bool(doc["depth"].get("l2_normalized", False))

            meta = DepthMetadata(
                preset="DEPTH_42",
                image_size=image_size,
                directions=directions,
                linearized_depth=linearized_depth,
                rotation_set=rotation_set,
                l2_normalized=l2_normalized,
            )
            desc = DepthModelDescriptor(features=feats, meta=meta)

            index_ids.append(fn)
            index_cls.append(cl)
            index_desc.append(desc)

        return index_ids, index_cls, index_desc

    raise ValueError(f"Unknown method: {method}")


# -------------------------
# Compute query descriptor (on the fly)
# -------------------------
def compute_query_descriptor(
    method: str,
    renderer: Renderer,
    image_size: int,
    obj_root: Path,
    filename: str,
    cls: str,
    l2_normalize: bool = False,
) -> Any:
    obj_path = obj_path_for(filename, cls, obj_root)
    mesh: Mesh = MeshLoader.load(obj_path)
    mesh = MeshNormalizer.normalize(mesh)

    if method == "lfd":
        lfd = LFDDescriptor(renderer=renderer, image_size=image_size)
        desc = lfd.compute(mesh)
        if l2_normalize:
            feats = l2_normalize_rows(np.asarray(desc.features, dtype=np.float32))
            return LFDModelDescriptor(features=feats, meta=desc.meta)
        return desc

    if method == "depth":
        # IMPORTANT: DepthBufferDescriptor already supports per-view L2 normalization internally,
        # but here we keep eval-time normalization as a switch that can match older DB entries.
        depth = DepthBufferDescriptor(renderer=renderer, image_size=image_size)
        desc = depth.compute(mesh)
        if l2_normalize:
            feats = l2_normalize_rows(np.asarray(desc.features, dtype=np.float32))
            return DepthModelDescriptor(features=feats, meta=desc.meta)
        return desc

    raise ValueError(f"Unknown method: {method}")


# -------------------------
# Retrieval + Evaluation
# -------------------------
def run_eval_for_method(
    method: str,
    query_items: List[Tuple[str, str]],
    index_ids: List[str],
    index_cls: List[str],
    index_desc: List[Any],
    obj_root: Path,
    image_size: int,
    metric: str,
    aggregation: str,
    depth_rotation_set: str,
    limit_queries: Optional[int] = None,
    l2_normalize: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
      - global_macro: metrics (classes with >=1 successful query)
      - per_class: dict[class] -> metrics (successful queries only)
      - per_query: list of per-query records (includes failures marked with failed=True)
      - failed_queries: list of failed query records
    """
    if limit_queries is not None:
        query_items = query_items[: max(0, limit_queries)]

    # Precompute index counts per class for R, recall denominators, and cap_k
    index_count_by_class: Dict[str, int] = defaultdict(int)
    for c in index_cls:
        index_count_by_class[c] += 1

    engine = SimilarityEngine(metric=metric, aggregation=aggregation, depth_rotation_set=depth_rotation_set)
    renderer = Renderer(width=image_size, height=image_size)

    per_query_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []
    per_class_query_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    try:
        for qi, (q_fn, q_cl) in enumerate(query_items, start=1):
            t0 = time.time()

            try:
                q_desc = compute_query_descriptor(
                    method=method,
                    renderer=renderer,
                    image_size=image_size,
                    obj_root=obj_root,
                    filename=q_fn,
                    cls=q_cl,
                    l2_normalize=l2_normalize,
                )
            except Exception as e:
                dt = round(time.time() - t0, 4)
                fail = {
                    "query_filename": q_fn,
                    "query_class": q_cl,
                    "time_sec": dt,
                    "failed": True,
                    "error": str(e),
                }
                per_query_records.append(fail)
                failed_records.append(fail)

                if qi == 1 or qi % 10 == 0 or qi == len(query_items):
                    print(f"[{method.upper()}] SKIP query {qi}/{len(query_items)}: {q_fn} ({q_cl}) err={e}")
                continue

            # distances to all index models
            dists = np.empty(len(index_desc), dtype=np.float32)
            for j, idx_d in enumerate(index_desc):
                dists[j] = float(engine.compare(q_desc, idx_d).distance)

            order = np.argsort(dists)

            # rank top-20 classes
            ranked_classes = [index_cls[i] for i in order[:K_PRIMARY]]
            relevance = [1 if c == q_cl else 0 for c in ranked_classes]

            total_rel = int(index_count_by_class.get(q_cl, 0))
            cap_k = int(min(K_PRIMARY, total_rel))  # effective k when R<20

            # per-query metrics
            q_metrics: Dict[str, float] = {}

            for k in [1, 5, 10]:
                q_metrics[f"P@{k}"] = precision_at_k(relevance, k)
                q_metrics[f"AP@{k}"] = average_precision_at_k(relevance, k, total_rel)

            # cap_k is its own column
            q_metrics["cap_k"] = float(cap_k)

            # Keep names as @20, compute with k_eff when R<20
            k_eff = cap_k if cap_k > 0 else K_PRIMARY
            q_metrics["P@20"] = precision_at_k(relevance, k_eff)
            q_metrics["AP@20"] = average_precision_at_k(relevance, k_eff, total_rel)
            q_metrics["R@20"] = recall_at_k(relevance, k_eff, total_rel)

            rec = {
                "query_filename": q_fn,
                "query_class": q_cl,
                "time_sec": round(time.time() - t0, 4),
                "failed": False,
                **{k: float(v) for k, v in q_metrics.items()},
            }
            per_query_records.append(rec)
            per_class_query_metrics[q_cl].append(q_metrics)

            if qi % 10 == 0 or qi == 1 or qi == len(query_items):
                print(f"[{method.upper()}] Query {qi}/{len(query_items)} done in {rec['time_sec']}s")

    finally:
        try:
            renderer.close()
        except Exception:
            pass

    # Aggregate per-class (successful only)
    per_class: Dict[str, Dict[str, float]] = {}
    for cl, metrics_list in per_class_query_metrics.items():
        agg: Dict[str, float] = {}
        for key in metrics_list[0].keys():
            agg[key] = float(np.mean([m[key] for m in metrics_list]))
        agg["num_queries"] = float(len(metrics_list))
        agg["num_index"] = float(index_count_by_class.get(cl, 0))
        agg["cap_k_class"] = float(min(K_PRIMARY, int(index_count_by_class.get(cl, 0))))
        per_class[cl] = agg

    # Global MACRO (mean over classes with >=1 successful query)
    macro: Dict[str, float] = {}
    if per_class:
        keys = [
            "AP@1", "AP@5", "AP@10", "AP@20",
            "P@1", "P@5", "P@10", "P@20",
            "R@20",
        ]
        for key in keys:
            macro[key] = float(np.mean([per_class[c].get(key, 0.0) for c in per_class.keys()]))

        macro["cap_k"] = float(np.mean([per_class[c].get("cap_k_class", float(K_PRIMARY)) for c in per_class.keys()]))

    ok_records = [r for r in per_query_records if not r.get("failed", False)]

    return {
        "method": method,
        "k_primary": K_PRIMARY,
        "settings": {
            "metric": metric,
            "aggregation": aggregation,
            "depth_rotation_set": depth_rotation_set,
            "image_size": image_size,
            "l2_normalize": bool(l2_normalize),
        },
        "global_macro": macro,
        "per_class": per_class,
        "per_query": per_query_records,
        "failed_queries": failed_records,
        "summary": {
            "num_queries_total": int(len(query_items)),
            "num_queries_ok": int(len(ok_records)),
            "num_queries_failed": int(len(failed_records)),
        },
    }


# -------------------------
# Plotting / Report helpers
# -------------------------
def _collect_all_classes(lfd_res: Optional[Dict[str, Any]], depth_res: Optional[Dict[str, Any]]) -> List[str]:
    classes_set = set()
    if lfd_res is not None:
        classes_set |= set(lfd_res["per_class"].keys())
    if depth_res is not None:
        classes_set |= set(depth_res["per_class"].keys())

    def qcount(res: Optional[Dict[str, Any]], c: str) -> int:
        if res is None:
            return 0
        return int(res["per_class"].get(c, {}).get("num_queries", 0.0))

    classes = [c for c in classes_set if (qcount(lfd_res, c) > 0) or (qcount(depth_res, c) > 0)]
    return sorted(classes)


def _class_row(res: Optional[Dict[str, Any]], c: str) -> Dict[str, float]:
    if res is None:
        return {}
    return res["per_class"].get(c, {})


# -------------------------
# Figures
# -------------------------
def make_tables_figure(outdir: Path, lfd_res: Optional[Dict[str, Any]], depth_res: Optional[Dict[str, Any]]) -> Path:
    _set_plot_style()

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Retrieval Evaluation (Tables)", fontsize=14)

    methods = []
    if lfd_res is not None:
        methods.append(("LFD", lfd_res))
    if depth_res is not None:
        methods.append(("DEPTH", depth_res))
    if not methods:
        raise RuntimeError("No results to plot.")

    header = [
        "Method",
        "mAP@20 (macro)",
        "P@1 (macro)",
        "P@5 (macro)",
        "P@10 (macro)",
        "P@20 (macro)",
        "R@20 (macro)",
        "cap_k (macro)",
        "failed_q",
    ]
    rows = []
    for name, res in methods:
        macro = res.get("global_macro", {})
        failed_q = int(res.get("summary", {}).get("num_queries_failed", 0))
        rows.append([
            name,
            f"{macro.get('AP@20', 0.0):.4f}",
            f"{macro.get('P@1', 0.0):.4f}",
            f"{macro.get('P@5', 0.0):.4f}",
            f"{macro.get('P@10', 0.0):.4f}",
            f"{macro.get('P@20', 0.0):.4f}",
            f"{macro.get('R@20', 0.0):.4f}",
            f"{macro.get('cap_k', float(K_PRIMARY)):.2f}",
            str(failed_q),
        ])

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.axis("off")
    tab1 = ax1.table(cellText=rows, colLabels=header, loc="center")
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(9)
    tab1.scale(1, 1.5)
    ax1.set_title("Global Summary (MACRO only; successful queries only)", pad=12)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis("off")

    ref = lfd_res if lfd_res is not None else depth_res
    assert ref is not None

    classes_sorted = sorted(
        ref["per_class"].keys(),
        key=lambda c: ref["per_class"][c].get("num_queries", 0.0),
        reverse=True,
    )
    classes_show = classes_sorted[:15]

    per_class_header = [
        "Class", "Q", "R(index)", "cap_k",
        "mAP@20 (LFD)", "mAP@20 (DEPTH)",
        "P@20 (LFD)", "P@20 (DEPTH)",
        "R@20 (LFD)", "R@20 (DEPTH)",
    ]
    per_class_rows = []
    for c in classes_show:
        l = _class_row(lfd_res, c)
        d = _class_row(depth_res, c)
        qn = int(max(l.get("num_queries", 0.0), d.get("num_queries", 0.0)))
        R = int(max(l.get("num_index", 0.0), d.get("num_index", 0.0)))
        cap_k = int(min(K_PRIMARY, R))
        per_class_rows.append([
            c,
            str(qn),
            str(R),
            str(cap_k),
            f"{float(l.get('AP@20', 0.0)):.4f}" if lfd_res else "—",
            f"{float(d.get('AP@20', 0.0)):.4f}" if depth_res else "—",
            f"{float(l.get('P@20', 0.0)):.4f}" if lfd_res else "—",
            f"{float(d.get('P@20', 0.0)):.4f}" if depth_res else "—",
            f"{float(l.get('R@20', 0.0)):.4f}" if lfd_res else "—",
            f"{float(d.get('R@20', 0.0)):.4f}" if depth_res else "—",
        ])

    tab2 = ax2.table(cellText=per_class_rows, colLabels=per_class_header, loc="center")
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(8.5)
    tab2.scale(1, 1.35)
    ax2.set_title("Per-Class Preview (Top 15 by query count). Full all-classes table is in retrieval_report.pdf", pad=12)

    outpath = outdir / "figure_tables.png"
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def make_graphs_figure(outdir: Path, lfd_res: Optional[Dict[str, Any]], depth_res: Optional[Dict[str, Any]]) -> Path:
    _set_plot_style()

    if lfd_res is None and depth_res is None:
        raise RuntimeError("No results to plot.")

    classes = _collect_all_classes(lfd_res, depth_res)

    def depth_key(c: str) -> float:
        d = _class_row(depth_res, c)
        return float(d.get("AP@20", 0.0))

    classes = sorted(classes, key=lambda c: (-depth_key(c), c))

    h = max(10.0, 0.32 * len(classes) + 3.0)
    fig = plt.figure(figsize=(16, h))
    fig.suptitle("Retrieval Evaluation (Graphs) — LFD vs Depth", fontsize=14)

    ax1 = fig.add_subplot(2, 1, 1)
    labels, macro_map20 = [], []

    if lfd_res is not None:
        labels.append("LFD")
        macro_map20.append(lfd_res["global_macro"].get("AP@20", 0.0))
    if depth_res is not None:
        labels.append("DEPTH")
        macro_map20.append(depth_res["global_macro"].get("AP@20", 0.0))

    x = np.arange(len(labels))
    ax1.bar(x, macro_map20, width=0.5, label="mAP@20 (macro)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("mAP@20")
    ax1.set_title("Global mAP@20 (MACRO only; successful queries only)")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    y = np.arange(len(classes))
    bar_h = 0.42

    lfd_vals = [float(_class_row(lfd_res, c).get("AP@20", 0.0)) if lfd_res else 0.0 for c in classes]
    dep_vals = [float(_class_row(depth_res, c).get("AP@20", 0.0)) if depth_res else 0.0 for c in classes]

    ax2.barh(y - bar_h / 2, lfd_vals, height=bar_h, label="mAP@20 (LFD)")
    ax2.barh(y + bar_h / 2, dep_vals, height=bar_h, label="mAP@20 (DEPTH)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(classes, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("mAP@20")
    ax2.set_title("Per-Class mAP@20 (ALL classes; sorted by mAP@20 (DEPTH) desc)")
    ax2.legend()

    fig.subplots_adjust(left=0.30)
    outpath = outdir / "figure_graphs.png"
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def _draw_table_page(
    pdf: PdfPages,
    title: str,
    col_labels: List[str],
    rows: List[List[str]],
    fontsize: int = 9,
    col_widths: Optional[List[float]] = None,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape-ish
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=13)

    tab = ax.table(cellText=rows, colLabels=col_labels, loc="center", colWidths=col_widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1, 1.2)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_retrieval_report_pdf(outdir: Path, lfd_res: Optional[Dict[str, Any]], depth_res: Optional[Dict[str, Any]]) -> Path:
    _set_plot_style()

    report_path = outdir / "retrieval_report.pdf"
    with PdfPages(report_path) as pdf:
        methods = []
        if lfd_res is not None:
            methods.append(("LFD", lfd_res))
        if depth_res is not None:
            methods.append(("DEPTH", depth_res))

        header = [
            "Method",
            "mAP@20 (macro)",
            "P@1 (macro)",
            "P@5 (macro)",
            "P@10 (macro)",
            "P@20 (macro)",
            "R@20 (macro)",
            "cap_k (macro)",
            "failed_q",
        ]
        rows = []
        for name, res in methods:
            macro = res.get("global_macro", {})
            failed_q = int(res.get("summary", {}).get("num_queries_failed", 0))
            rows.append([
                name,
                f"{macro.get('AP@20', 0.0):.4f}",
                f"{macro.get('P@1', 0.0):.4f}",
                f"{macro.get('P@5', 0.0):.4f}",
                f"{macro.get('P@10', 0.0):.4f}",
                f"{macro.get('P@20', 0.0):.4f}",
                f"{macro.get('R@20', 0.0):.4f}",
                f"{macro.get('cap_k', float(K_PRIMARY)):.2f}",
                str(failed_q),
            ])

        _draw_table_page(
            pdf,
            title="Retrieval Evaluation — Executive Summary (MACRO only; successful queries only)",
            col_labels=header,
            rows=rows,
            fontsize=10,
        )

        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Global mAP@20 (MACRO)", pad=12, fontsize=14)

        labels, vals = [], []
        if lfd_res is not None:
            labels.append("LFD")
            vals.append(lfd_res["global_macro"].get("AP@20", 0.0))
        if depth_res is not None:
            labels.append("DEPTH")
            vals.append(depth_res["global_macro"].get("AP@20", 0.0))

        x = np.arange(len(labels))
        ax.bar(x, vals, width=0.55, label="mAP@20 (macro)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("mAP@20")
        ax.legend()

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        classes = _collect_all_classes(lfd_res, depth_res)

        def depth_key(c: str) -> float:
            return float(_class_row(depth_res, c).get("AP@20", 0.0))

        classes = sorted(classes, key=lambda c: (-depth_key(c), c))

        h = max(8.27, 0.25 * len(classes) + 3.0)
        fig = plt.figure(figsize=(11.69, h))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Per-Class mAP@20 (ALL classes; sorted by mAP@20 (DEPTH) desc)", pad=12, fontsize=14)

        y = np.arange(len(classes))
        bar_h = 0.42
        lfd_vals = [float(_class_row(lfd_res, c).get("AP@20", 0.0)) if lfd_res else 0.0 for c in classes]
        dep_vals = [float(_class_row(depth_res, c).get("AP@20", 0.0)) if depth_res else 0.0 for c in classes]

        ax.barh(y - bar_h / 2, lfd_vals, height=bar_h, label="mAP@20 (LFD)")
        ax.barh(y + bar_h / 2, dep_vals, height=bar_h, label="mAP@20 (DEPTH)")
        ax.set_yticks(y)
        ax.set_yticklabels(classes, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("mAP@20")
        ax.legend()

        fig.subplots_adjust(left=0.33)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        col_labels = [
            "Class", "Q", "R(index)", "cap_k",
            "mAP@20 (LFD)", "mAP@20 (DEPTH)",
            "P@20 (LFD)", "P@20 (DEPTH)",
            "R@20 (LFD)", "R@20 (DEPTH)",
        ]

        table_rows: List[List[str]] = []
        for c in classes:
            l = _class_row(lfd_res, c)
            d = _class_row(depth_res, c)
            qn = int(max(l.get("num_queries", 0.0), d.get("num_queries", 0.0)))
            R = int(max(l.get("num_index", 0.0), d.get("num_index", 0.0)))
            cap_k = int(min(K_PRIMARY, R))
            table_rows.append([
                c,
                str(qn),
                str(R),
                str(cap_k),
                f"{float(l.get('AP@20', 0.0)):.4f}" if lfd_res else "—",
                f"{float(d.get('AP@20', 0.0)):.4f}" if depth_res else "—",
                f"{float(l.get('P@20', 0.0)):.4f}" if lfd_res else "—",
                f"{float(d.get('P@20', 0.0)):.4f}" if depth_res else "—",
                f"{float(l.get('R@20', 0.0)):.4f}" if lfd_res else "—",
                f"{float(d.get('R@20', 0.0)):.4f}" if depth_res else "—",
            ])

        rows_per_page = 30
        n_pages = int(math.ceil(len(table_rows) / rows_per_page)) if table_rows else 1
        for pi in range(n_pages):
            start = pi * rows_per_page
            end = min(len(table_rows), (pi + 1) * rows_per_page)
            page_rows = table_rows[start:end]
            _draw_table_page(
                pdf,
                title=f"Per-Class Metrics (ALL classes) — Page {pi+1}/{n_pages}",
                col_labels=col_labels,
                rows=page_rows,
                fontsize=9,
            )

    return report_path


# -------------------------
# Save outputs
# -------------------------
def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_per_query_csv(path: Path, method: str, per_query: List[Dict[str, Any]]) -> None:
    ok = [r for r in per_query if not r.get("failed", False)]
    if not ok:
        return
    cols = [
        "method", "query_filename", "query_class",
        "cap_k",
        "AP@1", "AP@5", "AP@10", "AP@20",
        "P@1", "P@5", "P@10", "P@20",
        "R@20",
        "time_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ok:
            row = {"method": method, **r}
            w.writerow({c: row.get(c, "") for c in cols})


def save_failed_queries_csv(path: Path, method: str, per_query: List[Dict[str, Any]]) -> None:
    failed = [r for r in per_query if r.get("failed", False)]
    if not failed:
        return
    cols = ["method", "query_filename", "query_class", "time_sec", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in failed:
            row = {"method": method, **r}
            w.writerow({c: row.get(c, "") for c in cols})


def save_per_class_csv(path: Path, lfd_res: Optional[Dict[str, Any]], depth_res: Optional[Dict[str, Any]]) -> None:
    classes = set()
    if lfd_res is not None:
        classes |= set(lfd_res["per_class"].keys())
    if depth_res is not None:
        classes |= set(depth_res["per_class"].keys())
    classes = sorted(classes)

    cols = [
        "class",
        "num_queries",
        "num_index",
        "cap_k_class",
        "lfd_mAP@20",
        "depth_mAP@20",
        "lfd_P@20",
        "depth_P@20",
        "lfd_R@20",
        "depth_R@20",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in classes:
            l = lfd_res["per_class"].get(c, {}) if lfd_res else {}
            d = depth_res["per_class"].get(c, {}) if depth_res else {}
            num_q = int(max(l.get("num_queries", 0.0), d.get("num_queries", 0.0)))
            num_i = int(max(l.get("num_index", 0.0), d.get("num_index", 0.0)))
            cap_k = int(min(K_PRIMARY, num_i))

            row = {
                "class": c,
                "num_queries": num_q,
                "num_index": num_i,
                "cap_k_class": cap_k,
                "lfd_mAP@20": float(l.get("AP@20", 0.0)) if lfd_res else "",
                "depth_mAP@20": float(d.get("AP@20", 0.0)) if depth_res else "",
                "lfd_P@20": float(l.get("P@20", 0.0)) if lfd_res else "",
                "depth_P@20": float(d.get("P@20", 0.0)) if depth_res else "",
                "lfd_R@20": float(l.get("R@20", 0.0)) if lfd_res else "",
                "depth_R@20": float(d.get("R@20", 0.0)) if depth_res else "",
            }
            w.writerow(row)


def save_raw_numbers_json(path: Path, lfd_res: Optional[Dict[str, Any]], depth_res: Optional[Dict[str, Any]]) -> None:
    payload: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "k_primary": K_PRIMARY,
        "notes": {
            "cap_definition": "cap_k = min(20, R) where R is #relevant items in index for the query class.",
            "metric_labeling": "All @20 metrics are computed with cap_k when R<20, but columns remain named @20. cap_k is reported.",
            "global": "Only MACRO global metrics are reported (equal weight per class with >=1 successful query).",
        },
        "methods": {},
    }
    if lfd_res is not None:
        payload["methods"]["lfd"] = {
            "settings": lfd_res.get("settings", {}),
            "summary": lfd_res.get("summary", {}),
            "global_macro": lfd_res.get("global_macro", {}),
            "per_class": lfd_res.get("per_class", {}),
        }
    if depth_res is not None:
        payload["methods"]["depth"] = {
            "settings": depth_res.get("settings", {}),
            "summary": depth_res.get("summary", {}),
            "global_macro": depth_res.get("global_macro", {}),
            "per_class": depth_res.get("per_class", {}),
        }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run retrieval evaluation on query split (LFD vs Depth).")
    ap.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)

    ap.add_argument("--obj-root", default=str(OBJ_ROOT_REL))
    ap.add_argument("--index-csv", default=str(INDEX_CSV_REL))
    ap.add_argument("--query-csv", default=str(QUERY_CSV_REL))

    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)

    ap.add_argument("--method", choices=["lfd", "depth"], default=None, help="Run one method only.")
    ap.add_argument("--both", action="store_true", help="Run both methods and compare.")
    ap.add_argument("--limit-queries", type=int, default=None, help="Debug: evaluate only first N queries.")

    ap.add_argument("--metric", choices=["l2", "l1", "cosine"], default="l2")
    ap.add_argument("--aggregation", choices=["mean", "sum"], default="mean")

    ap.add_argument("--depth-rotation-set", choices=["yaw12", "yaw24", "grid12", "grid24"], default="grid24")

    ap.add_argument(
        "--l2-normalize",
        action="store_true",
        help="Apply row-wise L2-normalization to query features before comparing (recommended if index is L2-normalized).",
    )

    ap.add_argument("--outdir", default="out/eval", help="Output directory for CSV/JSON/figures.")
    args = ap.parse_args()

    if not args.both and args.method is None:
        raise SystemExit("Use either --both or --method {lfd,depth}")

    root = PROJECT_ROOT
    obj_root = (root / args.obj_root).resolve()
    index_csv = (root / args.index_csv).resolve()
    query_csv = (root / args.query_csv).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Project root:", root)
    print("[INFO] OBJ root     :", obj_root)
    print("[INFO] INDEX CSV    :", index_csv)
    print("[INFO] QUERY CSV    :", query_csv)
    print("[INFO] Output dir   :", outdir)
    print("[INFO] k primary    :", K_PRIMARY)
    print("[INFO] image_size   :", args.image_size)
    print("[INFO] depth_rotset :", args.depth_rotation_set)
    print("[INFO] l2_normalize :", "ON" if args.l2_normalize else "OFF")
    if _HAS_SEABORN:
        print("[INFO] seaborn      : available (using nicer style)")
    else:
        print("[INFO] seaborn      : not available (matplotlib style)")

    query_items = read_split_csv(query_csv)

    client = connect_mongo(args.mongo_uri)
    coll: Collection = client[args.db][args.collection]

    lfd_res = None
    depth_res = None

    try:
        if args.both or args.method == "lfd":
            print("\n[LOAD] Index descriptors (LFD) from MongoDB...")
            idx_ids, idx_cls, idx_desc = load_index_descriptors(coll, method="lfd")
            print(f"[OK] Loaded {len(idx_desc)} index descriptors for LFD")

            print("[EVAL] Running LFD retrieval evaluation...")
            lfd_res = run_eval_for_method(
                method="lfd",
                query_items=query_items,
                index_ids=idx_ids,
                index_cls=idx_cls,
                index_desc=idx_desc,
                obj_root=obj_root,
                image_size=args.image_size,
                metric=args.metric,
                aggregation=args.aggregation,
                depth_rotation_set=args.depth_rotation_set,
                limit_queries=args.limit_queries,
                l2_normalize=args.l2_normalize,
            )
            save_json(outdir / "lfd_metrics.json", lfd_res)
            save_per_query_csv(outdir / "per_query_metrics_lfd.csv", "lfd", lfd_res["per_query"])
            save_failed_queries_csv(outdir / "failed_queries_lfd.csv", "lfd", lfd_res["per_query"])
            print("[OK] Saved LFD results")

        if args.both or args.method == "depth":
            print("\n[LOAD] Index descriptors (DEPTH) from MongoDB...")
            idx_ids, idx_cls, idx_desc = load_index_descriptors(coll, method="depth")
            print(f"[OK] Loaded {len(idx_desc)} index descriptors for DEPTH")

            print("[EVAL] Running DEPTH retrieval evaluation...")
            depth_res = run_eval_for_method(
                method="depth",
                query_items=query_items,
                index_ids=idx_ids,
                index_cls=idx_cls,
                index_desc=idx_desc,
                obj_root=obj_root,
                image_size=args.image_size,
                metric=args.metric,
                aggregation=args.aggregation,
                depth_rotation_set=args.depth_rotation_set,
                limit_queries=args.limit_queries,
                l2_normalize=args.l2_normalize,
            )
            save_json(outdir / "depth_metrics.json", depth_res)
            save_per_query_csv(outdir / "per_query_metrics_depth.csv", "depth", depth_res["per_query"])
            save_failed_queries_csv(outdir / "failed_queries_depth.csv", "depth", depth_res["per_query"])
            print("[OK] Saved DEPTH results")

        save_per_class_csv(outdir / "per_class_metrics.csv", lfd_res, depth_res)
        save_raw_numbers_json(outdir / "raw_numbers.json", lfd_res, depth_res)

        tables_path = make_tables_figure(outdir, lfd_res, depth_res)
        graphs_path = make_graphs_figure(outdir, lfd_res, depth_res)
        report_path = make_retrieval_report_pdf(outdir, lfd_res, depth_res)

        print("\n[DONE ✅]")
        if lfd_res:
            print("  LFD mAP@20 (macro)   :", f"{lfd_res['global_macro'].get('AP@20', 0.0):.4f}")
            print("  LFD failed queries   :", lfd_res.get("summary", {}).get("num_queries_failed", 0))
        if depth_res:
            print("  DEPTH mAP@20 (macro) :", f"{depth_res['global_macro'].get('AP@20', 0.0):.4f}")
            print("  DEPTH failed queries :", depth_res.get("summary", {}).get("num_queries_failed", 0))

        print("\n[FILES]")
        print(" ", outdir / "per_class_metrics.csv")
        print(" ", outdir / "raw_numbers.json")
        if lfd_res:
            print(" ", outdir / "lfd_metrics.json")
            print(" ", outdir / "per_query_metrics_lfd.csv")
            print(" ", outdir / "failed_queries_lfd.csv")
        if depth_res:
            print(" ", outdir / "depth_metrics.json")
            print(" ", outdir / "per_query_metrics_depth.csv")
            print(" ", outdir / "failed_queries_depth.csv")
        print(" ", tables_path)
        print(" ", graphs_path)
        print(" ", report_path)

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
