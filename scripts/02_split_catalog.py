#!/usr/bin/env python3
"""
02_split_catalog.py

Splits the catalog into:
- index.csv  (gallery / indexed set)
- query.csv  (probe / query set)

Input:
- ./data/splits/catalog.csv   columns: filename,class

Output:
- ./data/splits/index.csv          columns: filename,class
- ./data/splits/query.csv          columns: filename,class
- ./data/splits/split_summary.csv  per-class + totals

Rules (per class):
- If n >= 10: do 80% index / 20% query (with at least 1 query and at least 1 index)
- If 2 <= n < 10: index-only (query=0)
- If n == 1: index-only (query=0)

Notes:
- Deterministic split using a fixed seed (default 42).
- No file copying; split is by filename/class only.

Run:
    python scripts/02_split_catalog.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import random


MIN_CLASS_FOR_SPLIT = 10
QUERY_RATIO = 0.20
SEED = 42


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_catalog(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    rows: List[Tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "class" not in reader.fieldnames:
            raise ValueError(f"Catalog must have headers: filename,class. Got: {reader.fieldnames}")
        for r in reader:
            fn = (r.get("filename") or "").strip()
            cl = (r.get("class") or "").strip()
            if fn and cl:
                rows.append((fn, cl))
    if not rows:
        raise RuntimeError("Catalog is empty.")
    return rows


def write_rows(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@dataclass
class ClassSplit:
    cls: str
    total: int
    index_count: int
    query_count: int
    note: str


def compute_split_counts(n: int) -> Tuple[int, int, str]:
    """
    Returns: (index_count, query_count, note)
    """
    if n >= MIN_CLASS_FOR_SPLIT:
        q = int(round(n * QUERY_RATIO))
        q = max(q, 1)
        q = min(q, n - 1)  # ensure at least 1 in index
        i = n - q
        return i, q, "80/20"
    else:
        # index-only
        return n, 0, f"index-only (n<{MIN_CLASS_FOR_SPLIT})"


def main() -> None:
    root = project_root()
    split_dir = root / "data" / "splits"
    catalog_path = split_dir / "catalog.csv"

    out_index = split_dir / "index.csv"
    out_query = split_dir / "query.csv"
    out_summary = split_dir / "split_summary.csv"

    rows = read_catalog(catalog_path)

    # Group by class
    by_class: Dict[str, List[str]] = {}
    for fn, cl in rows:
        by_class.setdefault(cl, []).append(fn)

    rng = random.Random(SEED)

    index_rows: List[List[str]] = []
    query_rows: List[List[str]] = []
    summary_rows: List[ClassSplit] = []

    total_index = 0
    total_query = 0

    for cls in sorted(by_class.keys(), key=lambda x: x.lower()):
        files = list(by_class[cls])
        n = len(files)

        # Deterministic per-class shuffle
        rng.shuffle(files)

        index_n, query_n, note = compute_split_counts(n)

        idx_files = files[:index_n]
        qry_files = files[index_n:index_n + query_n]

        # Sanity checks
        if len(idx_files) != index_n or len(qry_files) != query_n:
            raise RuntimeError(f"Split mismatch for class '{cls}' (n={n}).")

        for fn in idx_files:
            index_rows.append([fn, cls])
        for fn in qry_files:
            query_rows.append([fn, cls])

        total_index += index_n
        total_query += query_n

        summary_rows.append(
            ClassSplit(
                cls=cls,
                total=n,
                index_count=index_n,
                query_count=query_n,
                note=note,
            )
        )

    # Stable ordering in outputs
    index_rows.sort(key=lambda r: (r[1].lower(), r[0].lower()))
    query_rows.sort(key=lambda r: (r[1].lower(), r[0].lower()))

    write_rows(out_index, ["filename", "class"], index_rows)
    write_rows(out_query, ["filename", "class"], query_rows)

    # Build summary CSV
    # Include per-class pct + totals row
    summary_table: List[List[str]] = []
    for s in sorted(summary_rows, key=lambda x: x.cls.lower()):
        idx_pct = (s.index_count / s.total * 100.0) if s.total else 0.0
        qry_pct = (s.query_count / s.total * 100.0) if s.total else 0.0
        summary_table.append([
            s.cls,
            str(s.total),
            str(s.index_count),
            str(s.query_count),
            f"{idx_pct:.2f}",
            f"{qry_pct:.2f}",
            s.note,
        ])

    total_classes = len(summary_rows)
    total_objects = total_index + total_query
    summary_table.append([
        "__TOTAL__",
        str(total_objects),
        str(total_index),
        str(total_query),
        f"{(total_index/total_objects*100.0) if total_objects else 0.0:.2f}",
        f"{(total_query/total_objects*100.0) if total_objects else 0.0:.2f}",
        f"classes={total_classes}, seed={SEED}, min_class_for_split={MIN_CLASS_FOR_SPLIT}",
    ])

    write_rows(
        out_summary,
        ["class", "total", "index_count", "query_count", "index_pct", "query_pct", "note"],
        summary_table,
    )

    print("\n[OK] Split complete ✅")
    print(f"- Index set : {out_index}  (rows={len(index_rows)})")
    print(f"- Query set : {out_query}  (rows={len(query_rows)})")
    print(f"- Summary   : {out_summary}")
    print("\n[SUMMARY]")
    print(f"  Classes total : {total_classes}")
    print(f"  Index objects : {total_index}")
    print(f"  Query objects : {total_query}")
    print(f"  Total objects : {total_objects}")
    print(f"  Rule: n>={MIN_CLASS_FOR_SPLIT} => 80/20, else index-only")
    print(f"  Seed: {SEED}\n")

    print("Next: python scripts/03_init_mongodb.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
