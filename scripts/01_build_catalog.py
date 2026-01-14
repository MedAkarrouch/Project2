#!/usr/bin/env python3
"""
01_build_catalog.py

Builds a clean catalog.csv from the IPET groundtruth file.

Input (auto-detected under ./data/raw):
- Database_vessels_sorted_per_id_and_groundtruth.txt
  Format per line: id,filename,class

Outputs:
- ./data/splits/catalog.csv        columns: filename,class
- ./data/splits/catalog_stats.csv  columns: class,count  (no split yet)

Rules:
- Drop the id column (we don't keep ids)
- Exclude class "Other"
- Keep class "Abstract"
- Deduplicate by filename (first occurrence wins)
- Robust CSV parsing (comma-separated)
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List


EXCLUDE_CLASSES = {"Other"}  # keep Abstract
GROUNDTRUTH_NAME = "Database_vessels_sorted_per_id_and_groundtruth.txt"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_groundtruth_file(raw_dir: Path) -> Path:
    candidates = list(raw_dir.glob(GROUNDTRUTH_NAME))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{GROUNDTRUTH_NAME}' under: {raw_dir}\n"
            f"Expected: {raw_dir / GROUNDTRUTH_NAME}"
        )
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


@dataclass(frozen=True)
class CatalogRow:
    filename: str
    label: str


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    root = project_root()
    raw_dir = root / "data" / "raw"
    out_dir = root / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = find_groundtruth_file(raw_dir)
    print(f"[INFO] Using groundtruth: {gt_path}")

    seen = set()
    catalog: List[CatalogRow] = []

    excluded = 0
    skipped_dupes = 0
    skipped_bad = 0

    with gt_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            # Expected row: [id, filename, class]
            if len(row) < 3:
                skipped_bad += 1
                continue

            _id = row[0].strip()
            filename = row[1].strip()
            label = row[2].strip()

            if not filename or not label:
                skipped_bad += 1
                continue

            if label in EXCLUDE_CLASSES:
                excluded += 1
                continue

            if filename in seen:
                skipped_dupes += 1
                continue

            seen.add(filename)
            catalog.append(CatalogRow(filename=filename, label=label))

    if not catalog:
        raise RuntimeError("Catalog is empty. Check parsing and the groundtruth file content.")

    # Stable order
    catalog.sort(key=lambda r: (r.label.lower(), r.filename.lower()))

    # Write catalog.csv
    catalog_csv = out_dir / "catalog.csv"
    catalog_rows = [[r.filename, r.label] for r in catalog]
    write_csv(catalog_csv, ["filename", "class"], catalog_rows)

    # Write catalog_stats.csv
    counts = Counter(r.label for r in catalog)
    stats_csv = out_dir / "catalog_stats.csv"
    stats_rows = [[cls, str(n)] for cls, n in sorted(counts.items(), key=lambda x: (-x[1], x[0].lower()))]
    write_csv(stats_csv, ["class", "count"], stats_rows)

    total_objs = sum(counts.values())

    print("\n[OK] Catalog built ✅")
    print(f"- Output: {catalog_csv}")
    print(f"- Stats : {stats_csv}\n")

    print("[SUMMARY]")
    print(f"  Classes kept       : {len(counts)}")
    print(f"  Objects kept       : {total_objs}")
    print(f"  Excluded (Other)   : {excluded}")
    print(f"  Skipped duplicates : {skipped_dupes}")
    print(f"  Skipped bad lines  : {skipped_bad}")
    print("\nNext: run -> python scripts/02_split_catalog.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
