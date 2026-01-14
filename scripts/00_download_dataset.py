#!/usr/bin/env python3
"""
One-command downloader + extractor for the IPET 3D Pottery Benchmark Dataset v1.

What it does (no args needed):
1) Downloads the DATASET ZIP into:   ./data/raw/
2) Extracts it into:                 ./data/raw/
3) Deletes the ZIP after successful extraction
4) Safe to re-run (skips if already extracted)

Run (from scripts/):
    python 00_download_dataset.py
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import requests


DATASET_URL = "https://www.ipet.gr/~akoutsou/benchmark/dataset/3DPotteryDataset_v_1.zip"


def _human_mb(n_bytes: int) -> str:
    """
    Convert a byte count into a human-readable size string (e.g. 1.2 MB).

    Args:
        n_bytes: The number of bytes to convert.

    Returns:
        A string like "1.2 MB".

    Example:
        >>> _human_mb(1234567) -> '1.2 MB'
    """
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def _project_root() -> Path:
    # scripts/00_download_dataset.py -> project root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def download_file(
    url: str,
    out_path: Path,
    chunk_size: int = 1024 * 1024, 
    timeout: int = 60
    ) -> None:
    """
    Downloads a file from a given URL to a local path.

    Args:
        url: The URL of the file to download.
        out_path: The local path where the file will be saved.
        chunk_size: The size of each chunk (in bytes) to write to disk.
        timeout: The timeout (in seconds) for the HTTP request.

    Returns:
        None

    Example:
        >>> download_file("https://example.com/file.zip", Path("file.zip"))
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists and looks non-empty, skip.
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] Already downloaded: {out_path} ({_human_mb(out_path.stat().st_size)})")
        return

    print(f"[DL] {url}")
    print(f"     -> {out_path}")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        total = r.headers.get("Content-Length")
        total = int(total) if total is not None else None

        downloaded = 0
        start = time.time()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                elapsed = max(time.time() - start, 1e-6)
                speed = downloaded / elapsed / (1024 * 1024)

                if total:
                    pct = downloaded / total * 100
                    print(
                        f"\r     {pct:6.2f}%  ({_human_mb(downloaded)}/{_human_mb(total)})  {speed:.2f} MB/s",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r     {_human_mb(downloaded)}  {speed:.2f} MB/s", end="", flush=True)

    print("\n[OK] Download complete.")


def verify_zip_is_readable(zip_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupted zip entry detected: {bad}")
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Bad ZIP file: {zip_path}") from e


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        raise FileNotFoundError(f"Zip not found or empty: {zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    # If raw already contains expected content, skip extraction.
    # We check for the "3D Models" folder somewhere under extract_dir after previous runs.
    already = any(p.name == "3D Models" and p.is_dir() for p in extract_dir.rglob("3D Models"))
    if already:
        print(f"[SKIP] Already extracted (found '3D Models' under): {extract_dir}")
        return

    print(f"[EXTRACT] {zip_path} -> {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Zip-slip protection
        base = extract_dir.resolve()
        for member in zf.infolist():
            target_path = (extract_dir / member.filename).resolve()
            if not str(target_path).startswith(str(base)):
                raise RuntimeError(f"Blocked suspicious path in zip: {member.filename}")

        zf.extractall(path=extract_dir)

    print("[OK] Extraction complete.")


def delete_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"[CLEAN] Deleted: {path}")


def main() -> None:
    root = _project_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_zip = raw_dir / "3DPotteryDataset_v_1.zip"

    # 1) Download dataset ZIP
    download_file(DATASET_URL, dataset_zip)

    # 2) Verify ZIP integrity
    print("[CHECK] Verifying ZIP integrity...")
    verify_zip_is_readable(dataset_zip)
    print("[OK] ZIP looks good.")

    # 3) Extract into data/raw/
    safe_extract_zip(dataset_zip, raw_dir)

    # 4) Delete ZIP
    delete_file(dataset_zip)

    print("\nAll done ✅")
    print(f"- Raw dataset extracted under: {raw_dir}")
    print("Tip: your groundtruth txt/xls files should be inside data/raw/ (or a subfolder).")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"\n[ERROR] HTTP error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
