#!/usr/bin/env python3
"""
03_init_mongodb.py

Initialize MongoDB for the ObjectLens project:
- Connect to local MongoDB
- Create database + collection
- Create DB indexes for fast lookup

This script does NOT insert models and does NOT compute descriptors.
That is handled by: 04_index_models.py

Run:
    python scripts/03_init_mongodb.py

Defaults:
- Mongo URI: mongodb://localhost:27017
- DB name : objectlens
- Coll    : models
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "objectlens"
COLLECTION_NAME = "models"

# How long we wait to confirm MongoDB is reachable
SERVER_TIMEOUT_MS = 4000


def main() -> None:
    print(f"[INFO] Connecting to MongoDB: {MONGO_URI}")

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=SERVER_TIMEOUT_MS)

    # Force a connection check now (otherwise MongoClient is lazy)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        raise RuntimeError(
            "Could not connect to MongoDB.\n"
            "Make sure MongoDB is running locally and listening on 27017.\n"
            "On Windows (typical): start the MongoDB service.\n"
            "On Linux: sudo systemctl start mongod\n"
        ) from e

    db = client[DB_NAME]

    # Create collection if it doesn't exist
    existing = db.list_collection_names()
    if COLLECTION_NAME in existing:
        print(f"[SKIP] Collection already exists: {DB_NAME}.{COLLECTION_NAME}")
        coll = db[COLLECTION_NAME]
    else:
        coll = db.create_collection(COLLECTION_NAME)
        print(f"[OK] Created collection: {DB_NAME}.{COLLECTION_NAME}")

    # Create DB indexes (lookup structures)
    # _id index exists by default.
    # We add indexes that we will frequently query by:
    # - class: filter / evaluation / analysis
    # - lfd.preset and depth.preset: useful to find docs already indexed with a method
    # - created_at: useful for debugging (optional)
    print("[INFO] Creating DB indexes (safe to re-run)...")

    # Fast retrieval by class label
    coll.create_index([("class", 1)], name="idx_class")

    # Quickly find docs that have descriptor blocks present
    coll.create_index([("lfd.preset", 1)], name="idx_lfd_preset")
    coll.create_index([("depth.preset", 1)], name="idx_depth_preset")

    # Optional debug: time-based queries
    coll.create_index([("created_at", 1)], name="idx_created_at")

    print("[OK] Indexes created/ensured ✅")

    # Print a tiny summary
    print("\n[SUMMARY]")
    print(f"  DB         : {DB_NAME}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Indexes    : {[i['name'] for i in coll.list_indexes()]}")
    print("\nNext: python scripts/04_index_models.py")

    client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
