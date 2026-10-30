#!/usr/bin/env python3
"""Remove scenarios present in a filtered dataset folder from the main data JSON files.

Usage:
  python scripts/remove_filtered_from_main.py --filtered-dir data/filtered_traps_20260508_20260512 \
      --data-dir data --backup

This will back up any modified files to `<filename>.bak.<timestamp>` before overwriting.
"""
import argparse
import json
import os
import time
from pathlib import Path


def load_filtered_ids(filtered_dir: Path) -> set:
    ids = set()
    if not filtered_dir.exists():
        return ids
    for p in filtered_dir.glob("*.json"):
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    sid = item.get("scenario_id") if isinstance(item, dict) else None
                    if sid:
                        ids.add(sid)
        except Exception:
            # ignore non-JSON or unexpected files
            continue
    return ids


def process_main_files(data_dir: Path, filtered_ids: set, backup: bool = True) -> None:
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        return

    json_files = sorted([p for p in data_dir.glob("*.json")])
    timestamp = int(time.time())

    for p in json_files:
        # skip any files inside filtered folders
        if "filtered_traps" in str(p.parent):
            continue

        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {p}: failed to read ({e})")
            continue

        if not isinstance(data, list):
            print(f"Skipping {p}: not a list of scenarios")
            continue

        original_count = len(data)
        new_data = [s for s in data if not (isinstance(s, dict) and s.get("scenario_id") in filtered_ids)]
        removed = original_count - len(new_data)
        if removed == 0:
            print(f"No filtered scenarios found in {p} (kept {original_count})")
            continue

        if backup:
            bak = p.with_name(p.name + f".bak.{timestamp}")
            with open(bak, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"Backed up {p} → {bak}")

        with open(p, "w") as fh:
            json.dump(new_data, fh, indent=2)
        print(f"Updated {p}: removed {removed} scenarios (now {len(new_data)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-dir", default="data/filtered_traps_20260508_20260512")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    args = parser.parse_args()

    filtered_dir = Path(args.filtered_dir)
    data_dir = Path(args.data_dir)

    filtered_ids = load_filtered_ids(filtered_dir)
    if not filtered_ids:
        print(f"No filtered scenarios found in {filtered_dir}. Nothing to do.")
        return

    print(f"Found {len(filtered_ids)} filtered scenario ids. Removing from main data files in {data_dir}...")
    process_main_files(data_dir, filtered_ids, backup=args.backup)


if __name__ == "__main__":
    main()
