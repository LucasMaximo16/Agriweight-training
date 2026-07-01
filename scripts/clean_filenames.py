#!/usr/bin/env python3
"""
Strips Roboflow's export suffix (`_jpg.rf.<hash>`) from image/label file
names in data/segmentation/raw/, restoring the original `cow_<id>_angle<n>`
naming used in data/labels/metadata.csv. Roboflow appends that hash on every
export, so re-run this after importing a fresh export.

Usage:
    python scripts/clean_filenames.py [--dry-run]
"""
import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "segmentation" / "raw"

ROBOFLOW_SUFFIX = re.compile(r"^(cow_\d+_angle\d+)_jpg\.rf\.[0-9a-f]+(\.\w+)$")


def planned_renames(directory: Path) -> list[tuple[Path, Path]]:
    renames = []
    for path in sorted(directory.iterdir()):
        match = ROBOFLOW_SUFFIX.match(path.name)
        if not match:
            continue
        new_name = f"{match.group(1)}{match.group(2)}"
        renames.append((path, path.with_name(new_name)))
    return renames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_renames = []
    for subdir in ("images", "labels"):
        all_renames.extend(planned_renames(RAW_DIR / subdir))

    if not all_renames:
        print("No Roboflow-suffixed file names found — nothing to do.")
        return

    targets = [dst for _, dst in all_renames]
    duplicates = {name for name in targets if targets.count(name) > 1}
    if duplicates:
        raise SystemExit(f"Refusing to rename: collisions would occur for {duplicates}")
    for _, dst in all_renames:
        if dst.exists():
            raise SystemExit(f"Refusing to rename: target already exists: {dst}")

    for src, dst in all_renames:
        print(f"{src.relative_to(REPO_ROOT)} -> {dst.name}")
        if not args.dry_run:
            src.rename(dst)

    print(f"{'Would rename' if args.dry_run else 'Renamed'} {len(all_renames)} files.")


if __name__ == "__main__":
    main()
