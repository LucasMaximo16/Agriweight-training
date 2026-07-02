#!/usr/bin/env python3
"""
Strips Roboflow's export suffix (`_<origext>.rf.<hash>`) from image/label
file names in data/segmentation/raw/ and data/negatives/raw/, restoring the
original naming (e.g. `cow_<id>_angle<n>` for cattle photos, whatever the
source used for negatives). Roboflow appends that hash on every export, so
re-run this after importing a fresh export.

Usage:
    python scripts/clean_filenames.py [--dry-run]
"""
import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_RAW_DIR = REPO_ROOT / "data" / "segmentation" / "raw"
NEGATIVES_RAW_DIR = REPO_ROOT / "data" / "negatives" / "raw"

# Matches e.g. "cow_0001_angle1_jpg.rf.<hash>.jpg" or "rgb_185_png.rf.<hash>.jpg"
# or "2170_adelante_jpg.rf.<hash>.jpg" — anything Roboflow exported, regardless
# of the original base name or original extension it encodes mid-filename.
ROBOFLOW_SUFFIX = re.compile(r"^(.+)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+(\.\w+)$")


def planned_renames(directory: Path) -> list[tuple[Path, Path]]:
    if not directory.exists():
        return []

    # Group by (base, ext) first: Roboflow sometimes exports the same source
    # image multiple times (its own augmentation pass), each getting a
    # different hash but the same base name. Stripping the hash naively would
    # collide those into one name — number them instead of losing files.
    groups: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(directory.iterdir()):
        match = ROBOFLOW_SUFFIX.match(path.name)
        if not match:
            continue
        groups.setdefault((match.group(1), match.group(2)), []).append(path)

    renames = []
    for (base, ext), paths in groups.items():
        if len(paths) == 1:
            renames.append((paths[0], paths[0].with_name(f"{base}{ext}")))
        else:
            for i, path in enumerate(paths, start=1):
                renames.append((path, path.with_name(f"{base}_{i}{ext}")))
    return renames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_renames = []
    for subdir in ("images", "labels"):
        all_renames.extend(planned_renames(SEGMENTATION_RAW_DIR / subdir))
    all_renames.extend(planned_renames(NEGATIVES_RAW_DIR))

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
