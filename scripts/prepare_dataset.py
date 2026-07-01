#!/usr/bin/env python3
"""
Splits the raw Roboflow export (data/segmentation/raw/) into train/valid/test
YOLO-seg folders, grouped by animal ID so the same cow never appears in more
than one split (image file names follow the pattern cow_<id>_angle<n>.jpg —
splitting by individual file would leak near-duplicate angles of the same
animal across splits and inflate validation metrics).

Also mixes in negative ("no cow here") images from data/negatives/raw/, if
present — dogs, horses, people, empty scenes, anything without cattle. These
have no label file, which Ultralytics YOLO treats as a background image (0
instances): without any, the model was only ever trained on photos that DO
contain a cow, so it never learned what "there's no cow here" looks like,
and tends to hallucinate a confident detection on any input. See
data/negatives/raw/README.md.

Usage:
    python scripts/prepare_dataset.py [--train 0.8] [--val 0.1] [--test 0.1] [--seed 42]

Output (regenerated on every run, not committed to git):
    data/segmentation/train/{images,labels}/
    data/segmentation/valid/{images,labels}/
    data/segmentation/test/{images,labels}/
    data/segmentation/data.yaml
"""
import argparse
import random
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "segmentation" / "raw"
NEGATIVES_RAW_DIR = REPO_ROOT / "data" / "negatives" / "raw"
OUT_DIR = REPO_ROOT / "data" / "segmentation"
COW_ID_PATTERN = re.compile(r"^(cow_\d+)_")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def group_by_animal(image_paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for image_path in image_paths:
        match = COW_ID_PATTERN.match(image_path.stem)
        if not match:
            raise ValueError(
                f"Filename does not match expected 'cow_<id>_angle<n>' pattern: {image_path.name}"
            )
        groups.setdefault(match.group(1), []).append(image_path)
    return groups


def split_groups(
    animal_ids: list[str], train: float, val: float, test: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    assert abs(train + val + test - 1.0) < 1e-6, "split ratios must sum to 1.0"
    shuffled = animal_ids[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train)
    n_val = round(n * val)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def split_files(paths: list[Path], train: float, val: float, test: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    assert abs(train + val + test - 1.0) < 1e-6, "split ratios must sum to 1.0"
    shuffled = paths[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train)
    n_val = round(n * val)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def populate_split(name: str, animal_ids: set[str], groups: dict[str, list[Path]]) -> int:
    images_out = OUT_DIR / name / "images"
    labels_out = OUT_DIR / name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for animal_id in animal_ids:
        for image_path in groups[animal_id]:
            label_path = RAW_DIR / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
            shutil.copy2(image_path, images_out / image_path.name)
            shutil.copy2(label_path, labels_out / label_path.name)
            count += 1
    return count


def populate_negatives(name: str, image_paths: list[Path]) -> int:
    # No label file is written for these on purpose — Ultralytics treats a
    # missing label as "background, 0 instances", exactly what a negative
    # example should teach the model.
    images_out = OUT_DIR / name / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        shutil.copy2(image_path, images_out / image_path.name)
    return len(image_paths)


def write_data_yaml(class_name: str = "boi") -> None:
    # Relative to this file's own directory, so the dataset is portable across
    # machines (e.g. prepared here, trained later on a Mac).
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    (OUT_DIR / "data.yaml").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_paths = sorted((RAW_DIR / "images").glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No images found under {RAW_DIR / 'images'}")

    groups = group_by_animal(image_paths)
    train_ids, val_ids, test_ids = split_groups(
        sorted(groups), args.train, args.val, args.test, args.seed
    )

    negative_paths = sorted(
        p for ext in IMAGE_EXTENSIONS for p in NEGATIVES_RAW_DIR.glob(f"*{ext}")
    ) if NEGATIVES_RAW_DIR.exists() else []
    neg_train, neg_val, neg_test = split_files(negative_paths, args.train, args.val, args.test, args.seed)

    for split_dir in ("train", "valid", "test"):
        shutil.rmtree(OUT_DIR / split_dir, ignore_errors=True)

    n_train = populate_split("train", train_ids, groups)
    n_val = populate_split("valid", val_ids, groups)
    n_test = populate_split("test", test_ids, groups)
    n_neg_train = populate_negatives("train", neg_train)
    n_neg_val = populate_negatives("valid", neg_val)
    n_neg_test = populate_negatives("test", neg_test)
    write_data_yaml()

    print(f"Animals: {len(groups)} total -> train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")
    print(f"Positive images: train {n_train}, val {n_val}, test {n_test}")
    print(f"Negative images: train {n_neg_train}, val {n_neg_val}, test {n_neg_test} (from {NEGATIVES_RAW_DIR})")
    print(f"Wrote {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
