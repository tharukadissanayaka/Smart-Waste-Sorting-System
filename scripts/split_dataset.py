from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_extension(ext: str) -> str:
    return ext.lower().strip()


def find_paired_samples(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue

        if normalize_extension(image_path.suffix) not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.is_file():
            pairs.append((image_path, label_path))

    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    shuffled_pairs = pairs.copy()
    random.Random(seed).shuffle(shuffled_pairs)

    total = len(shuffled_pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_pairs = shuffled_pairs[:train_count]
    val_pairs = shuffled_pairs[train_count : train_count + val_count]
    test_pairs = shuffled_pairs[train_count + val_count : train_count + val_count + test_count]

    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def prepare_output_dirs(output_dir: Path) -> None:
    for split_name in ("train", "val", "test"):
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)


def copy_split(split_name: str, pairs: list[tuple[Path, Path]], output_dir: Path) -> None:
    image_output_dir = output_dir / split_name / "images"
    label_output_dir = output_dir / split_name / "labels"

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_output_dir / image_path.name)
        shutil.copy2(label_path, label_output_dir / label_path.name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split YOLO image/label pairs into train, validation, and test sets."
    )
    parser.add_argument("--images", type=Path, required=True, help="Folder containing images.")
    parser.add_argument("--labels", type=Path, required=True, help="Folder containing YOLO .txt labels.")
    parser.add_argument("--output", type=Path, required=True, help="Destination folder for split data.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Testing split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    images_dir = args.images.resolve()
    labels_dir = args.labels.resolve()
    output_dir = args.output.resolve()

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images folder does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels folder does not exist: {labels_dir}")

    pairs = find_paired_samples(images_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError("No matching image/label pairs were found.")

    splits = split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    prepare_output_dirs(output_dir)
    for split_name, split_pairs_list in splits.items():
        copy_split(split_name, split_pairs_list, output_dir)

    print("Dataset split complete.")
    print(f"Total paired samples: {len(pairs)}")
    print(f"Training: {len(splits['train'])}")
    print(f"Validation: {len(splits['val'])}")
    print(f"Testing: {len(splits['test'])}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
