"""Export the small, deduplicated CIFAR-10 archive for lecture 2.3.

Run with the course Python environment. Existing CIFAR-10 files are reused;
pass --download explicitly if a full source download is needed.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10


def select_per_class(images, labels, count, seed, used_hashes):
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        accepted = []
        for index in rng.permutation(np.flatnonzero(labels == label)):
            fingerprint = hashlib.sha256(images[index].tobytes()).digest()
            if fingerprint not in used_hashes:
                accepted.append(int(index))
                used_hashes.add(fingerprint)
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise ValueError(f"Not enough unique images for class {label}.")
        selected.extend(accepted)
    return rng.permutation(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", type=Path, default=repo / "notebooks/data")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("cifar10_training_recipe_subset.npz"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    training = CIFAR10(args.data_root, train=True, download=args.download)
    testing = CIFAR10(args.data_root, train=False, download=args.download)
    train_labels = np.asarray(training.targets, dtype=np.int64)
    test_labels = np.asarray(testing.targets, dtype=np.int64)
    used_hashes = set()
    pool_indices = select_per_class(training.data, train_labels, 150, 7, used_hashes)
    test_indices = select_per_class(testing.data, test_labels, 100, 8, used_hashes)
    assert len(used_hashes) == 2500

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pool_images=training.data[pool_indices],
        pool_labels=train_labels[pool_indices],
        pool_indices=pool_indices,
        test_images=testing.data[test_indices],
        test_labels=test_labels[test_indices],
        test_indices=test_indices,
        classes=np.asarray(training.classes, dtype=str),
        subset_seeds=np.asarray([7, 8], dtype=np.int64),
    )
    print(f"Saved {args.output}: {args.output.stat().st_size / 1e6:.2f} MB")
    print("SHA-256:", hashlib.sha256(args.output.read_bytes()).hexdigest())
    print("Development pool: 150/class from official training; test: 100/class from official test.")
    print("All 2,500 images have distinct SHA-256 pixel hashes.")


if __name__ == "__main__":
    main()
