"""Export the version-2, deduplicated CIFAR-10 archive for lecture 2.3.

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
                        default=Path(__file__).with_name("cifar10_training_recipe_subset_v2.npz"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    training = CIFAR10(args.data_root, train=True, download=args.download)
    testing = CIFAR10(args.data_root, train=False, download=args.download)
    train_labels = np.asarray(training.targets, dtype=np.int64)
    test_labels = np.asarray(testing.targets, dtype=np.int64)
    # Reconstruct the previous notebook's test cohort without depending on an
    # existing archive. Its outcomes have already been inspected, so none of
    # those images (including exact pixel copies) can enter the new holdout.
    legacy_hashes = set()
    select_per_class(training.data, train_labels, 150, 7, legacy_hashes)
    previous_test_indices = select_per_class(testing.data, test_labels, 100, 8, legacy_hashes)
    excluded_hashes = {
        hashlib.sha256(testing.data[index].tobytes()).digest()
        for index in previous_test_indices
    }
    used_hashes = set(excluded_hashes)
    pool_indices = select_per_class(training.data, train_labels, 400, 7, used_hashes)
    test_indices = select_per_class(testing.data, test_labels, 200, 108, used_hashes)
    assert len(used_hashes) == 7000
    assert not set(previous_test_indices) & set(test_indices)

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
        subset_seeds=np.asarray([7, 108], dtype=np.int64),
        previous_test_indices=previous_test_indices,
        schema_version=np.asarray(2, dtype=np.int64),
    )
    print(f"Saved {args.output}: {args.output.stat().st_size / 1e6:.2f} MB")
    print("SHA-256:", hashlib.sha256(args.output.read_bytes()).hexdigest())
    print("Development pool: 400/class from official training; test: 200/class from official test.")
    print("All 6,000 images have distinct SHA-256 pixel hashes; the previous 1,000 test images are excluded.")
    print("Notebook split: 3,000 train / 1,000 validation / 2,000 test; quick mode trains on 1,000.")


if __name__ == "__main__":
    main()
