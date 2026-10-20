# CIFAR-10 training recipe subset, version 2

The archive `cifar10_training_recipe_subset_v2.npz` accompanies
[`aa2_2_3_training_generalization.ipynb`](../../notebooks/aa2_2_3_training_generalization.ipynb).
It is separate from the methodology subset. The earlier
`cifar10_training_recipe_subset.npz` is retained for older notebook revisions;
the current notebook loads only version 2.

- Source: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), by Alex Krizhevsky,
  Vinod Nair and Geoffrey Hinton. Torchvision checks source integrity before export.
- Compressed size: **16.95 MB**.
- SHA-256: `0f2d4ba224b4a7c3aea490b3c767c32c6fd1afde48b775558ea975f33702936a`.
- Development pool: **4,000 images**, 400 per class, from the official training partition.
- Reserved test: **2,000 images**, 200 per class, from the official test partition.
- All 6,000 images have different SHA-256 hashes of their raw RGB pixels.
- The 1,000 official-test images used in the earlier notebook, including exact
  pixel copies, are excluded from both the new development pool and test sample.
  Their outcomes had already been inspected. Their original indices document
  exclusion; their images and labels are absent from this archive.
- The original training/test boundary is preserved. Original numerical indices
  in the two partitions use separate namespaces.

## Selection and notebook modes

The exporter reconstructs the earlier development/test selection with NumPy
`default_rng` seeds 7 and 8. It records the old test indices and pixel hashes,
then excludes those hashes. It selects 400 unique development images per class
with seed 7 and 200 unique test images per class with seed 108. Each selection
randomly orders candidates within each class and skips hashes already used.

The notebook uses seed 7 to permute each development class. The first 300
candidates are standard-mode training; the remaining 100 are validation.
Quick mode uses only the first 100 training candidates. Validation is shuffled
before training so its ordering and membership are identical in both modes.
Every experiment within a run uses the same chosen split.

| `RUN_MODE` in the first code cell | Training | Validation | Test | Unused development images |
| --- | --- | --- | --- | --- |
| `'standard'` (default; saved outputs) | 3,000 | 1,000 | 2,000 | 0 |
| `'quick'` | 1,000 | 1,000 | 2,000 | 2,000 |

Training is balanced by class in both modes. Input normalization uses only the
selected training images. The training subsets are nested, but this selection
differs from the earlier small notebook and its separate data-size pilot;
earlier numerical results are not the outputs of this version. More training
images also give more optimizer updates for the same epoch count. Use validation
for further data-size experiments; do not tune using test scores.

## Archive contents

| Key | Shape | Type | Contents |
| --- | --- | --- | --- |
| `pool_images` | `(4000, 32, 32, 3)` | `uint8` | Raw development RGB images |
| `pool_labels` | `(4000,)` | `int64` | Development class IDs |
| `pool_indices` | `(4000,)` | `int64` | Original official-training indices |
| `test_images` | `(2000, 32, 32, 3)` | `uint8` | Reserved official-test images |
| `test_labels` | `(2000,)` | `int64` | Reserved class IDs |
| `test_indices` | `(2000,)` | `int64` | Original official-test indices |
| `classes` | `(10,)` | Unicode | Class names in label order |
| `subset_seeds` | `(2,)` | `int64` | Development/test selection seeds: 7, 108 |
| `previous_test_indices` | `(1000,)` | `int64` | Excluded version-1 official-test indices |
| `schema_version` | `()` | `int64` | Archive version: 2 |

Load with `numpy.load(path, allow_pickle=False)`. Images are neither normalized
nor augmented. Hash checks detect exact copies, not all near-duplicates.

## Local Jupyter and Google Colab

1. Locally, the notebook checks `assets/datasets/` from the repository root and
   `../assets/datasets/` from `notebooks/`.
2. It also checks `data/` for a cached archive.
3. If absent, it downloads only the 16.95 MB archive from
   `https://raw.githubusercontent.com/imatge-upc/aa2/main/assets/datasets/cifar10_training_recipe_subset_v2.npz`
   into `data/`. The versioned filename avoids reusing a version-1 Colab cache.
4. It verifies the same SHA-256 before loading either source.

Publish these files together:

```text
notebooks/aa2_2_3_training_generalization.ipynb
assets/datasets/cifar10_training_recipe_subset_v2.npz
assets/datasets/cifar10_training_recipe_subset.md
assets/datasets/prepare_training_recipe_subset.py
notebooks/aa2_2_3_training_generalization_results.md
```

Only the notebook and version-2 archive are needed for student execution. Notes,
exporter and results document provenance and verification. No change to the
methodology archive or notebook is required.

Set `GITHUB_REPO` and `GITHUB_REF` for a different repository or branch. A failed
download identifies the expected publication path. In Colab, select **Runtime →
Change runtime type → Hardware accelerator → T4 GPU → Save**, then **Runtime →
Run all**. The current run should print `Device: cuda`. Standard mode remains the
default on every device; choose quick mode explicitly for a shorter CPU session.
GPU timings have not been measured locally.

## Regeneration

Run from the repository root with the course environment and an existing source cache:

```bash
.venv/bin/python assets/datasets/prepare_training_recipe_subset.py
```

Use `--data-root PATH` for another cache or `--output PATH` for another output
location. Pass `--download` explicitly to allow the full source download.
Regeneration does not depend on the old archive being present. Normal notebook
execution never downloads the full CIFAR-10 source or pretrained weights.

Export environment: Python 3.12.7, NumPy 2.5.3, PyTorch 2.14.0, torchvision 0.29.1.
Original indices preserve exact samples even if future generator behavior changes.
The notebook prints runtime versions; training results can vary across devices
and versions.
