# CIFAR-10 training recipe subset

The archive `cifar10_training_recipe_subset.npz` accompanies
[`aa2_2_3_training_generalization.ipynb`](../../notebooks/aa2_2_3_training_generalization.ipynb).
It is separate from the existing methodology subset.

- Source: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), by Alex Krizhevsky,
  Vinod Nair and Geoffrey Hinton. Source files are checked by torchvision's
  built-in integrity checks before export.
- Compressed size: approximately 7.07 MB.
- SHA-256: `5fcac727dc645de80582f9cca26a1c39de427a8119607c02d65a82a9312884d3`.
- Development pool: 1,500 images, 150 per class, from the official training partition.
- Reserved test: 1,000 images, 100 per class, from the official test partition.
- All 2,500 images have different SHA-256 hashes of their raw RGB pixels.
- Export uses NumPy `default_rng` seeds 7 (development) and 8 (test).
  Each class is randomly ordered; repeated pixel hashes are skipped.
- Notebook seed 7 stratifies the development pool into 1,000 training and
  500 validation images. Normalization is estimated from those training images only.
- The original dataset's training/test boundary is preserved. Numerical indices
  in the two official partitions use separate namespaces.

| Key | Shape | Type | Contents |
| --- | --- | --- | --- |
| `pool_images` | `(1500, 32, 32, 3)` | `uint8` | Raw development RGB images |
| `pool_labels` | `(1500,)` | `int64` | Development class IDs |
| `pool_indices` | `(1500,)` | `int64` | Original official-training indices |
| `test_images` | `(1000, 32, 32, 3)` | `uint8` | Reserved official-test images |
| `test_labels` | `(1000,)` | `int64` | Reserved class IDs |
| `test_indices` | `(1000,)` | `int64` | Original official-test indices |
| `classes` | `(10,)` | Unicode | Class names in label order |
| `subset_seeds` | `(2,)` | `int64` | Development/test selection seeds |

Load using `numpy.load(path, allow_pickle=False)`. No image is normalized or
augmented in the archive. Hash checks detect exact copies, not all near-duplicates.

The notebook supports the same dataset in local Jupyter and Google Colab:

1. In a local checkout, it checks `assets/datasets/` from the repository root
   and `../assets/datasets/` from `notebooks/`.
2. It also checks `data/` for an archive cached by an earlier run.
3. If no local archive exists, it downloads only this 7.07 MB file from
   `https://raw.githubusercontent.com/imatge-upc/aa2/main/assets/datasets/cifar10_training_recipe_subset.npz`
   into `data/`. Subsequent executions in the same runtime reuse that copy.
4. It verifies the same SHA-256 checksum before loading either source.

Publish the notebook and archive in the repository at these exact paths:

```text
notebooks/aa2_2_3_training_generalization.ipynb
assets/datasets/cifar10_training_recipe_subset.npz
```

The dataset notes and export script can be published alongside the archive for
provenance and reproducibility. Set `GITHUB_REPO` and `GITHUB_REF` in the data cell
if the files are in a different repository or branch. A failed download gives a
message identifying the expected publication path. Opening the notebook using
the Colab badge then needs only **Run all**; the loader fetches the archive.
Colab caches last for the current runtime. Normal lecture execution never
downloads the full CIFAR-10 source dataset.

To regenerate using an existing source download, run from the repository root:

```bash
.venv/bin/python assets/datasets/prepare_training_recipe_subset.py
```

Use `--data-root PATH` for a different source cache. Pass `--download` explicitly
to allow torchvision to download the full source dataset. The exporter is separate
from notebook execution to keep lecture runs short.

Export environment: Python 3.12.7, NumPy 2.5.3, PyTorch 2.14.0,
torchvision 0.29.1. Original indices preserve the exact selected samples even
if future random-generator behavior changes. The notebook prints its runtime
library versions; exact training results can vary with hardware and versions.
