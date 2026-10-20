# Training and Generalization: larger-dataset execution

[Open the notebook](aa2_2_3_training_generalization.ipynb).

The default is now **3,000 training / 1,000 validation / 2,000 test images**.
All **25 code cells** ran in order in a fresh kernel, with **12 saved figures**
and no execution errors. Full execution took **625.8 seconds
(10.43 minutes), on CPU**. This includes imports,
plots, all search trials and the seed repeats; kernel startup is excluded.

The final configuration was **Tuned**, chosen by mean
validation loss over seeds 7, 17 and 27. The fixed seed-7 final model scored
**57.0% test accuracy**, compared with
**53.8%** for the seed-7 unregularized model:
**+3.2 percentage points**, exact paired McNemar **p=0.0040636**.
No test outcome was used to change the protocol.

## Dataset and runtime choices

- Standard mode uses 300 training images per class: 6% of the original
  50,000-image CIFAR-10 training partition. It remains a small teaching experiment.
- Set `RUN_MODE = 'quick'` in the first code cell for a nested 1,000-image
  training subset. Validation and test membership remain identical in both modes.
  Standard mode stays the default on every device.
- The version-2 archive is **16.95 MB**. It contains 4,000 development and 2,000
  reserved test images, with no exact pixel duplicates within or between splits.
- The previous notebook's 1,000 test images, including exact pixel copies,
  are excluded. The fresh holdout was evaluated only after selection finished.
- The Colab instructions select a **T4 GPU runtime** and check `Device: cuda`.
  Local CPU execution is supported; GPU/hosted-Colab timings were not measured.
- Mechanism demonstrations, architecture, maximum 30 epochs, patience 5,
  six eight-epoch search trials and three final training seeds are retained.

## Full summary

| Step | Experiment | Result | Fit time (s) |
| --- | --- | --- | --- |
| 1 | Data audit | 6,000 distinct archived images; within/cross-split exact duplicates: 0; previous test images excluded | — |
| 2 | Evaluation fixed | standard: 3,000 / 1,000 / 2,000 train/val/test; majority val 10.0%; initial loss 2.306 | — |
| 3 | Overfit one batch | 20 examples; 15 updates; loss 0.000528; accuracy 100.0% | 0.3 |
| 4 | Unregularized: last epoch | loss train/val 0.001/3.056; accuracy 100.0%/57.4% | — |
| 4 | Unregularized CNN | loss train/val 0.561/1.246; accuracy 82.8%/57.7%; epoch 9 | 62.5 |
| 4 | Narrow CNN | loss train/val 1.444/1.524; accuracy 46.8%/46.1%; epoch 6 | 5.5 |
| 4 | Small learning rate | loss train/val 1.835/1.824; accuracy 34.8%/34.4%; epoch 6 | 12.4 |
| 4 | With BatchNorm | loss train/val 0.511/1.186; accuracy 84.0%/58.2%; epoch 6 | 14.1 |
| 5 | Augmentation only | loss train/val 0.581/1.139; accuracy 79.7%/63.2%; epoch 26 | 64.5 |
| 5 | Weight decay only | loss train/val 0.588/1.247; accuracy 82.0%/57.9%; epoch 9 | 63.5 |
| 5 | Dropout only | loss train/val 0.875/1.249; accuracy 69.0%/55.9%; epoch 9 | 61.8 |
| 5 | Early stopping replay | patience 5; selected epoch 9; stop 14; 16 later epochs avoided | — |
| 6 | Random search | 6 trials × 8 epochs; winner Reference (reused); lr=0.001, wd=0 | 102.7 |
| 6 | Unregularized: 3 seeds | val loss 1.319 ± 0.068; val accuracy 54.5% ± 2.8% | 83.1 |
| 6 | Tuned: 3 seeds | val loss 1.171 ± 0.017; val accuracy 58.7% ± 1.0% | 151.0 |
| 6 | Majority: locked test | 10.0%; Wilson 95% [8.8%, 11.4%] | — |
| 6 | Unregularized: locked test | 53.8%; Wilson 95% [51.6%, 55.9%] | — |
| 6 | Final: locked test | 57.0%; Wilson 95% [54.8%, 59.1%] | — |
| 6 | McNemar: Majority vs final | p=1.3973e-214; paired table [[131, 69], [1008, 792]] | — |
| 6 | McNemar: Unregularized vs final | p=0.0040636; paired table [[866, 209], [273, 652]] | — |

## Reading the results

1. **Debugging precedes generalization.** The fixed 20-example batch reaches
   100% training accuracy. This checks the pipeline; it does not test recognition
   of unseen images.
2. **More examples still leave room to overfit.** The unregularized model's
   last epoch has 100.0% training accuracy and
   57.4% validation accuracy, with training/validation
   losses 0.001/3.056.
   Minimum validation loss instead selects epoch 9, with
   57.7% validation accuracy and loss
   1.246. Loss and accuracy need not rank checkpoints alike.
3. **Measure each intervention.** In the seed-7, full-budget comparisons,
   augmentation selects 63.2% validation accuracy
   and loss 1.139; weight decay selects
   57.9% and 1.247;
   Dropout selects 55.9% and
   1.249. The unregularized reference is
   57.7% and 1.246.
   These are effects of the particular settings under this budget.
4. **Optimization and generalization answer different questions.** At epoch 6,
   BatchNorm's training loss is 0.511 at its
   selected epoch 6; the original CNN's epoch-6 training
   loss is 0.936. Inspect the saved six-epoch
   curves alongside validation. This short diagnostic is not a tuned architecture ranking.
5. **A search may retain its reference.** The equal-budget search compares each
   trial with the reference's first eight epochs. The retained tuned settings are
   `{"augment": true, "batch_norm": false, "dropout": 0.0, "lr": 0.001, "weight_decay": 0.0, "width": 32}`.
   Three-seed selected validation loss is
   1.319 ± 0.068
   for the unregularized configuration and
   1.171 ± 0.017 for Tuned;
   accuracies are 54.5% ± 2.8%
   and 58.7% ± 1.0%. Seed 7 was
   fixed for testing in advance; it was not chosen for its score.
   The full-length augmentation demonstration selects epoch
   26; the actual seed-7 early-stopped refit selects
   epoch 14 and stops at
   19. It ends before a later improvement
   visible in the full demonstration. The final table correctly reports the
   checkpoint available under the declared patience rule.
6. **Report the locked test comparison.** The final model has
   57.0% accuracy, Wilson 95% interval
   [54.8%, 59.1%], and
   cross-entropy 1.202. The unregularized model has
   53.8% accuracy and cross-entropy
   1.338. The paired table is `[[866, 209], [273, 652]]`;
   exact McNemar p=0.0040636. The paired test provides evidence of an accuracy difference for these fixed models on this test population. It does not establish the same effect across training seeds or datasets.

The toy-experiment explanation remains explicit: a simple CNN learns from
scratch with limited data and computation. A modest absolute score does not
show that the techniques are generally ineffective. The observed comparisons
still need to be reported honestly. Wilson intervals concern a fixed model;
the three-seed standard deviation concerns retraining, and neither estimates
all uncertainty from split selection and search. The pooled Wilson interval
is an approximate summary for this class-balanced test sample.

This version selects a new development subset, expands validation and replaces
test. It therefore supersedes the old saved outputs and the earlier one-seed
data-size pilot. Their scores are not a controlled learning curve against this
run. Isolating a data-size effect would require a fixed validation protocol and
attention to the extra optimizer updates per epoch with more data.

## Verification and publication

- Verified image and label arrays against original CIFAR-10 indices, class
  balance, pixel uniqueness, split disjointness and exclusion of the actual
  version-1 test cohort. An independent exporter run reproduced the complete
  archive byte for byte.
- Checked loading from the repository root and `notebooks/`, simulated Colab
  download into an empty directory, cache reuse, missing-remote guidance and
  rejection of a corrupt archive. Both local and downloaded archives must pass
  the same SHA-256 check. A hosted Colab session was not executed.
- Checked the actual standard/quick splitting code: nested training subsets,
  identical validation membership and ordering, and correct balanced sizes.
  Quick mode also passed data preparation, the initial-loss check, one-batch
  memorization, and one plain and one augmented training epoch with validation
  and checkpoint restoration. No quick-mode test predictions were computed;
  only standard mode received a full end-to-end execution.
- Validated notebook schema, compiled sources, consecutive execution counts,
  saved outputs, figures and the declared default mode.

Publish the notebook with
`assets/datasets/cifar10_training_recipe_subset_v2.npz`. Include the updated
[dataset notes](../assets/datasets/cifar10_training_recipe_subset.md),
[exporter](../assets/datasets/prepare_training_recipe_subset.py) and this report.
The old archive remains available for older notebook revisions. The versioned
filename prevents a stale version-1 Colab cache from supplying old data.
The configured GitHub path must contain the new archive before students run
the notebook in Colab. No new dependency or full dataset download is needed.

Environment: Python 3.12.7, PyTorch 2.14.0, torchvision 0.29.1, NumPy 2.5.3,
matplotlib 3.11.1. Exact training outputs can vary across hardware and versions.
