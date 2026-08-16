# Training the map detector

This folder holds everything needed to train the RetinaNet model that detects
Pokemon Emerald overworld tiles (`map_detector.h5`). The original weights from
the upstream repo are gone (see `poke-AI/poke.AI#13`), but the repo still
contains the **real training data the author used**, so we can rebuild the
weights from scratch.

## The real dataset (bundled in this repo)

| Data | Location | Size |
|---|---|---|
| Training images (720x720) | `object_detection/training_data_new/` | 700 files, **313 referenced by the CSVs** |
| Validation images (720x720) | `object_detection/validation_data_new/` | 77 files, **38 referenced by the CSVs** |
| Original training annotations | `object_detection/training_csvs/keras_train.csv` | 3785 boxes |
| Original validation annotations | `object_detection/training_csvs/keras_validation.csv` | 408 boxes |
| Legacy Pascal/columnar CSVs | `object_detection/training_csvs/train.csv`, `validation.csv` | — |
| Empty labels file (broken) | `object_detection/training_csvs/labels.csv` | 0 bytes |

The image folders contain more files than the annotation CSVs reference — the
generator keys on unique paths in the CSV, so an epoch is **313 training batches
/ 38 validation batches** at batch size 1 (not 700/77). `--steps` must be sized
against those numbers.

Box counts by class (train / val):

| class | train | val | label id |
|---|---|---|---|
| pokecen | 42 | 3 | 0 |
| pokemart | 18 | 3 | 1 |
| npc | 791 | 98 | 2 |
| house | 124 | 19 | 3 |
| gym | 27 | 4 | 4 |
| exit | 51 | 6 | 5 |
| wall | 1735 | 209 | 6 |
| grass | 997 | 66 | 7 |

The label IDs above are **not arbitrary** — the application hard-codes them:

* `ai/standalone_backend.py:326` — `labels_to_names = {0: "pokecen", ... 7: "grass"}`
* `ai/mapper.py:368` — the same order in the tile conversion switch
* `ai/standalone_backend.py:131` — label 7 (grass) is detected but skipped when mapping

`training/data/real/classes.csv` reproduces this exact order, so the retrained
model drops straight into the app with no code changes.

## Why the CSVs were rewritten

The bundled CSVs contain absolute paths from the author's machine:

```
D:/App Development/pokemon_ai/object_detection/training_data_new/0.jpg,334,316,384,387,npc
```

Those paths are dead. The generator resolves image paths relative to the CSV's
directory, so the working copies in `training/data/real/` rewrite the prefix to
a repo-relative one:

```
../../../object_detection/training_data_new/0.jpg,334,316,384,387,npc
```

Regenerate them any time with:

```powershell
$newPrefix = "../../../object_detection/"
Get-Content object_detection/training_csvs/keras_train.csv |
  ForEach-Object { $_.Replace("D:/App Development/pokemon_ai/object_detection/", $newPrefix) } |
  Set-Content training/data/real/train_labels.csv
# same for keras_validation.csv -> val_labels.csv
```

## Training resolution: 400px

The app runs inference at a fixed size:

```python
# ai/standalone_backend.py:118
image, scale = resize_image(image, min_side = 400) # This model was trained with 400p images
```

So the model is trained at `--image-min-side 400 --image-max-side 400`.
Training at any other resolution creates a train/inference distribution shift.
The 720x720 source images are downscaled to 400x400 during training.

## Training command

Launched from the repo root with the project venv. The `scripts/train_map_detector.ps1`
wrapper encodes everything below (fresh or resume):

```powershell
uv run python -m keras_retinanet.bin.train `
  --backbone resnet50 `
  --weights object_detection/keras-retinanet/inference_graphs/resnet50_coco_best_v2.1.0.h5 `
  --batch-size 1 --steps 313 --epochs 12 `
  --lr 1e-4 --reduce-lr-patience 3 --reduce-lr-factor 0.5 `
  --gpu 1 --workers 0 `
  --image-min-side 400 --image-max-side 400 `
  --random-transform --compute-val-loss --no-evaluation `
  --snapshot-path training/snapshots_real `
  csv training/data/real/train_labels.csv training/data/real/classes.csv `
  --val-annotations training/data/real/val_labels.csv
```

Notes:

* Common flags go **before** the `csv` subcommand; `--val-annotations` after.
* `--weights <coco.h5>` starts from COCO-pretrained ResNet50 (feature maps are
  reusable; only the classifier heads need to relearn the 8 tile classes).
* `--steps 313` = one full pass over the 313 unique training images at batch 1.
* `--gpu 1` trains on the GTX 1060 — **the GTX 1070 is the display card**; training
  on it freezes the desktop. TF device indices differ from nvidia-smi's.
* `--no-evaluation` skips the per-epoch mAP callback (it builds a second model
  and spiked GPU memory); `--compute-val-loss` still reports val loss.
* `--random-transform` applies random flips/shifts/zooms per image.
* `--reduce-lr-*` drops the LR when val loss stops improving.
* A snapshot is saved at the end of every epoch into `training/snapshots_real/`
  (`resnet50_csv_XX.h5`).

### Resuming

Resume requires the **specific snapshot file** (not just the folder) plus the
epoch number the snapshot was written at, or the first resumed epoch
overwrites the snapshot file:

```powershell
powershell -File scripts/train_map_detector.ps1 `
  -Resume training/snapshots_real/resnet50_csv_03.h5 -InitialEpoch 3 -Epochs 12
```

The `--snapshot-path` argument is save-only; `--snapshot <file>` + `--initial-epoch <n>`
is the resume path. Omit `--weights`/`--snapshot` and train.py falls back to
ImageNet init (downloads ResNet-50-model.keras.h5).

Throughput measured on the GTX 1060: ~0.6s/step at 400px batch 1, so ~3 minutes
per epoch of 313 steps.

## Monitoring

The live run logs to `training/train_real.log` (stdout) and
`training/train_real_err.log` (stderr). Watch loss and val mAP:

```powershell
Get-Content training/train_real.log -Tail 5
```

Sanity signals: epoch-1 loss starts ~2.2 and drops to ~1.0 within the first
hundred steps; training mAP converges toward 1.0; val mAP tracks it. If val
mAP plateaus while train mAP is high, training is overfitting — stop early and
use the best snapshot.

## Converting a snapshot to a deployable model

```powershell
uv run python -m keras_retinanet.bin.convert_model `
  training/snapshots_real/resnet50_csv_XX.h5 `
  object_detection/keras-retinanet/inference_graphs/map_detector.h5 `
  --backbone resnet50
```

`convert_model` rewrites the 4-output training model (regression, classification,
anchors, prior) into the 3-output inference form (`boxes, scores, labels`) that
`ai/standalone_backend.py` expects via `models.load_model(path, backbone_name="resnet50")`.

`scripts/convert_latest_snapshot.ps1` does this for the newest snapshot.

## Current trained weights (Aug 2026)

The bundled real data trained 12 epochs from the COCO init (resumed across runs;
best val_loss **0.5259** at epoch 12, lr auto-reduced 1e-4 → 5e-5). The resulting
`map_detector.h5` was verified end-to-end through the app's load path and
inference settings (`min_side=400`, threshold 0.85, grass skipped):

* Validation image npc boxes: all detected near-perfectly at 0.92–1.00 confidence.
* `pokecen` (285,190,484,384 GT → predicted 285,188,487,381 @0.97) and `wall`
  boxes detected accurately; `npc` on image 0.jpg matches GT box exactly.
* Weak classes: `pokemart` (18 train boxes) and edge-cropped small `npc` boxes
  are frequently missed — expected with the severe class imbalance. Retraining
  with more `pokemart`/`gym`/`exit` annotations would close that gap.

## Synthetic smoke-test pipeline (proven the plumbing works)

Before touching the real data, the full pipeline was validated with generated
images so failures could be attributed to tooling rather than data.

* `make_synthetic_dataset.py` — deterministically renders 320x320 images with
  8 color-coded classes (SEED=42), writing `training/data/synthetic/` with
  `classes.csv`, `train_labels.csv`, `val_labels.csv`.
* Training on synthetic data reached training mAP 1.0000 and the converted
  model loaded fine through the app's exact load path.
* Caveat observed: a model trained at lr=1e-5 produced low raw scores at
  inference (focal-loss calibration artifact) even though mAP was 1.0. The
  app's `score < 0.85` cut-off means score calibration matters — one reason the
  real run uses lr=1e-4 and reduce-on-plateau.

## Critical environment lesson: cuDNN build

TensorFlow 2.10 GPU hard-crashes (`0xC0000409`, fast-fail, no stack trace)
on the **first convolution** when given a conda-forge-built cuDNN (both 8.1 and
8.4 tested) — the conda builds are incompatible with the bundled CUDA 11.2
runtime on this machine. The fix is the **official NVIDIA cuDNN 8.1.0.77 for
CUDA 11.2**:

```
https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.0/cudnn-11.2-windows-x64-v8.1.0.77.zip
```

`scripts/get_cuda_runtime.ps1` downloads it (no login required) into the
project-local `cuda_runtime/bin/`, which the venv's `sitecustomize.py` adds to
`PATH`. Re-run that script after recreating the venv.

## Critical lesson: the generator was not looping

Every training run previously "died" after exactly one epoch. It was not a
crash — TF 2.10 wraps the `keras.utils.Sequence` in `KerasSequenceAdapter`,
whose `generator_fn` enumerates `range(len(sequence))` and then raises
`StopIteration`:

```
WARNING:tensorflow:Your input ran out of data; interrupting training. ...
```

`fit_generator` then pads the rest of the epoch by replaying the last batch and
**stops after one epoch**. Fixes applied to the vendored fork (already editable
in `pyproject.toml` via `[tool.uv.sources]` → `editable = true`):

* `keras_retinanet/preprocessing/generator.py` — `__len__` returns
  `len(self.groups) * 1000` and `__getitem__` wraps `index % len(self.groups)`,
  calling `on_epoch_end()` on each wrap. This restores the upstream
  keras-retinanet infinite-looping behavior while keeping `list(range(len))`
  cheap for the adapter. (A literal `sys.maxsize` here OOMs the adapter, which
  materializes the range as a list when `shuffle=True`.)
* `keras_retinanet/bin/train.py` — passes
  `validation_steps=len(validation_generator.groups)` explicitly; otherwise the
  validation loop inherits the inflated length.

Verified: `fit_generator(steps_per_epoch=350, epochs=2)` on the 313-group
generator now completes both epochs with no "ran out of data" warning
(`epoch keys: [0, 1]`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Call to CreateProcess failed ... "ptxas.exe"` in stderr | Benign. ptxas isn't on PATH; the driver JIT-compiles instead. One warning per run. |
| `bfc_allocator ... near the threshold` warnings | Normal fragmentation chatter at ~6GB. If OOM, drop `--batch-size 1`. |
| `OOM at start` | Use batch 1 and/or `--image-max-side 400`. |
| `MemoryError ... PyFunc` after `len()` change | `__len__` returned `sys.maxsize`; the adapter materializes `list(range(len))`. Use `len(groups) * 1000`. |
| `Your input ran out of data; interrupting training` | Generator patch missing — re-run `uv sync` so the editable install picks up the repo fork. |
| Val mAP ≪ train mAP | Overfitting; fewer epochs or stronger `--random-transform`. |
| Converted model outputs garbage | Wrong snapshot (4-output training form), or `--backbone` mismatch. |
| `AssertionError` on CSV classes | Every class in the annotations must appear in `classes.csv` exactly once. |

## Retraining with new data

1. Annotate 720x720 screenshots with labelImg, save VOC XMLs per image.
2. Convert to CSV: `object_detection/data_processing/` holds the author's
   conversion scripts.
3. Drop the CSVs into `training/data/real/`, keep the class order in
   `classes.csv` fixed (app hard-codes label IDs), and rerun the training
   command above.
