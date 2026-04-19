# Training Outputs

This folder contains saved model checkpoints and evaluation artifacts produced by `src/train.py`.

## Checkpoint naming

The training script builds a run-specific folder using:

`<backbone>_<wandb-run-name>`

Each run writes its best checkpoint to:

`outputs/checkpoints/<run-tag>/best_model.pth`

and its results to:

`outputs/results/<run-tag>/`

## Known runs in this workspace

| Run tag | Checkpoint | Notes |
| --- | --- | --- |
| `efficientnet-b3_usual-salad-4` | `outputs/checkpoints/efficientnet-b3_usual-salad-4/best_model.pth` | EfficientNet-B3 training run |
| `efficientnet-b3_cardassian-spot-5` | `outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth` | EfficientNet-B3 run with alternate checkpoint name |
| `custom-cnn_comfy-frost-7` | `outputs/checkpoints/custom-cnn_comfy-frost-7/best_model.pth` | Custom CNN baseline run |
| `custom-cnn_frosty-deluge-6` | not found | Run folder exists, but no checkpoint file is currently present |

## Legacy checkpoint

The standalone file `outputs/checkpoints/best_model.pth` is an older checkpoint dated `2026-03-28 00:45:47`. It is not linked to a named run folder in the current codebase, so treat it as an orphaned/legacy checkpoint unless you can match it to an older W&B run manually.
