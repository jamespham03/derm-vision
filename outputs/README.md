# outputs/

This is where `src/train.py` writes everything: model weights, confusion matrices, metric dumps. Each W&B run gets its own folder so we don't accidentally overwrite a good checkpoint with a worse one.

The folder name is `<backbone>_<wandb-run-name>`, e.g. `efficientnet-b3_cardassian-spot-5`. Inside that folder you'll find:

- `best_model.pth` — the checkpoint with the best val loss seen during the run
- the confusion matrix and metric files produced by `src/evaluate.py`

Results plots also land under `outputs/results/<run-tag>/`.

## What's actually here

| Run tag | Checkpoint file | Notes |
|---|---|---|
| `efficientnet-b3_usual-salad-4` | `best_model.pth` | EfficientNet-B3, focal loss + class weights — see Run 2 in the experiments report |
| `efficientnet-b3_cardassian-spot-5` | `best_model-2.pth` | The one the web app loads. Best F1 we got. |
| `custom-cnn_comfy-frost-7` | `best_model.pth` | The from-scratch CNN baseline |
| `custom-cnn_frosty-deluge-6` | — | Folder exists but no checkpoint inside; safe to ignore |

Note that `cardassian-spot-5` is `best_model-2.pth`, not `best_model.pth`. We re-ran the eval after a tweak and the new checkpoint got a `-2` suffix instead of overwriting; just kept it that way.

## Legacy file

There's a stray `outputs/checkpoints/best_model.pth` at the top of the checkpoints folder, dated `2026-03-28 00:45:47`. It predates the per-run folder layout so it's not tied to any named run anymore. We're keeping it around in case we need to dig into an early result, but don't load it for anything that matters.
