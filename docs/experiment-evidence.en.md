# Experiment sources and reproduction limits

[中文](experiment-evidence.md) · English · [Back to README](../README_en.md)

Checked on 2026-10-13 against source [0a4e5d3](https://github.com/adenzhou1350/minimind-diffusion/tree/0a4e5d3cf5ff3a78381d126ce131702352a79b1b). This page separates code settings, historical written observations, one VLM training log recovered from an old local directory, and artifacts still unpublished. Historical training was not rerun.

## Start with what can be checked directly

- [Model tests](../tests/test_model.py) and [loss tests](../tests/test_loss.py) use small random models with no downloaded tokenizer, data or weights. Setup and commands are in the [README](../README_en.md#first-run-small-cpu-tests).
- [Sampling tests](../tests/test_sampling.py) check length, mask removal and code paths; passing them does not establish language quality.
- The [training entry point](../trainer/train_pretrain.py) and [data notes](../dataset/dataset.md) require external assets. No trained weights are bundled for an immediate conversation demo.

## Where the numbers came from

| Historical observation | Fixed source | Missing evidence |
|---|---|---|
| 23M screening: about 5% versus 25.5% reconstruction accuracy, the latter at 3,000 steps | [Loss-change commit message `fb96b34`](https://github.com/adenzhou1350/minimind-diffusion/commit/fb96b344f5e58b7f9074de72e19473afdee67927) | Raw outputs for both runs, complete configurations and asset fingerprints |
| Old implementation 0–5%; later result labeled 64M: 41.5%, loss 7.4→1.7; approximately 41%→44% from epoch 1 to 2 | [Historical README `c080996`](https://github.com/adenzhou1350/minimind-diffusion/blob/c08099625cbd36fec3bad8b2600d0f4036da03c2/README.md) | Matching logs, checkpoints, evaluation scripts and configuration mapping |
| VLM Stage 2: last training metric `22690/22691`, loss `2.1442`, then a save message | [Recovered raw log](evidence/vlm-sft-20260811-20260814.log), [verification notes](evidence/README.md) | Checkpoint hash, complete command, data fingerprints and source revision; a save message does not verify weight contents |
| VLM: similar hallucinations after swapping images | [Historical README `24a64b7`](https://github.com/adenzhou1350/minimind-diffusion/blob/24a64b72f876fcc5e3aad2c7ffe0b08277f6914c/README.md) | Per-image predictions, image sources and fixed controls; training loss does not establish grounding quality |

The README and commit links establish what was written at the time. At the source baseline above, its full public master history had no matching raw logs, per-example results or checkpoints. The VLM log in the table was subsequently recovered from the author's old local experiment directory and published; text-side logs remain missing. The 23M screening and later 64M record are not a controlled comparison with one shared configuration.

The recovered log spans **53 hours 46 minutes** between its first and last training metrics (`2026-08-11 20:54:07` to `2026-08-14 02:40:07`). This does not support the old “about 30 hours” estimate and is not GPU utilization time. Duplicate final save messages and a partial line are preserved byte-for-byte. SHA-256: `af63cb4c7cf16ecc4828991a828e528463b78aa999a88075ed7ebc8f55391dfe`. See [log notes](evidence/README.md) for provenance, the line-by-line check and limits.

## What the screening scripts actually do

The settings below come from the three scripts at the pinned baseline. `256×4` means hidden size 256 with 4 layers, not a verified parameter count. The historical label “23M” is retained only as a quotation of the commit record.

| Script | Model and training | Data row ranges (zero-based, upper bound excluded) | Evaluation |
|---|---|---|---|
| [screen_loss.py](https://github.com/adenzhou1350/minimind-diffusion/blob/0a4e5d3cf5ff3a78381d126ce131702352a79b1b/screen_loss.py) | 256×4, batch 16, 500 steps per config, lr 5e-4; 4 weight / mask-range configs | Train `[0, 4000)`; held-out `[4000, 4050)` | Randomly mask about 30% of non-padding tokens |
| [screen_scale.py](https://github.com/adenzhou1350/minimind-diffusion/blob/0a4e5d3cf5ff3a78381d126ce131702352a79b1b/screen_scale.py) | 512×6: 3,000 steps, lr 1e-3 / 3e-4; 256×4: 5,000 steps, lr 1e-3; batch 64 | Train `[0, 20000)`; held-out `[20000, 20200)` | Randomly mask about 20% of non-padding tokens |
| [screen_init.py](https://github.com/adenzhou1350/minimind-diffusion/blob/0a4e5d3cf5ff3a78381d126ce131702352a79b1b/screen_init.py) | 256×4, batch 64, 800 steps per config, lr 1e-3; random / mean / zero initialization | Train `[0, 8000)`; held-out `[8000, 8100)` | Randomly mask about 20% of non-padding tokens |

All three read `dataset/pretrain_t2t_mini.jsonl`, load a tokenizer from `model/`, use sequence length 128 and call `torch.manual_seed(0)`. Their device is hardcoded to `cuda`. They are training scripts, **not CPU smoke tests**, and do not parse command-line arguments.

The four configurations in `screen_loss.py` are:

1. Additional `1/t` weighting, `t ~ U(1e-4, 1.0)`.
2. Uniform weighting, `t ~ U(1e-4, 1.0)`.
3. Additional `1/t` weighting, `t ~ U(0.1, 0.7)`.
4. Uniform weighting, `t ~ U(0.1, 0.5)`.

Each first averages CE over masked tokens within a sequence, then optionally multiplies by `1/t`. This compares a specific repository implementation; it does not establish that the official LLaDA method fails. The main training implementation currently uses the fourth form.

Reconstruction accuracy is the number of correct argmax token predictions at masked positions divided by the total number of masked tokens. It is not question-answer accuracy. Evaluation resamples masks each time and may include BOS/EOS; fixed masks, data hashes and repeated-seed intervals were not published. The scripts use different held-out ranges and masking probabilities, so their numbers should not be combined in one comparison.

## Keeping evidence for the next run

Prepare dependencies, tokenizer, corpus and an available CUDA device first. If you choose to run screening, this PowerShell example saves its output. It performs real training and is not a promise to reproduce any historical number above.

```powershell
New-Item -ItemType Directory -Force out/evidence
git rev-parse HEAD > out/evidence/source.txt
python -m pip freeze > out/evidence/packages.txt
Get-FileHash model/tokenizer.json, model/tokenizer_config.json, dataset/pretrain_t2t_mini.jsonl -Algorithm SHA256 |
    Format-List | Out-File out/evidence/assets.txt
python -u screen_loss.py > out/evidence/screen-loss.log 2>&1
```

`out/` is gitignored; saving results does not publish them. A result should include the source revision and local changes, exact command, hardware/software versions, data/tokenizer fingerprints, training settings, evaluation split and masks/seeds, and full output. For recipe comparisons, hold other settings fixed, vary the factor under study and save repeated runs.

If old logs are recovered, identify their configuration before adding them to the source table. A new run should carry its own date and version, rather than being presented as the missing historical evidence.
