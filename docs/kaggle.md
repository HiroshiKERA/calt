# Kaggle Job Runner

This page explains how to run CALT training jobs on Kaggle from your local terminal.

## Install

Install CALT with Kaggle support:

```bash
pip install "calt-x[kaggle]"
```

Or install Kaggle separately:

```bash
pip install kaggle
```

## Authenticate Kaggle CLI

You need Kaggle API credentials before running jobs.

- Option A: set `KAGGLE_API_TOKEN`
- Option B: store token in `~/.kaggle/access_token`
- Option C (legacy): store `kaggle.json` in `~/.kaggle/kaggle.json`

## Run a job

Example with `examples/gf17_addition`:

```bash
calt remote run \
  --source-dir examples/gf17_addition \
  --script train.py \
  --kernel-id <your-kaggle-username>/calt-gf17-addition \
  --output-dir ./kaggle_outputs/gf17_addition \
  --include-path data \
  --accelerator NvidiaTeslaT4
```

Notes:

- `--source-dir` is copied and uploaded as the Kaggle job package.
- `--include-path` is uploaded as a Kaggle Dataset bundle and attached to the kernel.
  This is required because Kaggle script execution does not reliably expose arbitrary
  extra files from the kernel upload directory.
- CALT injects a bootstrap entrypoint so bundled sources (e.g. `calt/`) are
  added to `sys.path` before your training script runs.
- By default, the command waits for completion and downloads outputs to `--output-dir`.
- Use `--no-wait` to submit and exit immediately.
- `calt kaggle run` is still accepted as a backward-compatible alias.

## Common options

- `--gpu/--no-gpu`: enable or disable GPU runtime.
- `--internet/--no-internet`: toggle internet access in Kaggle runtime.
- `--private/--no-private`: toggle kernel visibility.
- `--timeout-sec`: timeout for submission/waiting.
- `--poll-interval-sec`: polling interval while waiting for status.
- `--debug-package`: keep and print packaged job directory + manifest path.
- `--bundle-dataset-id`: specify the dataset id for include bundle upload.
- `--bundle-dataset-title`: title for include bundle dataset.
- `--bundle-dataset-public`: make include bundle dataset public.

## Troubleshooting

- `kaggle CLI not found`: install `kaggle` or `calt-x[kaggle]`.
- Authentication errors: verify token setup in Kaggle settings.
- Job failed on Kaggle: run `kaggle kernels status <username/slug>` and inspect logs/output.
