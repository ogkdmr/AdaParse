# Nougat on Aurora XPU (Personal Setup Notes)

This is a reproducible setup for running **Nougat-only parsing** with AdaParse on Aurora compute nodes using the framework clone environment.

## 1) Create and activate environment

```bash
module load frameworks
conda create -n adaparse-env --clone /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1
mamba activate adaparse-env
```

## 2) Clone repo and install without overriding framework torch

```bash
git clone git@github.com:7shoe/AdaParse.git
cd AdaParse

pip install --upgrade pip setuptools wheel

# Critical: do not let pip resolve and replace torch/xpu stack from the cloned env
pip install -e . --no-deps

# Nougat-oriented extras without forcing torch install
pip install -e '.[nougat-xpu]'
```

## 3) Configure checkpoint

Either export directly:

```bash
export NOUGAT_CHECKPOINT="/lus/flare/projects/LUCID/ogokdemir/checkpoint"
```

Or place it in `.adaparse.env` at repo root:

```bash
NOUGAT_CHECKPOINT="/lus/flare/projects/LUCID/ogokdemir/checkpoint"
```

## 4) Quick sanity checks

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("xpu available:", hasattr(torch, "xpu") and torch.xpu.is_available())
PY
```

## 5) Run Nougat smoke script (no pytest)

Use the standalone script:

```bash
python scripts/nougat_smoke.py \
  --pdf-dir "/home/ogokdemir/AdaParse/data/small2" \
  --max-pdfs 3 \
  --log-dir "./tests/logs" \
  --out-jsonl "./tests/logs/nougat_smoke_output/test_output.jsonl" \
  --mmd_out "./tests/logs/nougat_smoke_output/mmds/"
 
```

Alternative: pass explicit checkpoint:

```bash
python scripts/nougat_smoke.py \
  --checkpoint "/absolute/path/to/nougat/checkpoint" \
  --pdf-dir "/home/ogokdemir/AdaParse/data/small2"
```

## 6) Where output appears

- Parsed JSONL: `tests/logs/nougat_smoke_output.jsonl`
- Nougat logs: `tests/logs/adaparse_nougat.log` (and related logs in `tests/logs`)

Success looks like:

- Script exits with code `0`
- Console prints `[ok] parsed documents: ...`
- JSONL file contains one JSON object per parsed PDF with `path`, `text`, `metadata`, `parser`

## 7) Troubleshooting

- **Checkpoint error**: ensure `NOUGAT_CHECKPOINT` points to an existing local checkpoint directory.
- **No PDFs found**: verify `--pdf-dir` contains `*.pdf` files.
- **Torch got replaced**: reinstall in this order again:
  1. recreate/activate cloned env
  2. `pip install -e . --no-deps`
  3. `pip install -e '.[nougat-xpu]'`
- **Still suspicious torch build**:
  ```bash
  python - <<'PY'
  import torch
  print(torch.__file__)
  print(torch.__version__)
  print("xpu:", hasattr(torch, "xpu") and torch.xpu.is_available())
  PY
  ```

## 8) Nougat-only reminder

For Parsl workflow configs, set parser name to:

```yaml
parser_settings:
  name: nougat
```

Do not use `name: adaparse` if you want to bypass classifier/delegation logic and force everything through Nougat.