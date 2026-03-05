#!/usr/bin/env bash

set -eo pipefail

# Keep this aligned with what works interactively:
#   module load frameworks
#   mamba activate adaparse-env

AP_ENV_NAME="${ADAPARSE_ENV_NAME:-adaparse-env}"
FRAMEWORKS_MODULE="${ADAPARSE_FRAMEWORKS_MODULE:-frameworks}"
HF_HOME_DEFAULT="/lus/flare/projects/LUCID/${USER}/hf_cache"
AP_ENV_PREFIX_DEFAULT="/lus/flare/projects/LUCID/${USER}/miniforge3/envs/${AP_ENV_NAME}"
AP_ENV_PREFIX="${ADAPARSE_ENV_PREFIX:-$AP_ENV_PREFIX_DEFAULT}"

# Guard against accidental inheritance of module-like values as env names.
if [[ "${AP_ENV_NAME}" == frameworks/* ]]; then
  echo "[warn] ADAPARSE_ENV_NAME looks like a module name (${AP_ENV_NAME}); using adaparse-env."
  AP_ENV_NAME="adaparse-env"
  AP_ENV_PREFIX="/lus/flare/projects/LUCID/${USER}/miniforge3/envs/${AP_ENV_NAME}"
fi

module purge
module load "${FRAMEWORKS_MODULE}"

# oneAPI is often already configured by the module; this is a harmless fallback.
if [[ -f "/opt/aurora/26.26.0/oneapi/setvars.sh" ]]; then
  source /opt/aurora/26.26.0/oneapi/setvars.sh --force >/dev/null 2>&1 || true
elif [[ -f "/opt/aurora/25.190.0/oneapi/setvars.sh" ]]; then
  source /opt/aurora/25.190.0/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

if command -v mamba >/dev/null 2>&1; then
  # Some module hooks reference unset zsh vars; avoid nounset-like failures.
  export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"
  eval "$(mamba shell hook --shell bash)"
  if [[ -d "${AP_ENV_PREFIX}" ]]; then
    if ! mamba activate "${AP_ENV_PREFIX}"; then
      echo "[warn] mamba activate ${AP_ENV_PREFIX} failed; trying env name ${AP_ENV_NAME}."
      mamba activate "${AP_ENV_NAME}"
    fi
  elif ! mamba activate "${AP_ENV_NAME}"; then
    echo "[warn] mamba activate ${AP_ENV_NAME} failed."
    exit 2
  fi
elif command -v conda >/dev/null 2>&1; then
  export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"
  eval "$(conda shell.bash hook)"
  if [[ -d "${AP_ENV_PREFIX}" ]]; then
    if ! conda activate "${AP_ENV_PREFIX}"; then
      echo "[warn] conda activate ${AP_ENV_PREFIX} failed; trying env name ${AP_ENV_NAME}."
      conda activate "${AP_ENV_NAME}"
    fi
  elif ! conda activate "${AP_ENV_NAME}"; then
    echo "[warn] conda activate ${AP_ENV_NAME} failed."
    exit 2
  fi
else
  echo "[error] neither mamba nor conda found on PATH" >&2
  exit 2
fi

export HF_HOME="${ADAPARSE_HF_HOME:-${HF_HOME:-$HF_HOME_DEFAULT}}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE-1}"

mkdir -p "$HOME/bin"
cat > "$HOME/bin/interchange.py" <<'PY'
#!/usr/bin/env python3
import runpy
import sys
sys.exit(
    runpy.run_module("parsl.executors.high_throughput.interchange", run_name="__main__")
)
PY
chmod +x "$HOME/bin/interchange.py"

echo "=== Worker bootstrap summary ==="
echo "frameworks module: ${FRAMEWORKS_MODULE}"
echo "env name: ${AP_ENV_NAME}"
echo "env prefix: ${AP_ENV_PREFIX}"
echo "HF_HOME: ${HF_HOME}"
which python && python --version
python -c "import torch; print('Torch:', torch.__version__, 'from', torch.__file__)"
python -c "import parsl; print('Parsl version:', parsl.__version__, 'from', parsl.__file__)"
python -c "import adaparse; print('AdaParse version:', adaparse.__version__, 'from', adaparse.__file__)"
python - <<'PY' || true
import torch
has_xpu = hasattr(torch, "xpu")
avail = bool(has_xpu and torch.xpu.is_available())
count = torch.xpu.device_count() if has_xpu else 0
print("Torch XPU availability:", avail, "count:", count)
PY
