#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/train_relevance_gpu.sh [human-reviewed-train.csv ...]

Prepare the chronological relevance dataset, train a new calibrated checkpoint on
CUDA, and evaluate both the raw checkpoint and the inference policy.

Optional environment variables:
  MODEL_VERSION=v2              Output suffix (policy-relevance-gate-v2)
  OUTPUT_DIR=path               New model directory; must not already exist
  WORK_DIR=path                 Persistent generated data/log directory
  BASE_MODEL=path               Backbone/checkpoint used to initialize training
  PYTHON_VERSION=3.12           Python installed by uv for the isolated environment
  VENV_DIR=.venv-train          Isolated training virtual environment
  EPOCHS=4                      Number of training epochs
  BATCH_SIZE=8                  Per-device train/evaluation batch size
  LEARNING_RATE=1e-5            Optimizer learning rate
  CLASS_WEIGHT_CAP=8            Maximum inverse-frequency class weight
  SEED=42                       Reproducibility seed
  PREPARE_DATA=1                Set to 0 to reuse CSVs already in WORK_DIR/data
  RUN_TESTS=1                   Set to 0 to skip the repository tests
  RESUME_FROM_CHECKPOINT=path   Resume an interrupted Trainer checkpoint
  ALLOW_CPU_TRAINING=0          Set to 1 only when CPU training is intentional

Examples:
  scripts/train_relevance_gpu.sh
  MODEL_VERSION=v3 BATCH_SIZE=16 scripts/train_relevance_gpu.sh reviewed.csv
  OUTPUT_DIR=models/policy-relevance-gate-v2 \
    RESUME_FROM_CHECKPOINT=models/policy-relevance-gate-v2/checkpoint-1182 \
    PREPARE_DATA=0 scripts/train_relevance_gpu.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

model_version="${MODEL_VERSION:-v2}"
if [[ ! "$model_version" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "MODEL_VERSION contains unsupported characters: $model_version" >&2
    exit 2
fi

output_dir="${OUTPUT_DIR:-models/policy-relevance-gate-${model_version}}"
work_dir="${WORK_DIR:-.training/policy-relevance-gate-${model_version}}"
base_model="${BASE_MODEL:-models/policy-signal-baseline-v2-large}"
venv_dir="${VENV_DIR:-.venv-train}"
python_version="${PYTHON_VERSION:-3.12}"
epochs="${EPOCHS:-4}"
batch_size="${BATCH_SIZE:-8}"
learning_rate="${LEARNING_RATE:-1e-5}"
class_weight_cap="${CLASS_WEIGHT_CAP:-8}"
seed="${SEED:-42}"
prepare_data="${PREPARE_DATA:-1}"
run_tests="${RUN_TESTS:-1}"
resume_checkpoint="${RESUME_FROM_CHECKPOINT:-}"
allow_cpu="${ALLOW_CPU_TRAINING:-0}"

for toggle in "$prepare_data" "$run_tests" "$allow_cpu"; do
    if [[ "$toggle" != 0 && "$toggle" != 1 ]]; then
        echo "PREPARE_DATA, RUN_TESTS, and ALLOW_CPU_TRAINING must be 0 or 1." >&2
        exit 2
    fi
done

on_error() {
    local exit_code=$?
    echo "Training failed at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
    echo "Generated data and checkpoints were preserved under: $work_dir and $output_dir" >&2
    exit "$exit_code"
}
trap on_error ERR

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [[ -e "$output_dir" && -z "$resume_checkpoint" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    echo "Choose another MODEL_VERSION/OUTPUT_DIR or set RESUME_FROM_CHECKPOINT." >&2
    exit 1
fi
if [[ -n "$resume_checkpoint" && ! -d "$resume_checkpoint" ]]; then
    echo "Resume checkpoint does not exist: $resume_checkpoint" >&2
    exit 1
fi

mkdir -p "$work_dir" "$(dirname -- "$output_dir")"
data_dir="$work_dir/data"
mkdir -p "$data_dir"

python_bin="$venv_dir/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Creating Python $python_version environment in $venv_dir"
    uv venv --python "$python_version" "$venv_dir"
fi

echo "Installing project and training dependencies"
uv pip install --python "$python_bin" -e '.[train,dev]'

if [[ ! -f "$base_model/config.json" ]]; then
    echo "Base model is incomplete or missing: $base_model/config.json" >&2
    exit 1
fi

weight_file=""
for candidate in "$base_model/model.safetensors" "$base_model/pytorch_model.bin"; do
    if [[ -f "$candidate" ]]; then
        weight_file="$candidate"
        break
    fi
done

needs_lfs=0
if [[ -z "$weight_file" ]]; then
    needs_lfs=1
elif head -c 128 "$weight_file" | grep -aq "git-lfs.github.com/spec"; then
    needs_lfs=1
fi

if [[ "$needs_lfs" == 1 ]]; then
    if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
        echo "The base-model weights require Git LFS, but git-lfs is unavailable." >&2
        exit 1
    fi
    echo "Downloading base-model weights through Git LFS"
    git lfs pull --include="${base_model#./}/**"
    for candidate in "$base_model/model.safetensors" "$base_model/pytorch_model.bin"; do
        if [[ -f "$candidate" ]]; then
            weight_file="$candidate"
            break
        fi
    done
fi

if [[ -z "$weight_file" || ! -s "$weight_file" ]]; then
    echo "No usable model weights found under: $base_model" >&2
    exit 1
fi
if head -c 128 "$weight_file" | grep -aq "git-lfs.github.com/spec"; then
    echo "Git LFS left a pointer instead of the real weights: $weight_file" >&2
    exit 1
fi

echo "Checking PyTorch and the training device"
if ! "$python_bin" - "$allow_cpu" <<'PY'
import sys

import torch

allow_cpu = sys.argv[1] == "1"
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
elif not allow_cpu:
    raise SystemExit(
        "CUDA is unavailable. Install a CUDA-enabled PyTorch build or set "
        "ALLOW_CPU_TRAINING=1 intentionally."
    )
PY
then
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
        --format=csv,noheader
fi

if [[ "$run_tests" == 1 ]]; then
    echo "Running repository tests before training"
    "$python_bin" -m pytest -q
    "$python_bin" -m ruff check .
fi

regression_path="$repo_root/data/relevance_regression.example.csv"
prepare_command=(
    "$python_bin" -m policy_signal.prepare_relevance_data
    --output "$data_dir"
)
for supplemental_path in "$@"; do
    if [[ ! -f "$supplemental_path" ]]; then
        echo "Supplemental CSV does not exist: $supplemental_path" >&2
        exit 1
    fi
    supplemental_absolute="$(realpath "$supplemental_path")"
    if [[ "$supplemental_absolute" == "$regression_path" ]]; then
        echo "Do not train on the reserved regression set: $supplemental_path" >&2
        exit 1
    fi
    prepare_command+=(--supplemental-train "$supplemental_path")
done

if [[ "$prepare_data" == 1 ]]; then
    echo "Preparing chronological train/validation/test splits"
    "${prepare_command[@]}" | tee "$work_dir/dataset-preparation.log"
else
    echo "Reusing prepared data from $data_dir"
fi

train_csv="$data_dir/relevance_train.csv"
validation_csv="$data_dir/relevance_validation.csv"
test_csv="$data_dir/relevance_test.csv"
dataset_manifest="$data_dir/relevance_dataset_manifest.json"
for required_file in "$train_csv" "$validation_csv" "$test_csv" "$dataset_manifest"; do
    if [[ ! -s "$required_file" ]]; then
        echo "Required prepared-data file is missing or empty: $required_file" >&2
        exit 1
    fi
done

train_command=(
    "$python_bin" -m policy_signal.train_relevance
    --train "$train_csv"
    --validation "$validation_csv"
    --test "$test_csv"
    --dataset-manifest "$dataset_manifest"
    --base-model "$base_model"
    --output "$output_dir"
    --epochs "$epochs"
    --batch-size "$batch_size"
    --learning-rate "$learning_rate"
    --class-weight-cap "$class_weight_cap"
    --seed "$seed"
)
if [[ -n "$resume_checkpoint" ]]; then
    train_command+=(--resume-from-checkpoint "$resume_checkpoint")
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$work_dir/huggingface-datasets-cache}"

echo "Starting relevance training"
printf 'Command:'
printf ' %q' "${train_command[@]}"
printf '\n'
"${train_command[@]}" 2>&1 | tee "$work_dir/training.log"

for artifact in \
    "$output_dir/config.json" \
    "$output_dir/model.safetensors" \
    "$output_dir/tokenizer.json" \
    "$output_dir/training_manifest.json" \
    "$output_dir/relevance_calibration.json" \
    "$output_dir/test_evaluation.json"; do
    if [[ ! -s "$artifact" ]]; then
        echo "Training completed without required artifact: $artifact" >&2
        exit 1
    fi
done

echo "Evaluating the raw checkpoint on the reserved repository regression set"
"$python_bin" -m policy_signal.evaluate_relevance \
    --data "$regression_path" \
    --model "$output_dir" \
    --no-procedural-override \
    --output "$output_dir/repository_relevance_regression.baseline.json" \
    > "$work_dir/relevance-regression-baseline.log"

echo "Evaluating the complete inference policy"
"$python_bin" -m policy_signal.evaluate_relevance \
    --data "$regression_path" \
    --model "$output_dir" \
    --output "$output_dir/repository_relevance_regression.json" \
    > "$work_dir/relevance-regression-inference.log"

{
    echo "trained_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    if git rev-parse HEAD >/dev/null 2>&1; then
        echo "git_commit=$(git rev-parse HEAD)"
    fi
    echo
    echo "[python_packages]"
    uv pip freeze --python "$python_bin"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo
        echo "[nvidia_smi]"
        nvidia-smi --query-gpu=name,driver_version,memory.total \
            --format=csv,noheader
    fi
} > "$output_dir/training_environment.txt"

"$python_bin" - "$output_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
test = json.loads((output / "test_evaluation.json").read_text(encoding="utf-8"))
calibration = json.loads(
    (output / "relevance_calibration.json").read_text(encoding="utf-8")
)
regression = json.loads(
    (output / "repository_relevance_regression.json").read_text(encoding="utf-8")
)["metrics"]
weights = output / "model.safetensors"
digest = hashlib.sha256()
with weights.open("rb") as source:
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)

print("\nTraining completed successfully")
print(f"Output: {output}")
print(f"Test macro F1: {test['f1_macro']:.6f}")
print(f"Test balanced accuracy: {test['balanced_accuracy']:.6f}")
print(f"Calibration temperature: {calibration['temperature']}")
print(f"Validation NLL: {calibration['validation_nll_before']} -> "
      f"{calibration['validation_nll_after']}")
print(f"Repository regression macro F1: {regression['f1_macro']:.6f}")
print(f"Weights SHA-256: {digest.hexdigest()}")
PY

echo
echo "The existing v1 checkpoint was not modified. Review the metrics before promotion."
echo "When approved, version the new artifact with:"
echo "  git add '$output_dir'"
echo "  git commit -m 'feat: add policy relevance gate ${model_version}'"
echo "  git push"
