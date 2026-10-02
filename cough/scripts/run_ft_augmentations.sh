#!/bin/bash
# Run from the cough project root (the directory containing pyproject.toml).
# Keeps the Mac awake while this script runs when caffeinate is available.
set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-data/coughvid_20211012_processed/ft_aug_logs}"
mkdir -p "$LOG_DIR"

AUGS=(
  gain_noise
  gain_noise_pitch
  gain_noise_time_shift
  specaugment
)

if command -v caffeinate >/dev/null 2>&1 && [ "${FT_AUG_CAFFEINATED:-0}" != "1" ]; then
  export FT_AUG_CAFFEINATED=1
  exec caffeinate -i "$0" "$@"
fi

printf 'FT augmentation batch started: %s\n' "$(date)"
printf 'Logs: %s\n\n' "$LOG_DIR"

failures=0
for aug in "${AUGS[@]}"; do
  model_name="p3_model1_ft_${aug}"
  log_file="$LOG_DIR/${model_name}.log"

  echo "============================================================" | tee -a "$log_file"
  echo "START $model_name : $(date)" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  if "$PYTHON_BIN" -m psupr_proj \
      --train-model --model model1 --profile p3 --augmentation "$aug" \
      2>&1 | tee -a "$log_file"; then
    echo "Training completed: $model_name" | tee -a "$log_file"
  else
    echo "TRAINING FAILED: $model_name" | tee -a "$log_file"
    failures=$((failures + 1))
    continue
  fi

  if "$PYTHON_BIN" -m psupr_proj \
      --evaluate-model "$model_name" --evaluation-split test \
      2>&1 | tee -a "$log_file"; then
    echo "Evaluation completed: $model_name" | tee -a "$log_file"
  else
    echo "EVALUATION FAILED: $model_name" | tee -a "$log_file"
    failures=$((failures + 1))
  fi

  echo "END $model_name : $(date)" | tee -a "$log_file"
  echo | tee -a "$log_file"
done

printf 'FT augmentation batch finished: %s\n' "$(date)"
printf 'Failures: %d\n' "$failures"
exit "$failures"
