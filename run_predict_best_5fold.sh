#!/usr/bin/env bash
# LongiSeg 5-fold inference with checkpoint_best.pth.
# Output: /mnt/hdd2/task2/longiseg/predict_results/mask_results_f{X}_best/

set -euo pipefail

PY_BIN=/home/lq/Projects_qin/run_nnunet_v2/bin
LONGISEG_PREDICT="$PY_BIN/LongiSeg_predict"

export nnUNet_raw=/mnt/hdd2/task2/nnunet/dataset/nnUNet_raw_train_val
export nnUNet_preprocessed=/mnt/hdd2/task2/longiseg/dataset/LongiSeg_preprocessed
export nnUNet_results=/mnt/hdd2/task2/longiseg/dataset/LongiSeg_results
export LongiSeg_raw=$nnUNet_raw
export LongiSeg_preprocessed=$nnUNet_preprocessed
export LongiSeg_results=$nnUNet_results
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR=/mnt/hdd2/task2/nnunet/Dataset007_task2_Ts/imagesTs
PAT_JSON=/mnt/hdd2/task2/longiseg/predict_results/patientsTs_5patients.json
OUT_BASE=/mnt/hdd2/task2/longiseg/predict_results
LOG_DIR=/mnt/hdd2/task2/longiseg
DATASET_ID=7
TRAINER=LongiSegTrainerDiffWeightingRP
PLANS=nnUNetPlans
CONFIG=2d
CHK=checkpoint_best.pth

for FOLD in 0 1 2 3 4; do
    OUT_DIR="$OUT_BASE/mask_results_f${FOLD}_best"
    LOG_FILE="$LOG_DIR/predict_best_${FOLD}.log"
    mkdir -p "$OUT_DIR"
    echo "===== [$(date '+%F %T')] FOLD ${FOLD} -> ${OUT_DIR} =====" | tee -a "$LOG_FILE"

    "$LONGISEG_PREDICT" \
        -i "$INPUT_DIR" \
        -o "$OUT_DIR" \
        -pat "$PAT_JSON" \
        -d "$DATASET_ID" \
        -p "$PLANS" \
        -tr "$TRAINER" \
        -c "$CONFIG" \
        -f "$FOLD" \
        -chk "$CHK" \
        -device cuda \
        2>&1 | tee -a "$LOG_FILE"

    echo "===== [$(date '+%F %T')] FOLD ${FOLD} DONE =====" | tee -a "$LOG_FILE"
done

echo "All 5 folds finished."
