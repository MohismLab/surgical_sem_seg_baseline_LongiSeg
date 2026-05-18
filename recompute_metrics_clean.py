"""Re-evaluate Longiseg with label-only class extraction (matches nnUNet new script).

Differences vs the original LongiSeg_cal_metrics_all_fold.py:
  - valid_classes = labels-only (NOT intersection with pred) → all 5 folds share the
    same prompt set per image, making them strictly comparable.
  - Does NOT emit the class=-1 per-image average row.

Outputs to /mnt/hdd2/task2/longiseg/predict_results/clean_eval/comparison_{f}_{p}_metrics.csv
"""
import csv
import glob
import logging
import os
from multiprocessing import Pool

import numpy as np
from medpy import metric
from PIL import Image

PRED_BASE = "/mnt/hdd2/task2/longiseg/predict_results"
LABEL_BASE = "/mnt/hdd2/task2/nnunet"  # /Dataset007_task2_Ts_{p}/labelsTs/{name}.png
IMAGE_DIR = "/mnt/hdd2/task2/nnunet/Dataset007_task2_Ts/imagesTs"
OUT_DIR = os.path.join(PRED_BASE, "clean_eval")
os.makedirs(OUT_DIR, exist_ok=True)

CROP = (289, 0, 1631, 1004)  # left, upper, right, lower
PATIENTS = ["19", "24", "71", "76", "78"]
FOLDS = [0, 1, 2, 3, 4]
NUM_WORKERS = 32

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("longiseg_clean")


def load(p):
    return np.array(Image.open(p))


def calc(pred_bin, label_bin, cls):
    if np.sum(label_bin) == 0:
        return None
    inter = np.logical_and(pred_bin, label_bin)
    union = np.logical_or(pred_bin, label_bin)
    iou = float(np.sum(inter) / np.sum(union)) if np.sum(union) > 0 else 0.0
    tp = float(np.sum(inter))
    fp = float(np.sum(pred_bin)) - tp
    fn = float(np.sum(label_bin)) - tp
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    try:
        hd95 = float(metric.binary.hd95(pred_bin, label_bin)) if np.sum(pred_bin) > 0 else np.nan
    except Exception:
        hd95 = np.nan
    return {"class": int(cls), "IOU": iou, "Dice": dice, "HD95": hd95,
            "Precision": prec, "Recall": rec}


def process_one(args):
    image_file, pred_dir, patient = args
    name = os.path.basename(image_file).split("_0000.")[0]
    pred_path = os.path.join(pred_dir, f"{name}.png")
    label_path = os.path.join(LABEL_BASE, f"Dataset007_task2_Ts_{patient}", "labelsTs", f"{name}.png")
    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        return None
    try:
        pred = load(pred_path)
        label = load(label_path)
        l, u, r, lo = CROP
        pred = pred[u:lo, l:r]
        label = label[u:lo, l:r]
    except Exception as e:
        logger.error(f"load fail {name}: {e}")
        return None

    label_classes = np.unique(label)
    valid = label_classes[label_classes > 0]
    rows = []
    for cls in valid:
        m = calc((pred == cls).astype(np.uint8), (label == cls).astype(np.uint8), cls)
        if m is not None:
            rows.append({"filename": os.path.basename(image_file), **m})
    return rows


def process_fold_patient(fold, patient):
    pred_dir = os.path.join(PRED_BASE, f"mask_results_{fold}_{patient}")
    if not os.path.isdir(pred_dir):
        logger.warning(f"missing pred dir: {pred_dir}")
        return
    out_csv = os.path.join(OUT_DIR, f"comparison_{fold}_{patient}_metrics.csv")
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, f"{patient}*.png")))
    args = [(im, pred_dir, patient) for im in images]
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_one, args)
    n_total = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "class", "IOU", "Dice", "HD95", "Precision", "Recall"])
        w.writeheader()
        for r in results:
            if r:
                for row in r:
                    w.writerow(row)
                    n_total += 1
    logger.info(f"fold{fold} p{patient}: {n_total} rows -> {out_csv}")


if __name__ == "__main__":
    for f in FOLDS:
        for p in PATIENTS:
            logger.info(f"== fold {f} patient {p} ==")
            process_fold_patient(f, p)
    logger.info("done.")
