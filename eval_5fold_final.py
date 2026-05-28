"""Fork of eval_5fold_best.py for FINAL-weight (checkpoint_final.pth) predictions.

Identical metric logic (label-only valid-class convention, CROP 289..1631 / 0..1004,
HD95 NaN when pred empty, Organ={26,27,28}, Instr=1..25), but reads the
`mask_results_f{F}_final` folders and writes `5fold_final_eval.csv`.
Original eval_5fold_best.py is untouched.
"""
import csv
import glob
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from medpy import metric
from PIL import Image

PRED_BASE = "/mnt/hdd2/task2/longiseg/predict_results"
LABEL_BASE = "/mnt/hdd2/task2/nnunet"
IMAGE_DIR = "/mnt/hdd2/task2/nnunet/Dataset007_task2_Ts/imagesTs"
OUT_DIR = os.path.join(PRED_BASE, "clean_eval_final")
SUMMARY_CSV = os.path.join(PRED_BASE, "5fold_final_eval.csv")
os.makedirs(OUT_DIR, exist_ok=True)

CROP = (289, 0, 1631, 1004)  # left, upper, right, lower
PATIENTS = ["19", "24", "71", "76", "78"]
FOLDS = [0, 1, 2, 3, 4]
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))
NUM_WORKERS = 32

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("longiseg_final")


def load(p):
    return np.array(Image.open(p))


def calc(pred_bin, label_bin):
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
    return {"IOU": iou, "Dice": dice, "HD95": hd95, "Precision": prec, "Recall": rec}


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
        m = calc((pred == cls).astype(np.uint8), (label == cls).astype(np.uint8))
        if m is None:
            continue
        rows.append({"filename": os.path.basename(image_file), "patient": patient, "class": int(cls), **m})
    return rows


def process_fold(fold):
    pred_dir = os.path.join(PRED_BASE, f"mask_results_f{fold}_final")
    if not os.path.isdir(pred_dir):
        raise SystemExit(f"missing pred dir: {pred_dir}")
    out_csv = os.path.join(OUT_DIR, f"per_image_f{fold}_final.csv")
    tasks = []
    for p in PATIENTS:
        images = sorted(glob.glob(os.path.join(IMAGE_DIR, f"{p}_*.png")))
        tasks.extend([(im, pred_dir, p) for im in images])
    logger.info(f"fold{fold}: {len(tasks)} images")
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_one, tasks)
    n_total = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "patient", "class", "IOU", "Dice", "HD95", "Precision", "Recall"])
        w.writeheader()
        for r in results:
            if not r:
                continue
            for row in r:
                w.writerow(row)
                n_total += 1
    logger.info(f"fold{fold}: {n_total} rows -> {out_csv}")
    return out_csv


def pooled(df, mask=None):
    sub = df if mask is None else df[mask]
    return (float(sub["IOU"].mean()), float(sub["Dice"].mean()), float(sub["HD95"].mean()), len(sub))


def aggregate(fold_csvs):
    rows = []
    for f, csv_path in zip(FOLDS, fold_csvs):
        df = pd.read_csv(csv_path)
        df["HD95"] = df["HD95"].replace([np.inf, -np.inf], np.nan)
        ov = pooled(df)
        og = pooled(df, df["class"].isin(ORGAN))
        it = pooled(df, df["class"].isin(INSTR))
        rows.append({
            "fold": f, "n_rows": ov[3],
            "Mean IOU": ov[0], "Mean Dice": ov[1], "Mean HD95": ov[2],
            "(Organ) Mean IOU": og[0], "(Organ) Mean Dice": og[1], "(Organ) Mean HD95": og[2], "(Organ) n": og[3],
            "(Instr) Mean IOU": it[0], "(Instr) Mean Dice": it[1], "(Instr) Mean HD95": it[2], "(Instr) n": it[3],
        })
    out = pd.DataFrame(rows)
    mean_row = {"fold": "mean", "n_rows": int(out["n_rows"].mean())}
    for col in out.columns:
        if col in ("fold", "n_rows"):
            continue
        mean_row[col] = float(out[col].mean())
    out = pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)
    out.to_csv(SUMMARY_CSV, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n", out.to_string(index=False))
    logger.info(f"Saved 5-fold summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    fold_csvs = [process_fold(f) for f in FOLDS]
    aggregate(fold_csvs)
    logger.info("done (FINAL weights).")
