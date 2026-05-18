"""Unified 5-fold aggregation for Longiseg — pooled all-row averaging.

Same convention as SAM-LoRA / SAM-Med2D / nnUNet:
  Overall = mean over ALL prompt rows
  Organ   = mean over rows whose class in {26,27,28}
  Instr   = mean over rows whose class in {1..25}
"""
import os
import numpy as np
import pandas as pd

BASE = "/mnt/hdd2/task2/longiseg/predict_results/clean_eval"
PATIENTS = [19, 24, 71, 76, 78]
ORGAN = set([26, 27, 28])
INSTR = set(range(1, 26))


def load_fold(f):
    files = [os.path.join(BASE, f"comparison_{f}_{p}_metrics.csv") for p in PATIENTS]
    df = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    df["HD95"] = df["HD95"].replace([np.inf, -np.inf], np.nan)
    return df


def pooled(df, mask=None):
    sub = df if mask is None else df[mask]
    return (
        float(sub["IOU"].mean()),
        float(sub["Dice"].mean()),
        float(sub["HD95"].mean()),
        len(sub),
    )


def main():
    rows = []
    for f in range(5):
        df = load_fold(f)
        ov = pooled(df)
        og = pooled(df, df["class"].isin(ORGAN))
        it = pooled(df, df["class"].isin(INSTR))
        rows.append({
            "fold": f,
            "n_rows": ov[3],
            "Mean IOU": ov[0],
            "Mean Dice": ov[1],
            "Mean HD95": ov[2],
            "(Organ) Mean IOU": og[0],
            "(Organ) Mean Dice": og[1],
            "(Organ) Mean HD95": og[2],
            "(Organ) n": og[3],
            "(Instr) Mean IOU": it[0],
            "(Instr) Mean Dice": it[1],
            "(Instr) Mean HD95": it[2],
            "(Instr) n": it[3],
        })

    out = pd.DataFrame(rows)
    out_path = "/mnt/hdd2/task2/longiseg/predict_results/longiseg_5fold.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")

    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
