"""
Evaluation Module:
  - Multi-Angle Cross-View Accuracy Matrix
  - Test-Time Augmentation (TTA) + BatchNorm Domain Adaptation
  - Excel Export with Embedded Plots
"""

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as ExcelImage
import tensorflow as tf

from .utils import load_angle_dataset, adapt_batchnorm, predict_with_tta

VIEW_ANGLES = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]

def run_evaluation_pipeline(
    model: tf.keras.Model,
    dataset_dir: Path,
    results_dir: Path,
    subjects: List[str] = None
):
    """
    Evaluates test conditions (NM, BG, CL) across all 11 angles.
    Generates Excel reports and accuracy curve figures.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    if subjects is None:
        available_subjects = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()]) if dataset_dir.exists() else []
        subjects = [s for s in [f"{i:03d}" for i in range(75, 125)] if s in available_subjects]
        if not subjects and available_subjects:
            subjects = available_subjects[max(1, len(available_subjects) // 2):] or available_subjects
        if not subjects:
            subjects = [f"{i:03d}" for i in range(75, 125)]

    test_sets = {
        "nm_05_06": ["nm-05", "nm-06"],
        "bg_01_02": ["bg-01", "bg-02"],
        "cl_01_02": ["cl-01", "cl-02"]
    }

    for name, views in test_sets.items():
        print(f"\n[+] Evaluating Condition: [{name}] across 11 view angles...")

        # Domain adaptation on target views
        full_ds = None
        for v in views:
            ds, _ = load_angle_dataset(dataset_dir, subjects, v, VIEW_ANGLES[0])
            if ds is not None:
                full_ds = ds if full_ds is None else full_ds.concatenate(ds)

        if full_ds is not None:
            adapt_batchnorm(model, full_ds, steps=10)

        records = []
        for train_angle in VIEW_ANGLES:
            row = {"Train_Angle": train_angle}
            for test_angle in VIEW_ANGLES:
                tds = None
                for v in views:
                    ds, _ = load_angle_dataset(dataset_dir, subjects, v, test_angle)
                    if ds is not None:
                        tds = ds if tds is None else tds.concatenate(ds)

                y_true, y_pred = [], []
                if tds is not None:
                    for imgs, lbls in tds:
                        preds = predict_with_tta(model, imgs.numpy())
                        y_true.extend(np.argmax(lbls.numpy(), axis=1))
                        y_pred.extend(np.argmax(preds, axis=1))
                    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
                else:
                    acc = 0.0

                row[test_angle] = acc
            records.append(row)

        df = pd.DataFrame(records).set_index("Train_Angle").T

        # Plot Accuracy Matrix
        plot_path = results_dir / f"accuracy_{name}.png"
        df.plot(figsize=(10, 6), marker="o")
        plt.title(f"Cross-View Accuracy with TTA & BN Adapt ({name})")
        plt.xlabel("Test View Angle (degrees)")
        plt.ylabel("Recognition Accuracy")
        plt.grid(True)
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Excel Export with Embedded Image
        excel_path = results_dir / f"results_{name}.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = name
        for r in dataframe_to_rows(df.reset_index(), index=False, header=True):
            ws.append(r)

        if plot_path.exists():
            img = ExcelImage(str(plot_path.resolve()))
            img.anchor = "H2"
            ws.add_image(img)

        wb.save(excel_path)
        print(f"[+] Saved results to: {excel_path} and {plot_path}")
