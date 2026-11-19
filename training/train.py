"""
Two-Stage Cross-View Gait Training Protocol with Few-Shot Fine-Tuning
"""

import sys
from pathlib import Path
from typing import List, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from models.resnet50_cbam import build_crossview_gait_model
from .utils import load_angle_dataset

VIEW_ANGLES = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]

def train_single_stage(model: tf.keras.Model, ds: Optional[tf.data.Dataset], epochs: int = 25) -> tf.keras.Model:
    if ds is None:
        print("[-] Error: Cannot train on empty dataset.")
        return model

    n = tf.data.experimental.cardinality(ds).numpy()
    if n <= 1:
        model.fit(ds, epochs=epochs)
        return model

    v = max(1, int(0.2 * n))
    val_ds = ds.take(v)
    tr_ds = ds.skip(v).map(lambda i, l: (tf.image.random_flip_left_right(i), l), tf.data.AUTOTUNE)

    callbacks = [EarlyStopping("val_loss", patience=8, restore_best_weights=True)]
    model.fit(tr_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model


def run_training_pipeline(
    dataset_dir: Path,
    checkpoints_dir: Path,
    epochs_stage1: int = 25,
    epochs_stage2: int = 25,
    epochs_fs: int = 10,
    batch_size: int = 16
) -> tf.keras.Model:
    """
    Executes the complete two-stage transfer learning + few-shot adaptation pipeline.
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STAGE 1: Subjects 001 - 074 (All Views & Covariates)
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("STAGE 1: Pre-training on Subjects 001 to 074 (All Conditions)")
    print("=" * 65)
    subjects1 = [f"{i:03d}" for i in range(1, 75)]
    views1 = ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02"]

    ds1 = None
    for angle in VIEW_ANGLES:
        for condition in views1:
            d, _ = load_angle_dataset(dataset_dir, subjects1, condition, angle, batch_size=batch_size)
            if d is not None:
                ds1 = d if ds1 is None else ds1.concatenate(d)

    if ds1 is None:
        print(f"[-] No images found in: {dataset_dir}")
        print("[-] Verify that CASIA-B dataset folders (001, 002, ...) are present.")
        sys.exit(1)

    model1 = build_crossview_gait_model(num_classes=len(subjects1))
    model1 = train_single_stage(model1, ds1, epochs=epochs_stage1)
    stage1_weights = checkpoints_dir / "stage1_model.weights.h5"
    model1.save_weights(str(stage1_weights))
    print(f"[+] Saved Stage 1 weights to: {stage1_weights}")

    # ---------------------------------------------------------
    # STAGE 2: Subjects 075 - 124 (Normal Gallery nm-01..04)
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("STAGE 2: Target Training on Subjects 075 to 124 (nm-01 to nm-04)")
    print("=" * 65)
    subjects2 = [f"{i:03d}" for i in range(75, 125)]
    views2 = ["nm-01", "nm-02", "nm-03", "nm-04"]

    ds2 = None
    for angle in VIEW_ANGLES:
        for condition in views2:
            d, _ = load_angle_dataset(dataset_dir, subjects2, condition, angle, batch_size=batch_size)
            if d is not None:
                ds2 = d if ds2 is None else ds2.concatenate(d)

    model2 = build_crossview_gait_model(num_classes=len(subjects2))
    # Transfer backbone weights from Stage 1
    for l1, l2 in zip(model1.layers, model2.layers):
        if l1.name == l2.name and l1.get_weights() and l2.get_weights():
            l2.set_weights(l1.get_weights())

    model2 = train_single_stage(model2, ds2, epochs=epochs_stage2)
    stage2_weights = checkpoints_dir / "stage2_model.weights.h5"
    model2.save_weights(str(stage2_weights))
    print(f"[+] Saved Stage 2 weights to: {stage2_weights}")

    # ---------------------------------------------------------
    # FEW-SHOT FINE-TUNING: Covariate Adaptation on bg/cl
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("FEW-SHOT FINE-TUNING: Bag (bg) and Coat (cl) Adaptation")
    print("=" * 65)
    views_fs = ["bg-01", "bg-02", "cl-01", "cl-02"]
    ds_fs = None
    for angle in VIEW_ANGLES:
        for condition in views_fs:
            d, _ = load_angle_dataset(dataset_dir, subjects2, condition, angle, batch_size=batch_size)
            if d is not None:
                ds_fs = d if ds_fs is None else ds_fs.concatenate(d)

    if ds_fs is not None:
        # Freeze backbone, fine-tune embedding & classifier head
        for layer in model2.layers:
            layer.trainable = False
        for layer in model2.layers[-4:]:
            layer.trainable = True

        model2.compile(
            optimizer=Adam(5e-6),
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"]
        )
        model2.fit(ds_fs, epochs=epochs_fs)
        final_weights = checkpoints_dir / "final_fewshot_model.weights.h5"
        model2.save_weights(str(final_weights))
        print(f"[+] Saved Final Few-Shot model weights to: {final_weights}")

    return model2
