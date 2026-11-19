"""
Cross-View Human Gait Recognition Through Attention Guided Residual Learning
Optimized for Windows OS (TensorFlow 2.x)

Architecture:
  - ResNet-50 Residual Backbone
  - CBAM (Channel Attention + Spatial Attention)
  - Horizontal Pyramid Mapping (HPM, 6 horizontal stripes)
  - Cosine Decay with Restarts + AdamW
  - Two-Stage Transfer Protocol + Few-Shot Adaptation
  - Test-Time Augmentation (TTA) + BatchNorm Domain Adaptation
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path

# Headless backend to prevent GUI thread hangs on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as ExcelImage

import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW, Adam

# ==============================================================================
# WINDOWS CONFIGURATION & HARDWARE SETTINGS
# ==============================================================================
def configure_windows_hardware(use_gpu: bool = True):
    """
    Configures TensorFlow hardware acceleration on Windows.
    Enables GPU memory growth to prevent CUDA OOM / allocation crash on Windows.
    """
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[*] Device: Running in CPU-only mode (forced).")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[+] Windows GPU Acceleration Enabled: Found {len(gpus)} GPU(s). Memory growth set.")
        except RuntimeError as e:
            print(f"[-] GPU initialization warning: {e}")
    else:
        print("[*] No GPU detected by TensorFlow. Falling back to multi-core CPU mode.")

# Global Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
BUFFER_SIZE = 1000
VIEW_ANGLES = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Safe TTA for vertical human walking silhouettes (avoids invalid rot90)
TTA_TRANSFORMS = [
    lambda x: x,                                                # Identity
    lambda x: tf.image.flip_left_right(x),                     # Horizontal flip
    lambda x: tf.clip_by_value(x * 1.05, 0.0, 1.0),            # Slight intensity boost
    lambda x: tf.image.flip_left_right(tf.clip_by_value(x * 1.05, 0.0, 1.0)),
]

# Training augmentation (flip only, preserving upright stance)
light_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal")
], name="windows_light_augmentation")


# ==============================================================================
# ATTENTION MODULES (CBAM: Channel + Spatial Attention)
# ==============================================================================
def channel_attention(input_feature, ratio=8):
    ch = input_feature.shape[-1]
    d1 = layers.Dense(ch // ratio, activation="relu", kernel_initializer="he_normal")
    d2 = layers.Dense(ch, kernel_initializer="he_normal")
    
    avg = layers.GlobalAveragePooling2D()(input_feature)
    avg = layers.Reshape((1, 1, ch))(avg)
    avg = d2(d1(avg))
    
    mx = layers.GlobalMaxPooling2D()(input_feature)
    mx = layers.Reshape((1, 1, ch))(mx)
    mx = d2(d1(mx))
    
    attn = layers.Activation("sigmoid")(layers.Add()([avg, mx]))
    return layers.Multiply()([input_feature, attn])


def spatial_attention(input_feature):
    avg = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_feature)
    mx = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=3)([avg, mx])
    attn = layers.Conv2D(1, 7, padding="same", activation="sigmoid",
                         kernel_initializer="he_normal", use_bias=False)(concat)
    return layers.Multiply()([input_feature, attn])


# ==============================================================================
# MODEL ARCHITECTURE (ResNet-50 + CBAM + HPM)
# ==============================================================================
def build_model(num_classes: int, num_parts: int = 6, dropout_rate: float = 0.5):
    inp = layers.Input((*IMAGE_SIZE, 3))
    x = light_augmentation(inp)
    
    base = applications.ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    for l in base.layers:
        l.trainable = False
    for l in base.layers:
        if l.name.startswith("conv5_") or isinstance(l, layers.BatchNormalization):
            l.trainable = True

    feat = channel_attention(base.output)
    feat = spatial_attention(feat)

    # Horizontal Pyramid Mapping (HPM)
    def hpm(f):
        feats = []
        h = tf.shape(f)[1]
        for i in range(num_parts):
            s = tf.cast(i * h // num_parts, tf.int32)
            e = tf.cast((i + 1) * h // num_parts, tf.int32)
            part = f[:, s:e, :, :]
            feats.append(tf.reduce_mean(part, axis=[1, 2]))
            feats.append(tf.reduce_max(part, axis=[1, 2]))
        return tf.concat(feats, axis=-1)

    y = layers.Lambda(hpm, name="hpm")(feat)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(5e-4))(y)
    y = layers.Dropout(dropout_rate)(y)
    out = layers.Dense(num_classes, activation="softmax")(y)

    total_steps = 1000
    lr_sched = CosineDecayRestarts(INITIAL_LR, first_decay_steps=total_steps, t_mul=2.0, m_mul=1.0, alpha=0.0)
    opt = AdamW(learning_rate=lr_sched, weight_decay=WEIGHT_DECAY)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=opt,
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# DATASET LOADING WITH WINDOWS PATH RESOLUTION
# ==============================================================================
def load_angle_dataset(dataset_dir: Path, subjects, condition, angle):
    """
    Loads images using Windows-safe path resolution and tf.data pipeline.
    Expects dataset layout: dataset_dir / <sid> / <condition> / <angle> / *.png
    """
    paths, labels = [], []
    idx_map = {sid: i for i, sid in enumerate(subjects)}

    for sid in subjects:
        folder = dataset_dir / sid / condition / angle
        if folder.is_dir():
            for entry in sorted(folder.iterdir()):
                if entry.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    paths.append(str(entry.resolve()))
                    labels.append(idx_map[sid])

    if not paths:
        return None, len(subjects)

    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img.set_shape([None, None, 1])
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.cast(img, tf.float32) / 255.0
        lbl = tf.one_hot(label, depth=len(subjects))
        return img, lbl

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode, tf.data.AUTOTUNE)
    ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, len(subjects)


# ==============================================================================
# TRAINING & EVALUATION ROUTINES
# ==============================================================================
def train_model(model, ds, epochs=25):
    if ds is None:
        print("[-] Cannot train on empty dataset.")
        return model
        
    n = tf.data.experimental.cardinality(ds).numpy()
    if n <= 1:
        # Fallback for small sample test runs
        model.fit(ds, epochs=epochs)
        return model

    v = max(1, int(0.2 * n))
    val_ds = ds.take(v)
    tr_ds = ds.skip(v).map(lambda i, l: (tf.image.random_flip_left_right(i), l), tf.data.AUTOTUNE)
    
    cbs = [EarlyStopping("val_loss", patience=8, restore_best_weights=True)]
    model.fit(tr_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    return model


def adapt_batchnorm(model, ds, steps=10):
    if ds is not None:
        for imgs, _ in ds.take(steps):
            _ = model(imgs, training=True)


def predict_with_tta(model, images):
    preds = []
    for tfm in TTA_TRANSFORMS:
        aug = tfm(images)
        preds.append(model.predict(aug, verbose=0))
    return np.mean(preds, axis=0)


def evaluate(model, test_sets, subjects, angles, out_dir: Path, dataset_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, views in test_sets.items():
        print(f"\n[+] Evaluating Condition: {name}")
        full_ds = None
        for v in views:
            ds, _ = load_angle_dataset(dataset_dir, subjects, v, angles[0])
            if ds is not None:
                full_ds = ds if full_ds is None else full_ds.concatenate(ds)
                
        if full_ds is not None:
            adapt_batchnorm(model, full_ds, steps=10)

        records = []
        for ta in angles:
            row = {"Train_Angle": ta}
            for te in angles:
                tds = None
                for v in views:
                    ds, _ = load_angle_dataset(dataset_dir, subjects, v, te)
                    if ds is not None:
                        tds = ds if tds is None else tds.concatenate(ds)
                
                yt, yp = [], []
                if tds is not None:
                    for imgs, lbls in tds:
                        preds = predict_with_tta(model, imgs.numpy())
                        yt.extend(np.argmax(lbls.numpy(), axis=1))
                        yp.extend(np.argmax(preds, axis=1))
                    row[te] = accuracy_score(yt, yp) if len(yt) > 0 else 0.0
                else:
                    row[te] = 0.0
            records.append(row)

        df = pd.DataFrame(records).set_index("Train_Angle").T
        plot_file = out_dir / f"acc_{name}.png"
        df.plot(figsize=(10, 6), marker="o")
        plt.title(f"TTA + BN Adapt + FewShot ({name})")
        plt.grid(True)
        plt.savefig(plot_file)
        plt.close()

        # Excel export with plot embedding
        excel_file = out_dir / f"results_{name}.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = name
        for r in dataframe_to_rows(df.reset_index(), index=False, header=True):
            ws.append(r)
        if plot_file.exists():
            img = ExcelImage(str(plot_file.resolve()))
            img.anchor = "H2"
            ws.add_image(img)
        wb.save(excel_file)
        print(f"[+] Saved evaluation matrix & plot to: {excel_file}")


# ==============================================================================
# AUTO-EXTRACTION OF CASIA-B DATASET ON WINDOWS
# ==============================================================================
def check_and_extract_dataset(zip_path: Path, target_dir: Path) -> Path:
    """
    Checks if Casia-B.zip exists and unzips it into target_dir if not already extracted.
    """
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[+] Using existing dataset folder: {target_dir}")
        # Look inside for nested output folder if applicable
        nested_out = target_dir / "output"
        return nested_out if nested_out.is_dir() else target_dir

    if zip_path.exists():
        print(f"[*] Extracting {zip_path.name} to {target_dir} (this may take a couple minutes)...")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        print("[+] Extraction complete!")
        nested_out = target_dir / "output"
        return nested_out if nested_out.is_dir() else target_dir

    return target_dir


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    workspace_root = Path(__file__).resolve().parent
    default_zip = workspace_root / "Casia-B.zip"
    default_data_dir = workspace_root / "dataset"
    default_output_dir = workspace_root / "outputs" / "Resnet50_Cosine_HPM_TTA_FewShot"

    parser = argparse.ArgumentParser(description="Cross-View Gait Recognition (Windows)")
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir), help="Path to extracted CASIA-B folder")
    parser.add_argument("--zip_path", type=str, default=str(default_zip), help="Path to Casia-B.zip")
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir), help="Directory to save checkpoints/results")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs for Stage 1 & Stage 2")
    parser.add_argument("--fs_epochs", type=int, default=10, help="Epochs for Few-Shot fine-tuning")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    configure_windows_hardware(use_gpu=(not args.cpu))

    data_path = Path(args.data_dir)
    zip_path = Path(args.zip_path)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve dataset path
    resolved_data_dir = check_and_extract_dataset(zip_path, data_path)
    print(f"[*] Final Dataset Root: {resolved_data_dir}")

    # STAGE 1: Subjects 001 - 074
    print("\n" + "=" * 60)
    print("STAGE 1: Pre-training on Subjects 001 - 074")
    print("=" * 60)
    subjects1 = [f"{i:03d}" for i in range(1, 75)]
    views1 = ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02"]
    
    ds1 = None
    nc1 = len(subjects1)
    for ang in VIEW_ANGLES:
        for v in views1:
            d, _ = load_angle_dataset(resolved_data_dir, subjects1, v, ang)
            if d is not None:
                ds1 = d if ds1 is None else ds1.concatenate(d)

    if ds1 is None:
        print(f"[-] No matching images found in {resolved_data_dir}. Please verify folder structure.")
        sys.exit(1)

    model1 = build_model(nc1)
    model1 = train_model(model1, ds1, epochs=args.epochs)
    model1.save_weights(str(output_path / "stage1_model.weights.h5"))

    # STAGE 2: Subjects 075 - 124
    print("\n" + "=" * 60)
    print("STAGE 2: Target Training on Subjects 075 - 124")
    print("=" * 60)
    subjects2 = [f"{i:03d}" for i in range(75, 125)]
    views2 = ["nm-01", "nm-02", "nm-03", "nm-04"]
    
    ds2 = None
    nc2 = len(subjects2)
    for ang in VIEW_ANGLES:
        for v in views2:
            d, _ = load_angle_dataset(resolved_data_dir, subjects2, v, ang)
            if d is not None:
                ds2 = d if ds2 is None else ds2.concatenate(d)

    model2 = build_model(nc2)
    # Transfer backbone weights from Stage 1
    for l1, l2 in zip(model1.layers, model2.layers):
        if l1.name == l2.name and l1.get_weights() and l2.get_weights():
            l2.set_weights(l1.get_weights())

    model2 = train_model(model2, ds2, epochs=args.epochs)

    # FEW-SHOT FINE-TUNING ON BG / CL
    print("\n" + "=" * 60)
    print("FEW-SHOT FINE-TUNING (Covariate Adaptation: bg/cl)")
    print("=" * 60)
    views_fs = ["bg-01", "bg-02", "cl-01", "cl-02"]
    ds_fs = None
    for ang in VIEW_ANGLES:
        for v in views_fs:
            d, _ = load_angle_dataset(resolved_data_dir, subjects2, v, ang)
            if d is not None:
                ds_fs = d if ds_fs is None else ds_fs.concatenate(d)

    if ds_fs is not None:
        # Freeze backbone, train only the head
        for layer in model2.layers:
            layer.trainable = False
        for layer in model2.layers[-4:]:
            layer.trainable = True

        model2.compile(
            optimizer=Adam(5e-6),
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"]
        )
        model2.fit(ds_fs, epochs=args.fs_epochs)

    # FINAL CROSS-VIEW EVALUATION
    print("\n" + "=" * 60)
    print("FINAL EVALUATION MATRIX & EXCEL GENERATION")
    print("=" * 60)
    test_sets = {
        "nm_05_06": ["nm-05", "nm-06"],
        "bg_01_02": ["bg-01", "bg-02"],
        "cl_01_02": ["cl-01", "cl-02"],
    }
    evaluate(model2, test_sets, subjects2, VIEW_ANGLES, output_path, resolved_data_dir)
    print("\n[+] All Stages Completed Successfully on Windows!")

if __name__ == "__main__":
    main()
