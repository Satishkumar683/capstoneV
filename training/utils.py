"""
Windows Utilities for Gait Recognition:
  - Hardware Detection & Configuration
  - Auto-Extraction of Casia-B.zip
  - tf.data Pipeline with Pathlib
  - Test-Time Augmentation (TTA)
  - Batch Normalization Domain Adaptation
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
BUFFER_SIZE = 1000

# TTA transforms (identity, horizontal flip, and slight contrast scaling - strictly preserving upright stance)
TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: tf.image.flip_left_right(x),
    lambda x: tf.clip_by_value(x * 1.05, 0.0, 1.0),
    lambda x: tf.image.flip_left_right(tf.clip_by_value(x * 1.05, 0.0, 1.0))
]

def configure_windows_environment(force_cpu: bool = False):
    """
    Configures TensorFlow on Windows.
    Enables GPU memory growth to prevent CUDA allocation failure.
    """
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[*] Hardware: Running in CPU mode (forced).")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[+] Windows GPU Acceleration Enabled: Found {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"[-] GPU Warning: {e}")
    else:
        print("[*] Hardware: No NVIDIA GPU detected by TensorFlow. Running on multi-core CPU.")


def ensure_dataset_extracted(zip_path: Path, target_dir: Path) -> Path:
    """
    Checks if CASIA-B dataset is extracted; if not, extracts Casia-B.zip.
    Returns the resolved folder containing subject folders (001, 002, etc.).
    """
    def find_subjects_folder(base: Path) -> Optional[Path]:
        if not base.exists():
            return None
        if any(p.is_dir() and p.name.isdigit() for p in base.iterdir()):
            return base.resolve()
        for sub in base.rglob("*"):
            if sub.is_dir() and any(p.is_dir() and p.name.isdigit() for p in sub.iterdir()):
                return sub.resolve()
        return None

    # Check if target directory already has subject folders
    found = find_subjects_folder(target_dir)
    if found:
        return found

    # If zip_path does not exist directly, look for any matching zip in project root
    if not zip_path.exists():
        candidates = [z for z in zip_path.parent.glob("*.zip") if "capstone" not in z.name.lower()]
        for c in candidates:
            if any(k in c.name.lower() for k in ["casia", "gait", "dataset"]):
                zip_path = c
                break
        else:
            if candidates:
                zip_path = candidates[0]

    if zip_path.exists():
        print(f"[*] Extracting {zip_path.name} to {target_dir}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        print("[+] Dataset extraction complete!")
        
        found = find_subjects_folder(target_dir)
        if found:
            return found
        return target_dir.resolve()

    print(f"[!] Warning: Neither {target_dir} nor {zip_path} were found.")
    return target_dir


def load_angle_dataset(
    dataset_dir: Path,
    subjects: List[str],
    condition: str,
    angle: str,
    batch_size: int = BATCH_SIZE
) -> Tuple[Optional[tf.data.Dataset], int]:
    """
    Loads images for specific subjects, condition, and view angle.
    Path structure: dataset_dir / <sid> / <condition> / <angle> / *.png
    """
    paths, labels = [], []
    idx_map = {sid: i for i, sid in enumerate(subjects)}

    for sid in subjects:
        folder = dataset_dir / sid / condition / angle
        if folder.is_dir():
            for img_file in sorted(folder.iterdir()):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    paths.append(str(img_file.resolve()))
                    labels.append(idx_map[sid])

    if not paths:
        return None, len(subjects)

    def decode(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=1, expand_animations=False)
        img.set_shape([None, None, 1])
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.cast(img, tf.float32) / 255.0
        lbl = tf.one_hot(label, depth=len(subjects))
        return img, lbl

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(BUFFER_SIZE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(subjects)


def adapt_batchnorm(model: tf.keras.Model, ds: Optional[tf.data.Dataset], steps: int = 10):
    """
    Updates running mean and variance of BatchNorm layers on target domain images.
    """
    if ds is not None:
        for imgs, _ in ds.take(steps):
            _ = model(imgs, training=True)


def predict_with_tta(model: tf.keras.Model, images: np.ndarray) -> np.ndarray:
    """
    Runs Test-Time Augmentation (TTA) and returns mean predicted probabilities.
    """
    preds = []
    for tfm in TTA_TRANSFORMS:
        augmented = tfm(images)
        preds.append(model.predict(augmented, verbose=0))
    return np.mean(preds, axis=0)
