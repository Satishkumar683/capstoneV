"""
Cross-View Human Gait Recognition Through Attention Guided Residual Learning
Unified Windows CLI Entry Point
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.utils import (
    configure_windows_environment,
    ensure_dataset_extracted
)
from training.train import run_training_pipeline
from training.test import run_evaluation_pipeline
from models.resnet50_cbam import build_crossview_gait_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-View Gait Recognition (ResNet-50 + CBAM + HPM) - Windows Edition"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "train", "eval"],
        default="all",
        help="Pipeline mode: 'all' (train & eval), 'train' only, or 'eval' only"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT / "dataset"),
        help="Path to folder where CASIA-B is (or will be) extracted"
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=str(PROJECT_ROOT / "Casia-B.zip"),
        help="Path to Casia-B.zip archive"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Base directory for saving checkpoints and results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs for Stage 1 and Stage 2 pretraining"
    )
    parser.add_argument(
        "--fs_epochs",
        type=int,
        default=10,
        help="Number of epochs for Few-Shot covariate fine-tuning"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and inference"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pre-trained weights for evaluation mode"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force execution on CPU even if NVIDIA GPU is available"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Hardware & Environment setup
    configure_windows_environment(force_cpu=args.cpu)

    # 2. Paths
    data_dir = Path(args.data_dir)
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"

    # 3. Resolve / Auto-extract dataset
    resolved_data_dir = ensure_dataset_extracted(zip_path, data_dir)
    print(f"[*] Active Dataset Directory: {resolved_data_dir}")

    model = None

    # 4. Train
    if args.mode in ["all", "train"]:
        model = run_training_pipeline(
            dataset_dir=resolved_data_dir,
            checkpoints_dir=checkpoints_dir,
            epochs_stage1=args.epochs,
            epochs_stage2=args.epochs,
            epochs_fs=args.fs_epochs,
            batch_size=args.batch_size
        )

    # 5. Evaluate
    if args.mode in ["all", "eval"]:
        if model is None:
            # Build model for evaluation
            num_test_subjects = 50  # Subjects 075 to 124
            model = build_crossview_gait_model(num_classes=num_test_subjects)
            weights_file = args.weights
            if weights_file is None:
                default_weights = checkpoints_dir / "final_fewshot_model.weights.h5"
                if default_weights.exists():
                    weights_file = default_weights
            
            if weights_file and Path(weights_file).exists():
                print(f"[+] Loading weights from: {weights_file}")
                model.load_weights(str(weights_file))
            else:
                print("[-] Warning: No pre-trained weights found. Running evaluation with initialized weights.")

        run_evaluation_pipeline(
            model=model,
            dataset_dir=resolved_data_dir,
            results_dir=results_dir
        )

    print("\n" + "=" * 65)
    print("Execution Finished! Check the 'outputs' directory for results.")
    print("=" * 65)


if __name__ == "__main__":
    main()
