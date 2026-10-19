from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure scripts directory is on PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scripts.data import load_pubmed_split, sample_records
    from scripts.training import run_training_pipeline
    from scripts.models import LED_SPEC
except ImportError:
    from data import load_pubmed_split, sample_records
    from training import run_training_pipeline
    from models import LED_SPEC

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU LED Training")
    parser.add_argument("--dataset-root", type=Path, default=SCRIPT_DIR.parent / "PubMed_DATASET")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR.parent / "fine_tuned_models")
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading data for multi-GPU LED training ({args.train_size} samples)...")
    
    # Load and sample data
    full_train = load_pubmed_split(args.dataset_root, "train", limit=args.train_size + 1000)
    full_val = load_pubmed_split(args.dataset_root, "validation", limit=500)
    
    train_records = sample_records(full_train, args.train_size, seed=42)
    val_records = sample_records(full_val, 500, seed=42)

    # Run the pipeline with the 4096 speed-up
    run_training_pipeline(
        model_name=LED_SPEC.name,
        train_records=train_records,
        val_records=val_records,
        output_dir=args.output_root / "led_large",
        epochs=args.epochs,
        is_led=True,
        max_source_length=4096,
        resume_from_checkpoint=False # Start fresh for clean multi-GPU state
    )

if __name__ == "__main__":
    main()
