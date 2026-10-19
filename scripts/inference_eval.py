import argparse
import json
import sys
import logging
import dataclasses
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import tqdm

# Ensure scripts directory is on PYTHONPATH if run as a script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Internal imports with fallback for standalone script execution
try:
    # Try absolute import as part of the 'scripts' package
    from scripts.data import DatasetRecord, load_pubmed_split, sample_records
    from scripts.models import BART_SPEC, LED_SPEC, HuggingFaceSummarizer
    from scripts.tis import build_corpus_statistics, compute_tis_batch
    from scripts.analysis import BinThresholds, evaluate_by_tis_bins
except ImportError:
    try:
        # Try relative import
        from .data import DatasetRecord, load_pubmed_split, sample_records
        from .models import BART_SPEC, LED_SPEC, HuggingFaceSummarizer
        from .tis import build_corpus_statistics, compute_tis_batch
        from .analysis import BinThresholds, evaluate_by_tis_bins
    except (ImportError, ValueError):
        # Fallback to direct import (when scripts/ is in sys.path)
        from data import DatasetRecord, load_pubmed_split, sample_records
        from models import BART_SPEC, LED_SPEC, HuggingFaceSummarizer
        from tis import build_corpus_statistics, compute_tis_batch
        from analysis import BinThresholds, evaluate_by_tis_bins



# JSON encoder for dataclasses
class DataclassEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)

def setup_logging(output_path: Path) -> logging.Logger:
    log_file = output_path.parent / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("tis_eval")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logging is called twice
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

def get_checkpoint_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.checkpoint.json"

def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_path: Path, data: Dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=DataclassEncoder)

def parse_args():
    parser = argparse.ArgumentParser(description="Run robust TIS-stratified evaluation with logging and checkpoints.")
    parser.add_argument("--dataset-root", type=Path, default=SCRIPT_DIR.parent / "PubMed_DATASET")
    parser.add_argument("--bart-path", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--led-path", type=str, default="allenai/led-large-16384")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size-bart", type=int, default=4)
    parser.add_argument("--batch-size-led", type=int, default=2)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "artifacts" / "reports" / "full_pipeline_report.json")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging(args.output)
    checkpoint_path = get_checkpoint_path(args.output)
    
    logger.info(f"{'='*70}")
    logger.info(f"TIS Stratified Evaluation Pipeline")
    logger.info(f"{'='*70}")
    
    checkpoint = None if args.no_checkpoint else load_checkpoint(checkpoint_path)
    
    if checkpoint:
        logger.info(f"✓ Resuming from checkpoint: {checkpoint_path}")
        phase = checkpoint.get("phase", "unknown")
    else:
        logger.info("Starting fresh evaluation run.")
        phase = "load_data"

    # Phase 1: Load Data
    if phase == "load_data":
        logger.info(f"Phase 1: Loading {args.split} split (limit={args.limit})...")
        records = load_pubmed_split(args.dataset_root, args.split, limit=args.limit)
        logger.info(f"  ✓ Loaded {len(records)} records.")
        phase = "compute_tis"
    else:
        # Checkpoint is guaranteed to be non-None if phase != "load_data"
        assert checkpoint is not None
        records = [DatasetRecord(**r) for r in checkpoint["records"]]

        logger.info(f"Phase 1: Loaded {len(records)} records from checkpoint.")

    # Phase 2: Compute TIS
    if phase == "compute_tis":
        logger.info("Phase 2: Computing TIS values...")
        corpus = build_corpus_statistics(r.article for r in records)
        tis_results = compute_tis_batch(records, cutoff_tokens=BART_SPEC.max_source_length, corpus_statistics=corpus)
        logger.info("  ✓ TIS computation complete.")
        
        if not args.no_checkpoint:
            save_checkpoint(checkpoint_path, {
                "phase": "bart_inference",
                "records": [r.to_dict() for r in records],
                "tis_results": tis_results
            })
        phase = "bart_inference"
    else:
        assert checkpoint is not None
        tis_results = checkpoint.get("tis_results", [])

        logger.info("Phase 2: TIS values loaded from checkpoint.")

    # Phase 3: BART Inference
    if phase == "bart_inference":
        logger.info(f"Phase 3: Running BART inference (model={args.bart_path})...")
        summarizer = HuggingFaceSummarizer(args.bart_path, max_source_length=BART_SPEC.max_source_length)
        with tqdm.tqdm(total=len(records), desc="BART Inference") as pbar:
            bart_preds = summarizer.generate_batch([r.article for r in records], batch_size=args.batch_size_bart, progress_bar=pbar)
        logger.info("  ✓ BART inference complete.")
        
        if not args.no_checkpoint:
            save_checkpoint(checkpoint_path, {
                "phase": "led_inference",
                "records": [r.to_dict() for r in records],
                "tis_results": tis_results,
                "bart_preds": bart_preds
            })
        phase = "led_inference"
    else:
        assert checkpoint is not None
        bart_preds = checkpoint.get("bart_preds", [])

        logger.info("Phase 3: BART predictions loaded from checkpoint.")

    # Phase 4: LED Inference
    if phase == "led_inference":
        logger.info(f"Phase 4: Running LED inference (model={args.led_path})...")
        summarizer = HuggingFaceSummarizer(args.led_path, max_source_length=LED_SPEC.max_source_length)
        with tqdm.tqdm(total=len(records), desc="LED Inference") as pbar:
            led_preds = summarizer.generate_batch([r.article for r in records], batch_size=args.batch_size_led, progress_bar=pbar)
        logger.info("  ✓ LED inference complete.")
        
        if not args.no_checkpoint:
            save_checkpoint(checkpoint_path, {
                "phase": "evaluation",
                "records": [r.to_dict() for r in records],
                "tis_results": tis_results,
                "bart_preds": bart_preds,
                "led_preds": led_preds
            })
        phase = "evaluation"
    else:
        assert checkpoint is not None
        led_preds = checkpoint.get("led_preds", [])

        logger.info("Phase 4: LED predictions loaded from checkpoint.")

    # Phase 5: Evaluation
    logger.info("Phase 5: Computing stratified metrics...")
    report = evaluate_by_tis_bins(
        records=records,
        tis_results=tis_results,
        bart_predictions=bart_preds,
        led_predictions=led_preds,
        thresholds=BinThresholds(low_upper=0.2, medium_upper=0.5)
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    if not args.no_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("✓ Evaluation complete. Checkpoint removed.")
    
    logger.info(f"Report saved to {args.output}")
    logger.info(f"Overall Pearson Correlation: {report['overall']['pearson_tis_vs_rouge2_gap']:.4f}")
    logger.info(f"{'='*70}")

if __name__ == "__main__":
    main()
