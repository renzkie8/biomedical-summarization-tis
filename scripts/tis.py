from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

# Ensure scripts directory is on PYTHONPATH if run as a script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scripts.data import DatasetRecord, load_pubmed_split
    from scripts.models import BART_SPEC
    from scripts.text import (
        build_document_frequencies,
        sentence_starts,
        sentence_token_counts,
        sentence_tfidf_score,
        simple_sentence_split,
        simple_tokenize,
        token_count,
        unigram_overlap_recall,
    )
except ImportError:
    try:
        from .data import DatasetRecord, load_pubmed_split
        from .models import BART_SPEC
        from .text import (
            build_document_frequencies,
            sentence_starts,
            sentence_token_counts,
            sentence_tfidf_score,
            simple_sentence_split,
            simple_tokenize,
            token_count,
            unigram_overlap_recall,
        )
    except (ImportError, ValueError):
        from data import DatasetRecord, load_pubmed_split
        from models import BART_SPEC
        from text import (
            build_document_frequencies,
            sentence_starts,
            sentence_token_counts,
            sentence_tfidf_score,
            simple_sentence_split,
            simple_tokenize,
            token_count,
            unigram_overlap_recall,
        )




@dataclass(frozen=True)
class CorpusStatistics:
    document_frequencies: dict[str, int]
    corpus_size: int


@dataclass(frozen=True)
class TISResult:
    article_length_tokens: int
    cutoff_tokens: int
    truncated_token_ratio: float
    tis: float
    retained_salience: float
    lost_salience: float
    total_salience: float
    total_sentences: int
    truncated_sentences: int


def build_corpus_statistics(texts: Iterable[str]) -> CorpusStatistics:
    document_frequencies, corpus_size = build_document_frequencies(texts)
    return CorpusStatistics(document_frequencies=document_frequencies, corpus_size=corpus_size)


def compute_tis(
    article: str,
    reference: str,
    cutoff_tokens: int = 1024,
    corpus_statistics: CorpusStatistics | None = None,
) -> TISResult:
    sentences = simple_sentence_split(article)
    reference_tokens = simple_tokenize(reference)
    total_tokens = token_count(article)
    starts = sentence_starts(sentences)
    lengths = sentence_token_counts(sentences)

    if corpus_statistics is None:
        # Fallback: use article-local TF-IDF if corpus statistics are unavailable.
        doc_freq = None
        corpus_size = None
    else:
        doc_freq = corpus_statistics.document_frequencies
        corpus_size = corpus_statistics.corpus_size

    retained_salience = 0.0
    lost_salience = 0.0
    truncated_sentences = 0

    for sentence, start, length in zip(sentences, starts, lengths):
        sentence_tokens = simple_tokenize(sentence)
        if not sentence_tokens:
            continue
        importance = sentence_tfidf_score(sentence, doc_freq, corpus_size)
        overlap = unigram_overlap_recall(sentence_tokens, reference_tokens)
        salience = importance * overlap
        end = start + length

        retained_tokens = max(0, min(end, cutoff_tokens) - start)
        lost_tokens = max(0, end - max(start, cutoff_tokens))

        if retained_tokens:
            retained_salience += salience * (retained_tokens / length)
        if lost_tokens:
            lost_salience += salience * (lost_tokens / length)
            truncated_sentences += 1

    total_salience = retained_salience + lost_salience
    tis = (lost_salience / total_salience) if total_salience else 0.0
    truncated_token_ratio = max(0.0, (total_tokens - cutoff_tokens) / total_tokens) if total_tokens else 0.0

    return TISResult(
        article_length_tokens=total_tokens,
        cutoff_tokens=cutoff_tokens,
        truncated_token_ratio=truncated_token_ratio,
        tis=tis,
        retained_salience=retained_salience,
        lost_salience=lost_salience,
        total_salience=total_salience,
        total_sentences=len(sentences),
        truncated_sentences=truncated_sentences,
    )


def compute_tis_batch(
    records: Sequence[DatasetRecord],
    cutoff_tokens: int = 1024,
    corpus_statistics: CorpusStatistics | None = None,
) -> list[TISResult]:
    return [compute_tis(record.article, record.abstract, cutoff_tokens=cutoff_tokens, corpus_statistics=corpus_statistics) for record in records]


def summarize_tis(results: Sequence[TISResult]) -> dict[str, float]:
    values = [result.tis for result in results]
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Compute Truncation Impact Score (TIS) for PubMed dataset splits.")
    parser.add_argument("--dataset-root", type=Path, default=SCRIPT_DIR.parent / "PubMed_DATASET")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--cutoff", type=int, default=BART_SPEC.max_source_length, help="Context window cutoff (tokens)")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "artifacts" / "reports" / "tis_analysis.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading {args.split} split from {args.dataset_root}...")
    records = load_pubmed_split(args.dataset_root, args.split, limit=args.limit)
    print(f"Loaded {len(records)} records.")
    
    print("Building corpus statistics...")
    corpus_stats = build_corpus_statistics(r.article for r in records)
    
    print(f"Computing TIS (cutoff={args.cutoff})...")
    tis_results = compute_tis_batch(records, cutoff_tokens=args.cutoff, corpus_statistics=corpus_stats)
    
    summary = summarize_tis(tis_results)
    
    output_data = {
        "metadata": {
            "split": args.split,
            "cutoff": args.cutoff,
            "record_count": len(records)
        },
        "summary": summary,
        "results": [
            {
                "row_index": i,
                "tis": r.tis,
                "truncated_token_ratio": r.truncated_token_ratio
            } for i, r in enumerate(tis_results)
        ]
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"Mean TIS: {summary['mean']:.4f}")

if __name__ == "__main__":
    main()
