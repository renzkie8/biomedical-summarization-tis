from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from scripts.text import simple_tokenize, token_count
except ImportError:
    try:
        from .text import simple_tokenize, token_count
    except (ImportError, ValueError):
        from text import simple_tokenize, token_count



DEFAULT_ARTICLE_COLUMN = "article"
DEFAULT_ABSTRACT_COLUMN = "abstract"

try:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
except (OverflowError, ValueError):
    try:
        csv.field_size_limit(2**31 - 1)
    except (OverflowError, ValueError):
        pass


@dataclass(frozen=True)
class DatasetRecord:
    article: str
    abstract: str
    row_index: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def resolve_split_path(dataset_root: str | Path, split: str) -> Path:
    root = Path(dataset_root)
    candidates = [
        root / f"{split}.csv" / f"{split}.csv",
        root / f"{split}.csv",
        root / split / f"{split}.csv",
        root / split,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {split!r} split under {root}")


def iter_pubmed_split(
    dataset_root: str | Path,
    split: str = "train",
    *,
    limit: int | None = None,
    skip: int = 0,
    article_column: str = DEFAULT_ARTICLE_COLUMN,
    abstract_column: str = DEFAULT_ABSTRACT_COLUMN,
) -> Iterator[DatasetRecord]:
    path = resolve_split_path(dataset_root, split)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a valid CSV header")
        if article_column not in reader.fieldnames or abstract_column not in reader.fieldnames:
            raise KeyError(
                f"Expected columns {article_column!r} and {abstract_column!r} in {path.name}, "
                f"found {reader.fieldnames}"
            )
        for index, row in enumerate(reader):
            if index < skip:
                continue
            article = (row.get(article_column) or "").strip()
            abstract = (row.get(abstract_column) or "").strip()
            if not article or not abstract:
                continue
            yield DatasetRecord(article=article, abstract=abstract, row_index=index)
            if limit is not None:
                limit -= 1
                if limit <= 0:
                    break


def load_pubmed_split(
    dataset_root: str | Path,
    split: str = "train",
    *,
    limit: int | None = None,
    skip: int = 0,
    article_column: str = DEFAULT_ARTICLE_COLUMN,
    abstract_column: str = DEFAULT_ABSTRACT_COLUMN,
) -> list[DatasetRecord]:
    return list(
        iter_pubmed_split(
            dataset_root,
            split,
            limit=limit,
            skip=skip,
            article_column=article_column,
            abstract_column=abstract_column,
        )
    )


def sample_records(records: Sequence[DatasetRecord], size: int, seed: int = 42) -> list[DatasetRecord]:
    if size >= len(records):
        return list(records)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), size))
    return [records[index] for index in indices]


def write_records(records: Sequence[DatasetRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[DEFAULT_ARTICLE_COLUMN, DEFAULT_ABSTRACT_COLUMN, "row_index"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
    return output


def dataset_statistics(records: Sequence[DatasetRecord]) -> dict[str, float | int]:
    article_token_lengths = [token_count(record.article) for record in records]
    abstract_token_lengths = [token_count(record.abstract) for record in records]
    total = len(records)
    return {
        "records": total,
        "avg_article_tokens": sum(article_token_lengths) / total if total else 0.0,
        "avg_abstract_tokens": sum(abstract_token_lengths) / total if total else 0.0,
        "min_article_tokens": min(article_token_lengths) if article_token_lengths else 0,
        "max_article_tokens": max(article_token_lengths) if article_token_lengths else 0,
    }


def take_texts(records: Iterable[DatasetRecord], field: str = "article") -> list[str]:
    if field not in {"article", "abstract"}:
        raise ValueError("field must be either 'article' or 'abstract'")
    values: list[str] = []
    for record in records:
        values.append(record.article if field == "article" else record.abstract)
    return values
