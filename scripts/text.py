from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def simple_tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def token_count(text: str) -> int:
    return len(simple_tokenize(text))


def simple_sentence_split(text: str) -> list[str]:
    cleaned = normalize_whitespace(text.strip())
    if not cleaned:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    return parts if parts else [cleaned]


def sentence_token_counts(sentences: Sequence[str]) -> list[int]:
    return [token_count(sentence) for sentence in sentences]


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def build_document_frequencies(texts: Iterable[str]) -> tuple[dict[str, int], int]:
    doc_freq: defaultdict[str, int] = defaultdict(int)
    corpus_size = 0
    for text in texts:
        corpus_size += 1
        unique_tokens = set(simple_tokenize(text))
        for token in unique_tokens:
            doc_freq[token] += 1
    return dict(doc_freq), corpus_size


def idf(token: str, document_frequencies: dict[str, int], corpus_size: int) -> float:
    return math.log((1 + corpus_size) / (1 + document_frequencies.get(token, 0))) + 1.0


def sentence_tfidf_score(
    sentence: str,
    document_frequencies: dict[str, int] | None = None,
    corpus_size: int | None = None,
) -> float:
    tokens = simple_tokenize(sentence)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    if document_frequencies is None or corpus_size is None:
        # Local fallback: use the sentence itself as the document universe.
        document_frequencies = {token: 1 for token in counts}
        corpus_size = 1
    score = 0.0
    for token, count in counts.items():
        score += count * idf(token, document_frequencies, corpus_size)
    return score


def unigram_overlap_recall(sentence_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    if not sentence_tokens or not reference_tokens:
        return 0.0
    sentence_counts = Counter(sentence_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(min(sentence_counts[token], reference_counts[token]) for token in reference_counts)
    return overlap / sum(reference_counts.values())


def sentence_starts(sentences: Sequence[str]) -> list[int]:
    starts: list[int] = []
    running = 0
    for sentence in sentences:
        starts.append(running)
        running += token_count(sentence)
    return starts


def lead_summary(article: str, max_tokens: int = 256, max_sentences: int = 3) -> str:
    sentences = simple_sentence_split(article)
    if not sentences:
        return ""
    selected: list[str] = []
    used_tokens = 0
    for sentence in sentences[:max_sentences]:
        tokens = simple_tokenize(sentence)
        if not tokens:
            continue
        if selected and used_tokens + len(tokens) > max_tokens:
            break
        selected.append(sentence.strip())
        used_tokens += len(tokens)
        if used_tokens >= max_tokens:
            break
    if selected:
        return " ".join(selected)
    tokens = simple_tokenize(article)[:max_tokens]
    return " ".join(tokens)


@dataclass(frozen=True)
class SentenceRanking:
    score: float
    position: int
    sentence: str


def rank_sentences(
    article: str,
    reference: str | None = None,
    document_frequencies: dict[str, int] | None = None,
    corpus_size: int | None = None,
) -> list[SentenceRanking]:
    sentences = simple_sentence_split(article)
    reference_tokens = simple_tokenize(reference or "")
    ranked: list[SentenceRanking] = []
    for position, sentence in enumerate(sentences):
        tokens = simple_tokenize(sentence)
        if not tokens:
            continue
        tfidf = sentence_tfidf_score(sentence, document_frequencies, corpus_size)
        overlap = unigram_overlap_recall(tokens, reference_tokens)
        score = tfidf * (1.0 + overlap)
        ranked.append(SentenceRanking(score=score, position=position, sentence=sentence))
    ranked.sort(key=lambda item: (-item.score, item.position))
    return ranked


def oracle_summary(
    article: str,
    reference: str,
    max_tokens: int = 256,
    document_frequencies: dict[str, int] | None = None,
    corpus_size: int | None = None,
) -> str:
    ranked = rank_sentences(article, reference, document_frequencies, corpus_size)
    if not ranked:
        return lead_summary(article, max_tokens=max_tokens)
    selected_positions: set[int] = set()
    used_tokens = 0
    sentences = simple_sentence_split(article)
    sentence_lengths = sentence_token_counts(sentences)
    for item in ranked:
        sentence_length = sentence_lengths[item.position]
        if used_tokens + sentence_length > max_tokens:
            continue
        selected_positions.add(item.position)
        used_tokens += sentence_length
        if used_tokens >= max_tokens:
            break
    ordered = [sentence for index, sentence in enumerate(sentences) if index in selected_positions]
    if ordered:
        return " ".join(ordered)
    return lead_summary(article, max_tokens=max_tokens)
