from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

try:
    from scripts.data import DatasetRecord
    from scripts.metrics import aggregate_metric_sets, summary_metrics
    from scripts.tis import TISResult
except ImportError:
    try:
        from .data import DatasetRecord
        from .metrics import aggregate_metric_sets, summary_metrics
        from .tis import TISResult
    except (ImportError, ValueError):
        from data import DatasetRecord
        from metrics import aggregate_metric_sets, summary_metrics
        from tis import TISResult




@dataclass(frozen=True)
class BinThresholds:
    low_upper: float = 0.2
    medium_upper: float = 0.5


def tis_bin_label(tis_value: float, thresholds: BinThresholds = BinThresholds()) -> str:
    if tis_value < thresholds.low_upper:
        return "low"
    if tis_value <= thresholds.medium_upper:
        return "medium"
    return "high"


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denom_x = sqrt(sum(a * a for a in dx))
    denom_y = sqrt(sum(b * b for b in dy))
    denominator = denom_x * denom_y
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def evaluate_by_tis_bins(
    records: Sequence[DatasetRecord],
    tis_results: Sequence[TISResult],
    bart_predictions: Sequence[str],
    led_predictions: Sequence[str],
    thresholds: BinThresholds = BinThresholds(),
) -> dict:
    if not (len(records) == len(tis_results) == len(bart_predictions) == len(led_predictions)):
        raise ValueError("records, tis_results, bart_predictions, and led_predictions must have the same length")

    grouped = {
        "low": {"bart": [], "led": [], "gaps": [], "tis": []},
        "medium": {"bart": [], "led": [], "gaps": [], "tis": []},
        "high": {"bart": [], "led": [], "gaps": [], "tis": []},
    }

    tis_values: list[float] = []
    rouge2_gaps: list[float] = []

    for record, tis_result, bart_text, led_text in zip(records, tis_results, bart_predictions, led_predictions):
        bart_metrics = summary_metrics(bart_text, record.abstract)
        led_metrics = summary_metrics(led_text, record.abstract)
        gap = led_metrics.rouge_2 - bart_metrics.rouge_2
        label = tis_bin_label(tis_result.tis, thresholds)

        grouped[label]["bart"].append(bart_metrics)
        grouped[label]["led"].append(led_metrics)
        grouped[label]["gaps"].append(gap)
        grouped[label]["tis"].append(tis_result.tis)

        tis_values.append(tis_result.tis)
        rouge2_gaps.append(gap)

    per_bin = {}
    for label, payload in grouped.items():
        counts = len(payload["gaps"])
        mean_gap = (sum(payload["gaps"]) / counts) if counts else 0.0
        mean_tis = (sum(payload["tis"]) / counts) if counts else 0.0
        per_bin[label] = {
            "count": counts,
            "mean_tis": mean_tis,
            "mean_rouge2_gap_led_minus_bart": mean_gap,
            "bart": aggregate_metric_sets(payload["bart"]),
            "led": aggregate_metric_sets(payload["led"]),
        }

    overall_bart = aggregate_metric_sets([summary_metrics(pred, record.abstract) for pred, record in zip(bart_predictions, records)])
    overall_led = aggregate_metric_sets([summary_metrics(pred, record.abstract) for pred, record in zip(led_predictions, records)])

    return {
        "count": len(records),
        "thresholds": {
            "low_upper": thresholds.low_upper,
            "medium_upper": thresholds.medium_upper,
        },
        "overall": {
            "bart": overall_bart,
            "led": overall_led,
            "mean_rouge2_gap_led_minus_bart": (sum(rouge2_gaps) / len(rouge2_gaps)) if rouge2_gaps else 0.0,
            "pearson_tis_vs_rouge2_gap": pearson_correlation(tis_values, rouge2_gaps),
        },
        "bins": per_bin,
    }
