from typing import Any
import numpy as np
from src.eval import calculate_rouge_single, calculate_bleu
from scipy.stats import spearmanr  # type: ignore


def get_list_of_scores(target: str, predictions: list[str], metric: str):
    if metric == "rougeL":
        return list(
            map(
                lambda pred: calculate_rouge_single(pred, target)["rougeL"],
                predictions,
            )
        )
    if metric == "bleu":
        return list(
            map(
                lambda pred: calculate_bleu([pred], [[target]]),
                predictions,
            )
        )


def get_oracle_scores(
    targets: list[str],
    output_sequences_by_sample: list[list[str]],
    log_probs_by_sample: np.ndarray[Any, np.dtype[np.float64]],
    metric: str,
):
    weighted_avgs = list[float]()
    corr_w_scores = list[float]()
    for i in range(len(targets)):
        quality_scores = get_list_of_scores(
            targets[i], output_sequences_by_sample[i], metric
        )
        assert quality_scores is not None
        weighted_avg = (
            float(np.average(log_probs_by_sample[i], weights=quality_scores))  # type: ignore
            if sum(quality_scores) > 0
            else 0
        )
        corr_w_score = float(
            spearmanr(log_probs_by_sample[i], quality_scores).statistic  # type: ignore
        )
        if str(corr_w_score) == "nan":
            corr_w_score = 0.0
        weighted_avgs.append(weighted_avg)
        corr_w_scores.append(corr_w_score)
    return weighted_avgs, corr_w_scores
