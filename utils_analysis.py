import json
import math
import pandas as pd

from typing import Literal
from src.eval import calculate_rouge, calculate_f1_score
from scipy.stats import spearmanr  # type: ignore


def argmax(iterable: list[float]):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def compute_metric_by_sample(
    predictions: list[str], targets: list[str], metric: Literal["rougeL", "f1"]
):
    metrics = list[float]()
    for pred, label in zip(predictions, targets):
        if metric == "rougeL":
            score: float = calculate_rouge(predictions=[pred], references=[[label]])[
                "rougeL"
            ]
        else:
            score = calculate_f1_score(prediction=pred, ground_truth=label)
        metrics.append(score)
    return metrics


def find_k_with_best_correlation(
    ground_truth_score: list[float], confidence_score_dict: dict[str, list[float]]
):
    correlations: list[float] = [
        abs(spearmanr(ground_truth_score, confidence_score_dict[str(k)]).statistic)
        for k in range(len(confidence_score_dict.keys()))
    ]
    correlations = list(map(lambda x: 0 if math.isnan(x) else x, correlations))
    best_k = argmax(correlations)
    return correlations, best_k


def prepare_scores(json_path: str, targets: list[str], metric: Literal["rougeL", "f1"]):
    scores = json.load(open(json_path, "r"))
    beam_score_ratios: dict[str, list[float]] = scores.pop("beam_score_ratios")
    beam_score_log_probs: dict[str, list[float]] = scores.pop("beam_score_log_probs")
    beam_score_sum_top_k: dict[str, list[float]] = scores.pop("beam_score_sum_top_k")
    beam_score_impt_weighted: dict[str, list[float]] = scores.pop(
        "importance_weighted_log_probs"
    )

    df_score = pd.DataFrame.from_dict(scores)
    metrics = compute_metric_by_sample(scores["sentences"], targets, metric)
    df_score[metric] = metrics

    _, best_k = find_k_with_best_correlation(metrics, beam_score_ratios)
    # _, bs_log_probs_k = find_k_with_best_correlation(metrics, beam_score_log_probs)
    # _, bs_sum_top_k = find_k_with_best_correlation(metrics, beam_score_sum_top_k)
    # _, bs_imp_wt_k = find_k_with_best_correlation(
    #     metrics, beam_score_importance_weighted
    # )

    df_score[f"beam_score_ratios_{best_k}"] = beam_score_ratios[str(best_k)]
    df_score[f"beam_score_log_probs_{best_k}"] = beam_score_log_probs[str(best_k)]
    df_score[f"beam_score_top_k_{best_k}"] = beam_score_sum_top_k[str(best_k)]
    df_score[f"beam_score_impt_wt_{best_k}"] = beam_score_impt_weighted[str(best_k)]
    return df_score, {
        "beam_score_ratios": beam_score_ratios,
        "beam_score_log_probs": beam_score_log_probs,
        "beam_score_sum_top_k": beam_score_sum_top_k,
        "beam_score_importance_wt": beam_score_impt_weighted,
    }
