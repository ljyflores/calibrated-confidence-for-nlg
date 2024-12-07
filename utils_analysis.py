import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Literal
from src.eval import calculate_rouge_single, calculate_f1_score, calculate_bleu
from scipy.stats import spearmanr  # type: ignore


@dataclass
class ConfidenceOutput:
    scores_dataframe: pd.DataFrame
    scores_by_beam: dict[str, dict[str, list[float]]]
    sentences: list[list[str]]


def argmax(iterable: list[float]):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def compute_metric_by_sample(
    predictions: list[str], targets: list[str], metric: Literal["rougeL", "f1", "bleu"]
):
    metrics = list[float]()
    for pred, label in zip(predictions, targets):
        if metric == "rougeL":
            score: float = calculate_rouge_single(prediction=pred, ground_truth=label)[
                "rougeL"
            ]
        elif metric == "bleu":
            score: float = calculate_bleu(predictions=[pred], references=[[label]])
        else:
            score = calculate_f1_score(prediction=pred, ground_truth=label)
        metrics.append(score)
    return metrics


def find_k_with_best_correlation(
    ground_truth_score: list[float], confidence_score_dict: dict[str, list[float]]
):
    correlations: list[float] = [
        abs(spearmanr(ground_truth_score, confidence_score_dict[str(k)]).statistic)  # type: ignore
        for k in range(len(confidence_score_dict.keys()))
    ]
    correlations = list(map(lambda x: 0 if math.isnan(x) else x, correlations))
    best_k = argmax(correlations)
    return correlations, best_k


def prepare_scores(
    json_path: str, targets: list[str], metric: Literal["rougeL", "f1", "bleu"]
):
    scores = json.load(open(json_path, "r"))
    beam_score_ratios: dict[str, list[float]] = (
        scores.pop("beam_score_ratios") if "beam_score_ratios" in scores else {}
    )
    beam_score_log_probs: dict[str, list[float]] = (
        scores.pop("beam_score_log_probs") if "beam_score_log_probs" in scores else {}
    )
    beam_score_sum_top_k: dict[str, list[float]] = (
        scores.pop("beam_score_sum_top_k") if "beam_score_sum_top_k" in scores else {}
    )
    beam_score_impt_weighted: dict[str, list[float]] = (
        scores.pop("importance_weighted_log_probs")
        if "importance_weighted_log_probs" in scores
        else {}
    )
    _ = (
        scores.pop("sequence_joint_log_probs")
        if "sequence_joint_log_probs" in scores
        else {}
    )
    sentences: list[list[str]] = scores.pop("sentences")
    top_sentences = [lst[0] for lst in sentences]

    if "mean_token_log_probs" in scores:
        scores.pop("mean_token_log_probs")

    df_score = pd.DataFrame.from_dict(scores)  # type: ignore
    df_score["sentences"] = top_sentences
    metrics = compute_metric_by_sample(top_sentences, targets, metric)
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
    return ConfidenceOutput(
        scores_dataframe=df_score,
        scores_by_beam={
            "beam_score_ratios": beam_score_ratios,
            "beam_score_log_probs": beam_score_log_probs,
            "beam_score_sum_top_k": beam_score_sum_top_k,
            "beam_score_importance_wt": beam_score_impt_weighted,
        },
        sentences=sentences,
    )


def plot_correlation(
    score_dictionary: dict[str, list[float]],
    metrics: list[float],
    corr_type: Literal["spearman", "pearson"],
):
    num_k = len(score_dictionary.keys())
    if corr_type == "pearson":
        correlations = [
            float(np.corrcoef(score_dictionary[str(k)], metrics)[0][1])
            for k in range(num_k)
        ]
    elif corr_type == "spearman":
        correlations = [
            float(spearmanr(score_dictionary[str(k)], metrics).statistic)  # type: ignore
            for k in range(num_k)
        ]
    plt.scatter(  # type: ignore
        x=list(range(num_k)),
        y=correlations,
    )
    plt.ylabel(  # type: ignore
        "Pearson Correlation" if corr_type == "pearson" else "Spearman Correlation"
    )
    plt.xlabel("k")  # type: ignore


def plot_sequence_probs(
    probs_1_: list[float],
    probs_2_: list[float],
    seq_len: int = 1,
):
    _, ax = plt.subplots(1, 2)  # type: ignore

    probs_1 = np.array(probs_1_)
    probs_2 = np.array(probs_2_)  # type: ignore

    ax[0].bar(
        x=list(range(len(probs_1))),  # type: ignore
        height=np.exp(seq_len * probs_1),  # type: ignore
    )
    ax[0].set_xlabel("Beam Number")
    ax[0].set_ylabel("Joint Sequence Probability")

    ax[1].bar(
        x=list(range(len(probs_2))),  # type: ignore
        height=np.exp(seq_len * probs_2),  # type: ignore
    )
    ax[1].set_xlabel("Beam Number")
    ax[1].set_ylabel("Joint Sequence Probability")
