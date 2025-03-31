import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("/home/mila/f/floresl/beam-search")

from analysis.utils_tail_probs import compute_tail_index, compute_entropy_by_sample
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
    json_path_test: str,
    json_path_val: str,
    targets_test: list[str],
    targets_val: list[str],
    metric: Literal["rougeL", "f1", "bleu"],
    temperature: float,
):
    # Determine k to use in ratio method using validation set
    scores_val = json.load(open(json_path_val, "r"))
    beam_score_ratios_val = scores_val.pop("beam_score_ratios")
    sentences_val = scores_val.pop("sentences")
    top_sentences_val = [
        str(lst[0]) if isinstance(lst[0], str) else "" for lst in sentences_val
    ]
    metrics_val = compute_metric_by_sample(top_sentences_val, targets_val, metric)
    _, best_k = find_k_with_best_correlation(metrics_val, beam_score_ratios_val)
    print(f"Best k: {best_k}")

    # Unpack metrics from JSON of test set
    scores = json.load(open(json_path_test, "r"))
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
    log_probs_by_sample = turn_log_prob_dict_into_np_array(beam_score_log_probs)

    _ = (  # type: ignore
        scores.pop("sequence_joint_log_probs")
        if "sequence_joint_log_probs" in scores
        else {}
    )
    sentences = scores.pop("sentences")
    top_sentences = [
        str(lst[0]) if isinstance(lst[0], str) else "" for lst in sentences
    ]

    if "mean_token_log_probs" in scores:
        scores.pop("mean_token_log_probs")

    df_score = pd.DataFrame.from_dict(scores)  # type: ignore
    df_score["sentences"] = top_sentences
    df_score["tail_index"] = compute_tail_index(log_probs_by_sample, temperature)
    df_score["beam_score_entropy"] = compute_entropy_by_sample(log_probs_by_sample, temperature)
    metrics = compute_metric_by_sample(top_sentences, targets_test, metric)
    df_score[metric] = metrics

    df_score[f"beam_score_ratios"] = beam_score_ratios[str(best_k)]
    # df_score[f"beam_score_log_probs_{best_k}"] = beam_score_log_probs[str(best_k)]
    df_score[f"beam_score_sum_top_{best_k}"] = beam_score_sum_top_k[str(best_k)]
    df_score[f"beam_score_impt_wt"] = beam_score_impt_weighted[str(best_k)]
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
    title: str | None = None,
    save_name: str | None = None,
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
        "Pearson Correlation" if corr_type == "pearson" else "Spearman Correlation",
        fontsize=16,
    )
    plt.xlabel("k", fontsize=16)  # type: ignore
    plt.xticks(fontsize=16)  # type: ignore
    plt.yticks(fontsize=16)  # type: ignore
    if title:
        plt.title(title)  # type: ignore
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")  # type: ignore


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


def turn_log_prob_dict_into_np_array(
    dict_of_log_probs_by_beam_idx: dict[str, list[float]]
):
    log_probs_by_idx = np.array(
        [
            dict_of_log_probs_by_beam_idx[str(idx)]
            for idx in range(len(dict_of_log_probs_by_beam_idx.keys()))
        ]
    )
    log_probs_by_sample = log_probs_by_idx.T
    return log_probs_by_sample
