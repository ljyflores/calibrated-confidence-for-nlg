import argparse
import pandas as pd
import random

from scipy.stats import spearmanr  # type: ignore

from analysis.utils_analysis import prepare_scores
from analysis.analyze import datasets  # type: ignore


def test_correlations(
    quality_scores: list[float],
    confidence_scores_ours: list[float],
    confidence_scores_theirs: list[float],
):
    assert (
        len(quality_scores)
        == len(confidence_scores_ours)
        == len(confidence_scores_theirs)
    )
    num_iters = 10000

    # First, calculate the observed improvement from the current data
    current_corr_ours = float(spearmanr(quality_scores, confidence_scores_ours).statistic)  # type: ignore
    current_corr_theirs = float(spearmanr(quality_scores, confidence_scores_theirs).statistic)  # type: ignore
    current_improvement = abs(current_corr_ours) - abs(current_corr_theirs)

    # Next, bootstrap, and count the number of times that we observe an improvement more rare than ours
    num_sampled_better_than_current = 0
    for _ in range(num_iters):
        # Bootstrap
        selected_indices = random.choices(list(range(len(quality_scores))), k=1000)
        selected_quality_scores = [quality_scores[idx] for idx in selected_indices]
        selected_conf_scores_ours = [
            confidence_scores_ours[idx] for idx in selected_indices
        ]
        selected_conf_scores_theirs = [
            confidence_scores_theirs[idx] for idx in selected_indices
        ]
        # Compute correlations and improvement
        sampled_corr_ours = float(spearmanr(selected_quality_scores, selected_conf_scores_ours).statistic)  # type: ignore
        sampled_corr_theirs = float(spearmanr(selected_quality_scores, selected_conf_scores_theirs).statistic)  # type: ignore
        sampled_improvement = abs(sampled_corr_ours) - abs(sampled_corr_theirs)
        # Check if the sampled improvement is rarer than what we've observed
        if sampled_improvement > 2 * current_improvement:
            num_sampled_better_than_current += 1
    return num_sampled_better_than_current / num_iters


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.00)
    args = parser.parse_args()

    metric = datasets[args.dataset]["metric"]  # type: ignore
    test_filepath = datasets[args.dataset]["path"]  # type: ignore
    scores_filepath = datasets[args.dataset]["scores"][args.model]  # type: ignore
    temperature = args.temperature

    targets = [str(s) for s in pd.read_csv(test_filepath)["target"].fillna("")]  # type: ignore
    results = prepare_scores(scores_filepath, targets, metric, temperature)  # type: ignore

    confidence_score_df = results.scores_dataframe
    col_ratio = str(
        list(
            filter(
                lambda s: s.startswith("beam_score_ratios"), confidence_score_df.columns
            )
        )[0]
    )
    col_tail = "tail_index"

    confidence_scores_ratio = [float(x) for x in confidence_score_df[col_ratio]]  # type: ignore
    confidence_scores_slope = [float(x) for x in confidence_score_df[col_tail]]  # type: ignore
    confidence_scores_baseline = [float(x) for x in confidence_score_df[args.baseline]]  # type: ignore
    quality_scores = [float(x) for x in confidence_score_df[metric]]  # type: ignore

    p_val_ratio = test_correlations(
        quality_scores, confidence_scores_ratio, confidence_scores_baseline
    )
    p_val_slope = test_correlations(
        quality_scores, confidence_scores_slope, confidence_scores_baseline
    )
    print(f"Ratio: {p_val_ratio}")
    print(f"Slope: {p_val_slope}")
