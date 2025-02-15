import random

from scipy.stats import spearmanr  # type: ignore


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
    print(current_corr_ours, current_corr_theirs)
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
