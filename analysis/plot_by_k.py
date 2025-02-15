import argparse
import pandas as pd

from analysis.analyze import datasets  # type: ignore
from analysis.utils_analysis import plot_correlation, prepare_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    metric = datasets[args.dataset]["metric"]  # type: ignore
    test_filepath = datasets[args.dataset]["path_test"]  # type: ignore
    scores_filepath = datasets[args.dataset]["scores_test"][args.model]  # type: ignore
    temperature = 1.0

    targets = [str(s) for s in pd.read_csv(test_filepath)["target"].fillna("")]  # type: ignore
    results = prepare_scores(scores_filepath, scores_filepath, targets, targets, metric, temperature)  # type: ignore
    plot_correlation(
        score_dictionary=results.scores_by_beam["beam_score_ratios"],
        metrics=[float(x) for x in results.scores_dataframe[metric]],  # type: ignore
        corr_type="spearman",
        title=str(args.dataset).title(),
        save_name=f"/home/mila/f/floresl/beam-search/analysis/images/{args.dataset}_{args.model}.jpg",
    )
