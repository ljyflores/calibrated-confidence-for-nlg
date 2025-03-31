import argparse
import json
import pandas as pd

from analysis.analyze import datasets  # type: ignore
from analysis.utils_analysis import plot_correlation, prepare_scores, turn_log_prob_dict_into_np_array
from analysis.utils_tail_probs import compute_tail_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--temperature", type=str, required=True)
    args = parser.parse_args()

    metric = datasets[args.dataset]["metric"]  # type: ignore
    test_filepath = datasets[args.dataset]["path_test"]  # type: ignore
    scores_filepath = datasets[args.dataset]["scores_test"][args.model]  # type: ignore
    temperature = float(args.temperature)

    targets = [str(s) for s in pd.read_csv(test_filepath)["target"].fillna("")]  # type: ignore
    results = prepare_scores(scores_filepath, scores_filepath, targets, targets, metric, temperature)  # type: ignore

    scores = json.load(open(str(scores_filepath), "r")) # type: ignore
    beam_score_log_probs: dict[str, list[float]] = (
        scores.pop("beam_score_log_probs") if "beam_score_log_probs" in scores else {}
    )
    log_probs_by_sample = turn_log_prob_dict_into_np_array(beam_score_log_probs)

    tail_indices_by_k = {
        str(k): compute_tail_index(log_probs_by_sample[:, :k], temperature) 
        for k in range(log_probs_by_sample.shape[1])
        }
    plot_correlation(
        score_dictionary=tail_indices_by_k,
        metrics=[float(x) for x in results.scores_dataframe[metric]],  # type: ignore
        corr_type="spearman",
        title=str(args.dataset).title(),
        save_name=f"/home/mila/f/floresl/beam-search/analysis/images/plots_by_num_beams/{args.dataset}_{args.model}.jpg",
    )
