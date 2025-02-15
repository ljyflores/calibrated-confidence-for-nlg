import argparse
import pandas as pd

from analysis.utils_analysis import prepare_scores
from analysis.utils_stat_test import test_correlations

datasets = {  # type: ignore
    "debatesum": {
        "path_test": "data/debatesum/test.csv",
        "path_val": "data/debatesum/val.csv",
        "metric": "rougeL",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/bart-base_data_debatesum_checkpoint-1500_debatesum.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_debatesum_checkpoint-1500_debatesum.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-debatesum-checkpoint-1500_debatesum.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-debatesum-checkpoint-1500_debatesum.json",
        },
    },
    "flores": {
        "path_test": "data/flores/test.csv",
        "path_val": "data/flores/val.csv",
        "metric": "bleu",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/facebook-bart-base_data-flores-checkpoint-260_flores.json",  # "/home/mila/f/floresl/beam-search/results/final/bart-base_data_flores_checkpoint-240_flores.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/google-flan-t5-base_data-flores-checkpoint-200_flores.json",  # "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_flores_checkpoint-260_flores.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-flores-checkpoint-260_flores.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-flores-checkpoint-200_flores.json",
        },
    },
    "hotpotqa": {
        "path_test": "data/hotpotqa/test.csv",
        "path_val": "data/hotpotqa/val.csv",
        "metric": "f1",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/bart-base_data_hotpotqa_checkpoint-26835_hotpotqa.json",  # "/home/mila/f/floresl/beam-search/results/final/facebook-bart-base_data-hotpotqa-checkpoint-180_hotpotqa.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_hotpotqa_checkpoint-26835_hotpotqa.json",  # "/home/mila/f/floresl/beam-search/results/final/google-flan-t5-base_data-hotpotqa-checkpoint-220_hotpotqa.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-hotpotqa-checkpoint-26835_hotpotqa.json",  # "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-hotpotqa-checkpoint-180_hotpotqa.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-hotpotqa-checkpoint-26835_hotpotqa.json",  # "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-hotpotqa-checkpoint-220_hotpotqa.json",
        },
    },
    "squad": {
        "path_test": "data/squad/test.csv",
        "path_val": "data/squad/val.csv",
        "metric": "f1",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/bart-base_data_squad_checkpoint-220_squad.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_squad_checkpoint-240_squad.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-squad-checkpoint-220_squad.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-squad-checkpoint-240_squad.json",
        },
    },
    "cnn": {
        "path_test": "data/cnn/test.csv",
        "path_val": "data/cnn/val.csv",
        "metric": "rougeL",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/bart-base_data_cnn_checkpoint-200_cnn.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_cnn_checkpoint-200_cnn.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-cnn-checkpoint-140_cnn.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-cnn-checkpoint-200_cnn.json",
        },
    },
    "reddit": {
        "path_test": "data/reddit/test.csv",
        "path_val": "data/reddit/val.csv",
        "metric": "rougeL",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/facebook-bart-base_data-reddit-checkpoint-140_reddit.json",  # "/home/mila/f/floresl/beam-search/results/final/bart-base_data_reddit_checkpoint-200_reddit.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/google-flan-t5-base_data-reddit-checkpoint-200_reddit.json",  # "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_reddit_checkpoint-200_reddit.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-reddit-checkpoint-140_reddit.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-reddit-checkpoint-200_reddit.json",
        },
    },
    "wmt_de_en": {
        "path_test": "data/wmt_de_en/test.csv",
        "path_val": "data/wmt_de_en/val.csv",
        "metric": "bleu",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/bart-base_data_wmt_de_en_checkpoint-200_wmt_de_en.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_wmt_de_en_checkpoint-200_wmt_de_en.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-wmt_de_en-checkpoint-200_wmt_de_en.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-wmt_de_en-checkpoint-200_wmt_de_en.json",
        },
    },
    "wmt_ru_en": {
        "path_test": "data/wmt_ru_en/test.csv",
        "path_val": "data/wmt_ru_en/val.csv",
        "metric": "bleu",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/bart-base_data-wmt_ru_en-checkpoint-6000_wmt_ru_en.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data-wmt_ru_en-checkpoint-6000_wmt_ru_en.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-wmt_ru_en-checkpoint-6000_wmt_ru_en.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-wmt_ru_en-checkpoint-6000_wmt_ru_en.json",
        },
    },
    "xsum": {
        "path_test": "data/xsum/test.csv",
        "path_val": "data/xsum/val.csv",
        "metric": "rougeL",
        "scores_test": {
            "bart": "/home/mila/f/floresl/beam-search/results/final/facebook-bart-base_data-xsum-checkpoint-120_xsum.json",  # "/home/mila/f/floresl/beam-search/results/final/bart-base_data_xsum_checkpoint-200_xsum.json",
            "t5": "/home/mila/f/floresl/beam-search/results/final/google-flan-t5-base_data-xsum-checkpoint-200_xsum.json",  # "/home/mila/f/floresl/beam-search/results/final/flan-t5-base_data_xsum_checkpoint-200_xsum.json",
        },
        "scores_val": {
            "bart": "/home/mila/f/floresl/beam-search/results/val/facebook-bart-base_data-xsum-checkpoint-120_xsum.json",
            "t5": "/home/mila/f/floresl/beam-search/results/val/google-flan-t5-base_data-xsum-checkpoint-200_xsum.json",
        },
    },
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    args = parser.parse_args()

    metric = str(datasets[args.dataset]["metric"])  # type: ignore
    test_scores_filepath = datasets[args.dataset]["scores_test"][args.model]  # type: ignore
    val_scores_filepath = datasets[args.dataset]["scores_val"][args.model]  # type: ignore
    temperature = args.temperature

    test_filepath = datasets[args.dataset]["path_test"]  # type: ignore
    test_targets = [str(s) for s in pd.read_csv(test_filepath)["target"].fillna("")]  # type: ignore

    val_filepath = datasets[args.dataset]["path_val"]  # type: ignore
    val_targets = [str(s) for s in pd.read_csv(val_filepath)["target"].fillna("")]  # type: ignore

    results = prepare_scores(test_scores_filepath, val_scores_filepath, test_targets, val_targets, metric, temperature)  # type: ignore
    correlations = results.scores_dataframe.drop(  # type: ignore
        ["sentences", "dropout_sentences"], axis=1
    ).corr(method="spearman")[metric]
    print(correlations)  # type: ignore

    scores: dict[str, float] = dict(correlations)  # type: ignore
    scores.pop(metric)
    ratio = scores.pop("beam_score_ratios")
    tail = scores.pop("tail_index")

    scores_list = sorted([(k, v) for (k, v) in scores.items()], key=lambda x: abs(x[1]))
    baseline = scores_list[-1][0]

    confidence_scores_ratio = [float(x) for x in results.scores_dataframe["beam_score_ratios"]]  # type: ignore
    confidence_scores_tail = [float(x) for x in results.scores_dataframe["tail_index"]]  # type: ignore
    confidence_scores_base = [float(x) for x in results.scores_dataframe[baseline]]  # type: ignore
    quality_scores = [float(x) for x in results.scores_dataframe[metric]]  # type: ignore

    p_val_slope = test_correlations(
        quality_scores, confidence_scores_ratio, confidence_scores_base
    )
    p_val_tail = test_correlations(
        quality_scores, confidence_scores_tail, confidence_scores_base
    )
    print(f"Baseline: {baseline}, value: {scores_list[-1][1]}")
    print(f"Slope: {ratio}, p-val: {p_val_slope}")
    print(f"Tail: {tail}, p-val: {p_val_tail}")
