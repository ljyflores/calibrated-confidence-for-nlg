import argparse
import pandas as pd

from analysis.utils_analysis import prepare_scores

datasets = {
    "debatesum": {
        "path": "data/debatesum/test.csv",
        "metric": "rougeL",
        "scores": {
            "bart": "results/bart-base_data_debatesum_checkpoint-1500_debatesum.json",
            "t5": "/home/mila/f/floresl/beam-search/results/flan-t5-base_data_debatesum_checkpoint-1500_debatesum.json",
        },
    },
    "flores": {
        "path": "data/flores/test.csv",
        "metric": "bleu",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/bart-base_data_flores_checkpoint-1000_flores.json",
            "t5": "/home/mila/f/floresl/beam-search/results/flan-t5-base_data_flores_checkpoint-1000_flores.json",
        },
    },
    "hotpotqa": {
        "path": "data/hotpotqa/test.csv",
        "metric": "f1",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/bart-base_data_hotpotqa_checkpoint-26835_hotpotqa.json",
            "t5": "/home/mila/f/floresl/beam-search/results/flan-t5-base_data_hotpotqa_checkpoint-26835_hotpotqa.json",
        },
    },
    "squad": {
        "path": "data/squad/test.csv",
        "metric": "f1",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/bart-base_data_squad_checkpoint-26280_squad.json",
            "t5": "/home/mila/f/floresl/beam-search/results/flan-t5-base_data_squad_checkpoint-26280_squad.json",
        },
    },
    "cnn": {
        "path": "data/cnn/test.csv",
        "metric": "rougeL",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/ljyflores_facebook-bart-base_data-cnn-checkpoint-6000_cnn.json",
            "t5": "/home/mila/f/floresl/beam-search/results/ljyflores_google-flan-t5-base_data-cnn-checkpoint-6000_cnn.json",
        },
    },
    "reddit": {
        "path": "data/reddit/test.csv",
        "metric": "rougeL",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/ljyflores_facebook-bart-base_data-reddit-checkpoint-6000_reddit.json",
            "t5": "/home/mila/f/floresl/beam-search/results/ljyflores_google-flan-t5-base_data-reddit-checkpoint-6000_reddit.json",
        },
    },
    "wmt_de_en": {
        "path": "data/wmt_de_en/test.csv",
        "metric": "bleu",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/ljyflores_facebook-bart-base_data-wmt_de_en-checkpoint-6000_wmt_de_en.json",
            "t5": "/home/mila/f/floresl/beam-search/results/ljyflores_google-flan-t5-base_data-wmt_de_en-checkpoint-6000_wmt_de_en.json",
        },
    },
    "wmt_ru_en": {
        "path": "data/wmt_ru_en/test.csv",
        "metric": "bleu",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/ljyflores_facebook-bart-base_data-wmt_ru_en-checkpoint-6000_wmt_ru_en.json",
            "t5": "/home/mila/f/floresl/beam-search/results/ljyflores_google-flan-t5-base_data-wmt_ru_en-checkpoint-6000_wmt_ru_en.json",
        },
    },
    "xsum": {
        "path": "data/xsum/test.csv",
        "metric": "rougeL",
        "scores": {
            "bart": "/home/mila/f/floresl/beam-search/results/ljyflores_facebook-bart-base_data-xsum-checkpoint-6000_xsum.json",
            "t5": "/home/mila/f/floresl/beam-search/results/ljyflores_google-flan-t5-base_data-xsum-checkpoint-6000_xsum.json",
        },
    },
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    args = parser.parse_args()

    metric = datasets[args.dataset]["metric"]
    test_filepath = datasets[args.dataset]["path"]
    scores_filepath = datasets[args.dataset]["scores"][args.model]
    temperature = args.temperature

    targets = [str(s) for s in pd.read_csv(test_filepath)["target"].fillna("")]

    results = prepare_scores(scores_filepath, targets, metric, temperature)

    print(
        results.scores_dataframe.drop(["sentences", "dropout_sentences"], axis=1).corr(
            method="spearman"
        )[metric]
    )
