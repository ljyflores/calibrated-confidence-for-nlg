import argparse
import json
import pandas as pd

from datasets import Dataset  # type: ignore
from torch import Tensor, device
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
from tqdm import tqdm

from src.preprocess import encode_dataset
from src.confidence import (
    get_confidence_scores,
    merge_confidence_outputs,
    ConfidenceOutput,
)


def set_items_to_device(dictionary: dict[str, Tensor], device: device):
    for item in dictionary:
        dictionary[item] = dictionary[item].to(device)
    return dictionary


def turn_dataset_into_batches(dataset: Dataset, batch_size: int):
    for i in tqdm(range(dataset.num_rows // batch_size)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        item: dict[str, Tensor | list[Tensor]] = dataset[start_idx:end_idx].copy()
        yield item


def main(dataset: str, model: str, num_beams: int):
    MODEL_PATH = model
    DATA_PATH = dataset
    OUTPUT_PATH = f"{'_'.join(MODEL_PATH.split('/')[-3:])}_{DATA_PATH.split('/')[1]}"
    INPUT_COLUMN = "source"
    OUTPUT_COLUMN = "target"
    EVAL_BATCH_SIZE = 1

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # type: ignore
    model_ = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).cuda()  # type: ignore

    df = pd.read_csv(DATA_PATH)  # type: ignore
    ds = Dataset.from_pandas(df)
    ds = encode_dataset(
        ds,
        tokenizer,  # type: ignore
        INPUT_COLUMN,
        OUTPUT_COLUMN,
    )

    # Run the test set, collect the beam scores and the outputs
    confidence_results = ConfidenceOutput()
    for batch in turn_dataset_into_batches(ds, EVAL_BATCH_SIZE):
        batch = set_items_to_device(batch, model_.device)  # type: ignore
        confidence_output = get_confidence_scores(
            model=model_, tokenizer=tokenizer, item=batch, num_beams=num_beams  # type: ignore
        )
        confidence_results = merge_confidence_outputs(
            confidence_results, confidence_output
        )

    # Save
    json.dump(
        confidence_results.to_dict(),  # type: ignore
        open(f"results/{OUTPUT_PATH}.json", "w", encoding="utf-8"),
        ensure_ascii=True,
        indent=4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model", type=str, required=False, default="facebook/bart-base"
    )
    parser.add_argument("--num_beams", type=int, required=False, default=100)
    args_dict = vars(parser.parse_args())

    print(args_dict)
    main(**args_dict)
