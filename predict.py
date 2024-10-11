import argparse
import numpy as np
import pandas as pd

from datasets import Dataset
from torch import Tensor, device
from transformers import Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, DataCollatorForSeq2Seq  # type: ignore
from tqdm import tqdm

from src.preprocess import encode_dataset
from src.beam_search import get_beam_score_output


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


def main(dataset: str, model: str):
    MODEL_PATH = model
    DATA_PATH = dataset
    OUTPUT_PATH = f"{MODEL_PATH.split('/')[-1]}_{DATA_PATH.split('/')[1]}"
    INPUT_COLUMN = "source"
    OUTPUT_COLUMN = "target"
    EVAL_BATCH_SIZE = 2

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # type: ignore
    model_ = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).cuda()  # type: ignore

    df = pd.read_csv(DATA_PATH)  # type: ignore
    ds = Dataset.from_pandas(df)
    ds = encode_dataset(
        ds,
        tokenizer,
        INPUT_COLUMN,
        OUTPUT_COLUMN,
    )

    # Run the test set, collect the beam scores and the outputs
    sentences, scores = list[str](), list[float]()
    for batch in turn_dataset_into_batches(ds, EVAL_BATCH_SIZE):
        batch = set_items_to_device(batch, model_.device)
        batch_sentences, batch_scores = get_beam_score_output(model_, tokenizer, batch)
        sentences.extend(batch_sentences)
        scores.extend(batch_scores)

    # Save
    with open(f"results/{OUTPUT_PATH}_sentences", "w") as f:
        for line in sentences:
            f.write(f"{line}\n")
    np.array(scores).tofile(f"results/{OUTPUT_PATH}_scores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model", type=str, required=False, default="facebook/bart-base"
    )
    args_dict = vars(parser.parse_args())

    print(args_dict)
    main(**args_dict)
