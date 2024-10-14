import argparse
import os
import pandas as pd

from datasets import Dataset  # type: ignore
from transformers import Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, DataCollatorForSeq2Seq  # type: ignore

from src.preprocess import encode_dataset


def main(
    dataset: str,
    epochs: int,
):
    DATA_PATH = dataset
    MODEL_PATH = "google/flan-t5-base"  # "facebook/bart-base"
    MODEL_OUTPUT_PATH = f"{MODEL_PATH}_{DATA_PATH}"
    TRAIN_BATCH_SIZE = 5
    TRAIN_GRAD_ACC = 2
    INPUT_COLUMN = "source"
    OUTPUT_COLUMN = "target"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # type: ignore
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)  # type: ignore

    training_df = pd.read_csv(f"{DATA_PATH}/train.csv")  # type: ignore
    validation_df = pd.read_csv(f"{DATA_PATH}/test.csv")  # type: ignore
    # test_df = pd.read_csv(f"{DATA_PATH}/test.csv")

    training_pool = Dataset.from_pandas(training_df)
    validation_pool = Dataset.from_pandas(validation_df)
    # test_pool = Dataset.from_pandas(test_df)

    training_dataset = encode_dataset(
        training_pool,
        tokenizer,  # type: ignore
        INPUT_COLUMN,
        OUTPUT_COLUMN,
    )
    validation_dataset = encode_dataset(
        validation_pool,
        tokenizer,  # type: ignore
        INPUT_COLUMN,
        OUTPUT_COLUMN,
    )

    training_args = Seq2SeqTrainingArguments(
        f"outputs/{MODEL_OUTPUT_PATH}",
        # Training parameters
        num_train_epochs=epochs,
        learning_rate=5e-5,
        warmup_steps=0,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=TRAIN_GRAD_ACC,
        gradient_checkpointing=True,
        fp16=False if "flan-t5" in MODEL_PATH else True,
        lr_scheduler_type="constant",
        # Evaluation parameters
        eval_strategy="epoch",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        # generation_max_length=768,
        include_inputs_for_metrics=True,
        # Logging parameters
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",
        run_name=MODEL_OUTPUT_PATH,
        # Saving parameters
        save_strategy="epoch",
        save_total_limit=3,
    )

    trainer = Seq2SeqTrainer(
        model=model,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=training_dataset,  # type: ignore
        eval_dataset=validation_dataset,  # type: ignore
        data_collator=DataCollatorForSeq2Seq(tokenizer),
    )
    trainer.train()  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3, required=False)
    parser.add_argument(
        "--wandb_project_name", type=str, required=False, default="huggingface"
    )
    args_dict = vars(parser.parse_args())
    os.environ["WANDB_PROJECT"] = args_dict.pop("wandb_project_name")

    print(args_dict)
    main(**args_dict)
