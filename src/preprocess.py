from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
from transformers import PreTrainedTokenizer  # pyright: ignore[reportMissingTypeStubs]


def prepare_encode_function(
    tokenizer: PreTrainedTokenizer,
    input_column_name: str = "input",
    output_column_name: str = "label",
):

    def encode(examples: dict[str, list[str]]):
        """This function takes a batch of samples,
        and tokenizes them into IDs for the model."""
        # Tokenize the Findings (the input)
        model_inputs = tokenizer(
            [str(s) for s in examples[input_column_name]],
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize the Impressions (the output)
        labels = tokenizer(
            [str(s) for s in examples[output_column_name]],
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # Set the label as the token ids (i.e. the vocab IDs) of the findings
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return encode


def encode_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    input_column_name: str,
    output_column_name: str,
):
    encoding_fn = prepare_encode_function(
        tokenizer, input_column_name, output_column_name
    )
    columns_to_remove = list(
        set(dataset.column_names).difference(
            set(["input_ids", "attention_mask", "labels"])
        )
    )
    prepared_dataset = dataset.map(  # pyright: ignore[reportUnknownMemberType]
        encoding_fn, batched=True, remove_columns=columns_to_remove
    )
    prepared_dataset.set_format(  # pyright: ignore[reportUnknownMemberType]
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return prepared_dataset
