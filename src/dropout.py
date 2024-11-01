import torch

from torch import Tensor
from transformers import (  # pyright: ignore[reportMissingTypeStubs]
    PreTrainedModel,
    GenerationConfig,
)
from typing import Tuple


def get_dropout_predictions(
    model: PreTrainedModel,
    item: dict[str, Tensor],
    num_dropout_samples: int = 10,
):
    config = GenerationConfig(
        max_new_tokens=200,
        num_beams=1,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Set to train mode, so that dropout is activated during the forward pass
    model.train()

    with torch.no_grad():

        # Expand the batch dimension, which effectively performs `num_models`forward passes
        item.pop("labels")
        batch_size, _ = item["input_ids"].shape

        # Generate num_models copies of each entry
        # This turns the tensor from batch_size x seq_len
        # into (batch_size * num_models) x seq_len
        item["input_ids"] = torch.repeat_interleave(
            item["input_ids"].clone(), num_dropout_samples, dim=0
        ).to(model.device)
        item["attention_mask"] = torch.repeat_interleave(
            item["attention_mask"].clone(), num_dropout_samples, dim=0
        ).to(model.device)

        # Get the logits
        token_scores: Tuple[Tensor] = model.generate(  # type: ignore
            input_ids=item.get("input_ids"),
            generation_config=config,
        ).scores  # type: ignore
        probs = torch.stack(token_scores, dim=1).softmax(dim=-1)
        _, seq_len, vocab_size = probs.shape
        probs = probs.reshape(batch_size, num_dropout_samples, seq_len, vocab_size)

    model.eval()  # Return the model to eval mode
    return probs
