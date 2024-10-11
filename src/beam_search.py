from torch import Tensor
from transformers import (  # pyright: ignore[reportMissingTypeStubs]
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

NUM_BEAMS = 100
MAX_NEW_TOKENS = 200

beam_search_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    num_beams=NUM_BEAMS,
    num_return_sequences=NUM_BEAMS,
    return_dict_in_generate=True,
    output_scores=True,
)


def get_beam_score_output(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, item: dict[str, Tensor]
):
    batch_size, _ = item["input_ids"].shape
    output = model.generate(  # type: ignore
        input_ids=item.get("input_ids"),
        generation_config=beam_search_config,
    )  # type: ignore

    # Get sequences
    sequences: Tensor = output.sequences.reshape(  # type: ignore
        batch_size, beam_search_config.num_beams, -1  # type: ignore
    )[
        :, 0, :
    ]  # Get top beam
    sentences = tokenizer.batch_decode(  # type: ignore
        sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Compute scores
    sequence_scores: Tensor = output.sequences_scores  # type: ignore
    scores_per_item = sequence_scores.reshape(batch_size, NUM_BEAMS)
    best_beam_score, worst_beam_score = scores_per_item[:, 0], scores_per_item[:, -1]
    best_to_worst_ratio = (best_beam_score - worst_beam_score).exp()
    scores = [float(score) for score in best_to_worst_ratio]
    return sentences, scores
