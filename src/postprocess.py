import torch

from torch import Tensor
from typing_extensions import Tuple
from transformers import PreTrainedTokenizer  # type: ignore


def decode_multi_output_sentences(sequences: Tensor, tokenizer: PreTrainedTokenizer):
    outputs = list[list[str]]()
    assert sequences.dim() == 3
    for input_idx in range(sequences.shape[0]):
        sentences = tokenizer.batch_decode(  # type: ignore
            sequences[input_idx],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        outputs.append(sentences)
    return outputs


def reshape_sequence_probs_by_beam(
    sequence_scores: Tensor, batch_size: int, num_beams: int
):
    sequence_scores = sequence_scores.reshape(batch_size, num_beams)
    return sequence_scores


def reshape_token_probs_by_beam(scores: Tuple[Tensor], batch_size: int, num_beams: int):
    token_probs: Tensor = torch.stack(scores, dim=1).softmax(dim=-1)
    _, sequence_length, vocab_size = token_probs.shape
    token_probs = token_probs.reshape(
        batch_size, num_beams, sequence_length, vocab_size
    )
    return token_probs


def decode_beam_search_sentences(
    sequences: Tensor, tokenizer: PreTrainedTokenizer, batch_size: int, num_beams: int
):
    # Input: (batch_size * num_beams) x seq_len
    assert sequences.dim() == 2
    sequences = sequences.reshape(batch_size, num_beams, -1)
    list_of_sentences = decode_multi_output_sentences(sequences, tokenizer)
    return [lst[0] for lst in list_of_sentences]


def decode_monte_carlo_dropout_sentences(
    probabilities: Tensor, tokenizer: PreTrainedTokenizer
):
    assert probabilities.dim() == 4
    token_ids = probabilities.argmax(dim=-1)
    return decode_multi_output_sentences(token_ids, tokenizer)
