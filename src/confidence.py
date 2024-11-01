import numpy as np

from dataclasses_json import dataclass_json
from dataclasses import dataclass
from itertools import combinations
from torch import Tensor
from torch.nn.functional import kl_div
from transformers import (  # pyright: ignore[reportMissingTypeStubs]
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import Literal
from src.dropout import get_dropout_predictions
from src.eval import calculate_bleu, calculate_meteor
from src.postprocess import *

MAX_NEW_TOKENS = 200


@dataclass_json
@dataclass
class ConfidenceOutput:
    sentences: list[str] | None = None
    dropout_sentences: list[list[str]] | None = None
    length_normalized_log_probs: list[float] | None = None
    importance_weighted_log_probs: dict[int, list[float]] | None = None
    beam_score_log_probs: dict[int, list[float]] | None = None
    beam_score_ratios: dict[int, list[float]] | None = None
    beam_score_sum_top_k: dict[int, list[float]] | None = None
    mean_token_entropy: list[float] | None = None
    dropout_bleu_variance: list[float] | None = None
    dropout_meteor_score: list[float] | None = None
    dropout_entropy: list[float] | None = None
    dropout_disagreement: list[float] | None = None


def update_dictionary(dict_a: dict[int, list[float]], dict_b: dict[int, list[float]]):
    for key in dict_b:
        if key in dict_a:
            dict_a[key].extend(dict_b[key])
        else:
            dict_a[key] = dict_b[key]
    return dict_a


def merge_confidence_outputs(conf1: ConfidenceOutput, conf2: ConfidenceOutput):
    return ConfidenceOutput(
        sentences=(conf1.sentences or []) + (conf2.sentences or []),
        dropout_sentences=(conf1.dropout_sentences or [])
        + (conf2.dropout_sentences or []),
        length_normalized_log_probs=(conf1.length_normalized_log_probs or [])
        + (conf2.length_normalized_log_probs or []),
        importance_weighted_log_probs=update_dictionary(
            (conf1.importance_weighted_log_probs or {}),
            (conf2.importance_weighted_log_probs or {}),
        ),
        beam_score_ratios=update_dictionary(
            (conf1.beam_score_ratios or {}), (conf2.beam_score_ratios or {})
        ),
        beam_score_log_probs=update_dictionary(
            (conf1.beam_score_log_probs or {}), (conf2.beam_score_log_probs or {})
        ),
        beam_score_sum_top_k=update_dictionary(
            (conf1.beam_score_sum_top_k or {}), (conf2.beam_score_sum_top_k or {})
        ),
        mean_token_entropy=(conf1.mean_token_entropy or [])
        + (conf2.mean_token_entropy or []),
        dropout_bleu_variance=(conf1.dropout_bleu_variance or [])
        + (conf2.dropout_bleu_variance or []),
        dropout_meteor_score=(conf1.dropout_meteor_score or [])
        + (conf2.dropout_meteor_score or []),
        dropout_entropy=(conf1.dropout_entropy or []) + (conf2.dropout_entropy or []),
        dropout_disagreement=(conf1.dropout_disagreement or [])
        + (conf2.dropout_disagreement or []),
    )


def compute_average_pairwise_similarity(
    texts: list[str], similarity_method: Literal["bleu", "meteor"]
):
    id_pairs = combinations(iterable=list(range(len(texts))), r=2)
    # Monte Carlo Dropout: https://arxiv.org/pdf/2305.15040
    if similarity_method == "bleu":
        bleu_scores = map(
            lambda pair: calculate_bleu([texts[pair[0]]], [[texts[pair[1]]]]),
            id_pairs,
        )
        score = float(np.var(list(bleu_scores)))
    # Dropout Based Lexical Similarity: https://arxiv.org/pdf/2211.14880
    elif similarity_method == "meteor":
        meteor_scores = map(
            lambda pair: calculate_meteor([texts[pair[0]]], [[texts[pair[1]]]]),
            id_pairs,
        )
        score = float(np.mean(list(meteor_scores)))
    else:
        raise ValueError("Similarity method is not recognized")
    return score


def compute_monte_carlo_mean_token_entropy(dropout_token_probs_tensor: Tensor):
    # Input: batch_size x num_dropout x seq_len x vocab_size
    return [
        float(np.mean(compute_mean_token_entropy(dropout_token_probs_tensor[i])))
        for i in range(dropout_token_probs_tensor.shape[0])
    ]


def reorganize_beam_scores(scores_per_beam: Tensor):
    # Input: batch_size x num_beams
    assert scores_per_beam.dim() == 2
    scores_by_k_dict = {
        k: [float(x) for x in scores_per_beam[:, k]]
        for k in range(scores_per_beam.shape[1])
    }
    return scores_by_k_dict


def compute_beam_score_sum_top_k(scores_per_beam: Tensor):
    # Input: batch_size x num_beams
    assert scores_per_beam.dim() == 2
    scores_by_k_dict = {
        k: [float(x) for x in scores_per_beam[:, : k + 1].sum(dim=-1)]
        for k in range(scores_per_beam.shape[1])
    }
    return scores_by_k_dict


def compute_beam_score_ratios(scores_per_beam: Tensor):
    # Input: batch_size x num_beams
    assert scores_per_beam.dim() == 2
    best_beam_score = scores_per_beam[:, 0]
    scores_by_k_dict = {
        k: [float(x) for x in (best_beam_score - scores_per_beam[:, k]).exp()]
        for k in range(scores_per_beam.shape[1])
    }
    return scores_by_k_dict


def compute_importance_weighted_log_probs(scores_per_beam: Tensor):
    # Input: batch_size x num_beams
    assert scores_per_beam.dim() == 2
    scores_by_k_dict = dict[int, list[float]]()
    for i in range(scores_per_beam.shape[1]):
        top_k_scores = scores_per_beam[:, : i + 1]
        beam_probs = top_k_scores.exp()
        beam_importance_weights = beam_probs / beam_probs.sum(dim=-1).unsqueeze(-1)
        scores = (-1.0 * beam_importance_weights * top_k_scores).sum(dim=-1)
        scores = [float(x) for x in scores]
        scores_by_k_dict[i] = scores
    return scores_by_k_dict


def compute_mean_token_entropy(token_probs_tensor: Tensor):
    stability_constant = 1e-20
    # Input: batch_size x num_beams x vocab_size
    assert token_probs_tensor.dim() == 3
    entropy = (
        -1.0 * token_probs_tensor * (token_probs_tensor + stability_constant).log()
    )
    return [float(x) for x in entropy.mean(dim=2).mean(dim=1).detach()]


def compute_disagreement(dropout_token_probs: Tensor):
    assert dropout_token_probs.dim() == 4
    # actual_token_probs: batch_size x seq_len x vocab_size
    # dropout_token_probs: batch_size x num_dropout x seq_len x vocab_size
    _, num_dropout, _, _ = dropout_token_probs.shape
    mean_dropout_token_probs = dropout_token_probs.mean(dim=1)
    mean_dropout_token_probs = mean_dropout_token_probs.unsqueeze(dim=1).repeat(
        1, num_dropout, 1, 1
    )
    kl_div_tensor = kl_div(
        mean_dropout_token_probs.log(), dropout_token_probs, reduction="none"
    )
    disagreement_scores = kl_div_tensor.sum(dim=-1).mean(dim=-1).sum(dim=-1)
    return [float(x) for x in disagreement_scores]


def get_confidence_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    item: dict[str, Tensor],
    num_beams: int = 100,
):
    beam_search_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        return_dict_in_generate=True,
        output_scores=True,
    )

    batch_size, _ = item["input_ids"].shape
    model.eval()
    output = model.generate(  # type: ignore
        input_ids=item.get("input_ids"),
        generation_config=beam_search_config,
    )  # type: ignore
    dropout_probs = get_dropout_predictions(
        model,
        item,
        num_dropout_samples=10,
    )

    # Post-process outputs
    sentences = decode_beam_search_sentences(
        output.sequences, tokenizer, batch_size, num_beams  # type: ignore
    )
    sequence_probs = reshape_sequence_probs_by_beam(
        output.sequences_scores, batch_size, num_beams  # type: ignore
    )
    token_probs = reshape_token_probs_by_beam(output.scores, batch_size, num_beams)  # type: ignore
    dropout_sentences = decode_monte_carlo_dropout_sentences(dropout_probs, tokenizer)

    # Compute beam scores
    scores_beam_score_log_probs = reorganize_beam_scores(sequence_probs)

    # Compute beam score ratios
    scores_beam_score_ratios = compute_beam_score_ratios(sequence_probs)

    # Compute beam score sums
    scores_beam_score_sums = compute_beam_score_sum_top_k(sequence_probs)

    # Compute length normalized log probs
    scores_length_norm_log_probs = [float(x) for x in sequence_probs[:, 0]]

    # Compute importance weighted probs
    scores_importance_weighted_log_probs = compute_importance_weighted_log_probs(
        sequence_probs
    )

    # Compute mean token entropy
    scores_mean_token_entropy = compute_mean_token_entropy(token_probs[:, 0, :, :])

    # Compute BLEU and Meteor scores
    scores_dropout_bleu_variance = [
        compute_average_pairwise_similarity(sents, "bleu")
        for sents in dropout_sentences
    ]

    scores_dropout_meteor_score = [
        compute_average_pairwise_similarity(sents, "meteor")
        for sents in dropout_sentences
    ]

    # Compute dropout disagreement
    scores_dropout_disagreement = compute_disagreement(dropout_probs)

    scores_dropout_entropy = compute_monte_carlo_mean_token_entropy(dropout_probs)

    return ConfidenceOutput(
        sentences=sentences,
        dropout_sentences=dropout_sentences,
        length_normalized_log_probs=scores_length_norm_log_probs,
        importance_weighted_log_probs=scores_importance_weighted_log_probs,
        beam_score_log_probs=scores_beam_score_log_probs,
        beam_score_ratios=scores_beam_score_ratios,
        beam_score_sum_top_k=scores_beam_score_sums,
        mean_token_entropy=scores_mean_token_entropy,
        dropout_bleu_variance=scores_dropout_bleu_variance,
        dropout_meteor_score=scores_dropout_meteor_score,
        dropout_entropy=scores_dropout_entropy,
        dropout_disagreement=scores_dropout_disagreement,
    )
