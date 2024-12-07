import numpy as np
from torch import Tensor
from scipy.special import kl_div  # type: ignore
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy  # type: ignore
from typing import Any


def compute_js_divergence(list1: list[float], list2: list[float]):  # type: ignore
    # Standardize to become probabilities
    list1 = [x / sum(list1) for x in list1]
    list2 = [x / sum(list2) for x in list2]
    return jensenshannon(list1, list2)


def compute_total_difference(list1: list[float], list2: list[float]):
    return sum(list1) - sum(list2)


def compute_total_ratio(list1: list[float], list2: list[float]):
    return sum(list1) / sum(list2)


def compute_beam_score(list1: list[float], list2: list[float]):
    return np.exp(np.log(list2[0]) - np.log(list2[-1]))


def compute_largest_slope(list1: list[float], list2: list[float]):
    slope = 0
    for i in range(len(list2) - 1):
        for j in range(i + 1, len(list2)):
            slope = max(slope, abs(list2[j] - list2[i]) / (j - i))
    return slope


def compute_top_k_sumd_etpy(list1: list[float], list2: list[float], k: int = 3):  # type: ignore
    new_list = [sum(list2[:k])] + list2[k:]
    new_list = [x / sum(new_list) for x in new_list]
    return entropy(new_list)  # type: ignore


def softmax(
    nums_: Tensor | np.ndarray[float, Any] | list[float], temperature: float = 1
):
    nums = np.array(nums_)
    nums_w_temp = nums / temperature
    nums_w_exp = np.exp(nums_w_temp)
    return nums_w_exp / nums_w_exp.sum()


def gini_coefficient(x: list[float]):
    """Compute Gini coefficient of array of values"""
    diffsum: float = 0.0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += float(np.sum(np.abs(xi - x[i:])))  # type: ignore
    return float(diffsum / (len(x) ** 2 * np.mean(x)))


def mad_sd_ratio(nums_: list[float]):
    nums = np.array(nums_)
    ad = nums - nums.mean()
    mad = np.abs(ad).mean()
    sd = np.std(nums)
    return mad / sd


def compute_js_divergence(list1: list[float], list2: list[float]):
    # Standardize to become probabilities
    list1 = softmax(list1)
    list2 = softmax(list2)
    return jensenshannon(list1, list2)


def compute_kl_divergence(list1: list[float], list2: list[float]):
    # Standardize to become probabilities
    list1 = softmax(list1)
    list2 = softmax(list2)
    return kl_div(list1, list2).sum()


def tail_index(nums_: list[float]):
    nums = np.array(nums_)
    nums = nums / nums.sum()
    return float(np.sum(nums**2) / 2.0)


def compute_tail_index(log_probs_by_sample: np.ndarray[Any, np.dtype[np.float64]]):
    tail_indices = list[float]()
    for i in range(log_probs_by_sample.shape[0]):
        probs = softmax(log_probs_by_sample[i], temperature=1)
        tail_indices.append(tail_index(probs))
    return tail_indices


def compute_js_from_uniform(log_probs_by_sample: np.ndarray[Any, np.dtype[np.float64]]):
    js_distances = list[float]()
    n_beams = log_probs_by_sample.shape[1]
    for i in range(log_probs_by_sample.shape[0]):
        js = jensenshannon(
            softmax(log_probs_by_sample[i], temperature=1),
            np.array([1 / n_beams] * n_beams),
        )
        js_distances.append(float(js))
    return js_distances
