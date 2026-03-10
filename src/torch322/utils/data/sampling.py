import math
import itertools


def divide_big_weights(weights, weight_limit):
    """
    Create a list of (index, weight) couples such that for any index 0 <= i < len(weights), the list contains
    (i, 1/k) repeated k times, where k=ceil(weight/weight_limit)
    :param weights: list of positive numbers.
    :param weight_limit: limit of weight.
    :return: the list of (index, weight).
    """
    assert weight_limit > 0
    chunks = []
    for index, weight in enumerate(weights):
        assert weight >= 0
        k = math.ceil(weight / weight_limit)
        if k == 0:
            k = 1
        chunks.append([(index, 1 / k)] * k)
    result = list(itertools.chain.from_iterable(chunks))
    return result
