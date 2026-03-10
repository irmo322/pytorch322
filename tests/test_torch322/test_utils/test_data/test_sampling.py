from torch322.utils.data.sampling import divide_big_weights

from ...unittest_utils import myAssertAlmostEqual

import unittest


class TestSampling(unittest.TestCase):

    def test_divide_big_weights(self):
        weights = [2, 24, 48, 12.5, 0, 52]
        weight_limit = 25

        result_ref = [(0, 1), (1, 1), (2, 1/2), (2, 1/2), (3, 1), (4, 1), (5, 1/3), (5, 1/3), (5, 1/3)]
        result = divide_big_weights(weights, weight_limit)

        myAssertAlmostEqual(self, result_ref, result)
