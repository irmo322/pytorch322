import torch

import torch322.nn.functional as F

import unittest


class TestMiscellaneous(unittest.TestCase):

    def test_add_constant_channel(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])

        r1 = F.add_constant_channel(a, 0)
        r2 = F.add_constant_channel(a, 1)
        r3 = F.add_constant_channel(a, -2)
        r4 = F.add_constant_channel(a, -1)
        r5 = F.add_constant_channel(a, 0, constant_value=322)
        r6 = F.add_constant_channel(a, 0, location="append")

        expected_r1 = [[1, 1, 1], [1, 2, 3], [4, 5, 6]]
        expected_r2 = [[1, 1, 2, 3], [1, 4, 5, 6]]
        expected_r5 = [[322, 322, 322], [1, 2, 3], [4, 5, 6]]
        expected_r6 = [[1, 2, 3], [4, 5, 6], [1, 1, 1]]

        self.assertListEqual(r1.tolist(), expected_r1)
        self.assertListEqual(r2.tolist(), expected_r2)
        self.assertListEqual(r3.tolist(), expected_r1)
        self.assertListEqual(r4.tolist(), expected_r2)
        self.assertListEqual(r5.tolist(), expected_r5)
        self.assertListEqual(r6.tolist(), expected_r6)
