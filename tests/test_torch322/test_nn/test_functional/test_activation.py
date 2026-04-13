import torch

import torch322.nn.functional as F

import unittest


class TestActivation(unittest.TestCase):

    def test_crelu(self):
        x = torch.tensor([[1, -2, 3, -4], [-5, -6, 7, 8]])

        y_a = F.crelu(x, dim=0)
        y_b = F.crelu(x, dim=1)
        y_c = F.crelu(x, dim=-2)
        y_d = F.crelu(x, dim=-1)

        y_1 = [[1, 0, 3, 0], [0, 0, 7, 8], [0, 2, 0, 4], [5, 6, 0, 0]]
        y_2 = [[1, 0, 3, 0, 0, 2, 0, 4], [0, 0, 7, 8, 5, 6, 0, 0]]

        self.assertListEqual(y_1, y_a.tolist())
        self.assertListEqual(y_1, y_c.tolist())

        self.assertListEqual(y_2, y_b.tolist())
        self.assertListEqual(y_2, y_d.tolist())
