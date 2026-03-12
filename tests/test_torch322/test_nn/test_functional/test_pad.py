from torch322.nn.functional.pad import pad322

import torch

import unittest


class TestPad(unittest.TestCase):

    def test_pad322(self):
        input_ = torch.randn(4, 5, 2, 3)
        channel_dim = 1
        spatial_dims = [2, 3]
        spatial_paddings = [[3, 4], [1, 2]]
        padding_values = [[0.5, 0.6], [0.7, 0.8]]

        padded_input = pad322(input_, channel_dim, spatial_dims, spatial_paddings, padding_values)

        print(input_.size())
        print(padded_input.size())

        for i in range(input_.size(0)):
            print(f"i={i}")
            print(input_[i])
            print(padded_input[i])

        self.assertLess(list(padded_input.size()), [4, 7, 9, 6])


