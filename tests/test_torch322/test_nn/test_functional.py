from torch322.nn.functional import pad322, pad323, crelu

from pathlib import Path
import json

import torch

import unittest


_HERE = Path(__file__).parent
_DATA_FOLDER = _HERE / "../../data"


class TestFunctional(unittest.TestCase):

    def test_pad322(self):
        generate = False  # To generate ref file, launch with generate = True

        input_ = torch.arange(4 * 5 * 2 * 3, dtype=torch.float).view(4, 5, 2, 3)
        channel_dim = 1
        spatial_dims = [2, 3]
        spatial_paddings = [[3, 4], [1, 2]]
        padding_values = [[0.5, 1.5], [2.5, 3.5]]

        padded_input = pad322(input_, channel_dim, spatial_dims, spatial_paddings, padding_values)

        print(input_.size())
        print(padded_input.size())

        for i in range(input_.size(0)):
            print(f"i={i}")
            print(input_[i])
            print(padded_input[i])

        expected_padded_input_file_path = _DATA_FOLDER / "expected_pad322_result.json"

        if generate:
            with open(expected_padded_input_file_path, "w") as f:
                f.write(json.dumps(padded_input.tolist()))

        with open(expected_padded_input_file_path, "r") as f:
            expected_padded_input = json.load(f)

        self.assertListEqual(padded_input.tolist(), expected_padded_input)

    def test_pad323(self):
        generate = False  # To generate ref file, launch with generate = True

        input_ = torch.arange(4 * 5 * 2 * 3, dtype=torch.float).view(4, 5, 2, 3)
        channel_dim = 1
        paddings = [
            [3, "high", 2, 3.5],
            [2, "low", 3, 0.5],
            [3, "low", 1, 2.5],
        ]

        padded_input = pad323(input_, channel_dim, paddings)

        print(input_.size())
        print(padded_input.size())

        for i in range(input_.size(0)):
            print(f"i={i}")
            print(input_[i])
            print(padded_input[i])

        expected_padded_input_file_path = _DATA_FOLDER / "expected_pad323_result.json"

        if generate:
            with open(expected_padded_input_file_path, "w") as f:
                f.write(json.dumps(padded_input.tolist()))

        with open(expected_padded_input_file_path, "r") as f:
            expected_padded_input = json.load(f)

        self.assertListEqual(padded_input.tolist(), expected_padded_input)

        # Idem with negative indexes
        channel_dim = -3
        paddings = [
            [-1, "high", 2, 3.5],
            [-2, "low", 3, 0.5],
            [-1, "low", 1, 2.5],
        ]

        padded_input_neg = pad323(input_, channel_dim, paddings)

        self.assertListEqual(padded_input_neg.tolist(), expected_padded_input)

    def test_crelu(self):
        x = torch.tensor([[1, -2, 3, -4], [-5, -6, 7, 8]])

        y_a = crelu(x, dim=0)
        y_b = crelu(x, dim=1)
        y_c = crelu(x, dim=-2)
        y_d = crelu(x, dim=-1)

        y_1 = [[1, 0, 3, 0], [0, 0, 7, 8], [0, 2, 0, 4], [5, 6, 0, 0]]
        y_2 = [[1, 0, 3, 0, 0, 2, 0, 4], [0, 0, 7, 8, 5, 6, 0, 0]]

        self.assertListEqual(y_1, y_a.tolist())
        self.assertListEqual(y_1, y_c.tolist())

        self.assertListEqual(y_2, y_b.tolist())
        self.assertListEqual(y_2, y_d.tolist())
