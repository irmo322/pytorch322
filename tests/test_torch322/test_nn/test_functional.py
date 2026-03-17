from torch322.nn.functional import pad322

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

        expected_padded_input_file_path = _DATA_FOLDER / "expected_padded_input.json"

        if generate:
            with open(expected_padded_input_file_path, "w") as f:
                f.write(json.dumps(padded_input.tolist()))

        with open(expected_padded_input_file_path, "r") as f:
            expected_padded_input = json.load(f)

        self.assertListEqual(padded_input.tolist(), expected_padded_input)
