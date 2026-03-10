from torch322.utils.data.ParallelDataset import ParallelDataset

from ...unittest_utils import myAssertAlmostEqual

import torch
from torch.utils.data import TensorDataset

import unittest


class TestParallelDataset(unittest.TestCase):

    def test_01(self):
        dataset_01 = TensorDataset(
            torch.tensor([
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]
            ])
        )
        dataset_02 = TensorDataset(
            torch.tensor([
                [1, 2],
                [2, 3],
                [3, 4]
            ]),
            torch.tensor([
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]
            ])
        )

        parallel_dataset = ParallelDataset([dataset_01, dataset_02])

        myAssertAlmostEqual(self, 3, len(parallel_dataset))

        expected_dataset = [
            [
                [[1, 2, 3]],
                [[1, 2], [10, 11, 12]]
            ],
            [
                [[2, 3, 4]],
                [[2, 3], [13, 14, 15]]
            ],
            [
                [[3, 4, 5]],
                [[3, 4], [16, 17, 18]]
            ]
        ]

        # print(parallel_dataset[0])

        myAssertAlmostEqual(self, expected_dataset, [*parallel_dataset])
