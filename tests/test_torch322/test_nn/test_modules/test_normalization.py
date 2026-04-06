import torch

from torch322.nn import ChannelNorm

import unittest

from ...unittest_utils import myAssertAlmostEqual


class TestNormalization(unittest.TestCase):

    def test_channel_norm(self):
        channel_norm = ChannelNorm(-2, constant_channel=True)

        x = torch.tensor([
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
            ],
            [
                [0.0, 1000.0, 1.0, -1.0],
                [0.0, 1000.0, 2.0, 2.0],
                [0.0, 1000.0, 3.0, -2.0],
            ]
        ])

        y_expected = torch.tensor([
            [
                [ 1.0000e+00,  1.0000e+00,  1.1547e+00,  1.4142e+00],
                [ 1.0000e+00,  1.0000e+00,  1.1547e+00,  1.4142e+00],
                [ 1.0000e+00, -1.0000e+00, -1.1547e+00,  0.0000e+00],
                [ 1.0000e+00, -1.0000e+00,  0.0000e+00,  0.0000e+00],
            ],
            [
                [ 2.0000e+00,  1.1547e-03,  5.1640e-01,  6.3246e-01],
                [ 0.0000e+00,  1.1547e+00,  5.1640e-01, -6.3246e-01],
                [ 0.0000e+00,  1.1547e+00,  1.0328e+00,  1.2649e+00],
                [ 0.0000e+00,  1.1547e+00,  1.5492e+00, -1.2649e+00],
            ]
        ])

        y = channel_norm(x)

        # print(y)

        myAssertAlmostEqual(self, y, y_expected, relative_delta=0.0001)
