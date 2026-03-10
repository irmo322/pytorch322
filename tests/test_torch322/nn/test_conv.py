from torch322.nn.conv import Conv2dStdWeight, ConvTranspose2dStdWeight

import torch

import unittest


class TestConv(unittest.TestCase):

    def test_conv2d_std_weight_stat_property(self):
        # This test may exceptionally fail as it checks
        # some statistical properties using uncontrolled randomness.
        batch_size = 100
        in_channels = 16
        out_channels = 10
        kernel_size = 8
        stride = 3
        groups = 2
        bias = True
        input_width = 30
        input_height = 25
        expected_output_width = 8
        expected_output_height = 6

        lin = Conv2dStdWeight(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias)
        input_tensor = torch.randn(batch_size, in_channels, input_width, input_height)
        output_tensor = lin(input_tensor)

        self.assertListEqual(
            list(output_tensor.size()),
            [batch_size, out_channels, expected_output_width, expected_output_height]
        )

        mean_tensor = torch.mean(output_tensor, dim=0)
        std_tensor = torch.std(output_tensor, dim=0)

        # for q in [0.1 * i for i in range(1, 10)]:
        #     print(f'q = {q}')
        #     print("mean quantile:", mean_tensor.quantile(q).item())
        #     print("std quantile:", std_tensor.quantile(q).item())

        self.assertGreater(mean_tensor.quantile(0.25).item(), -0.2)
        self.assertLess(mean_tensor.quantile(0.75).item(), +0.2)

        self.assertGreater(std_tensor.quantile(0.25).item(), 0.8)
        self.assertLess(std_tensor.quantile(0.75).item(), 1.2)
