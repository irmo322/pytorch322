from torch322.nn.linear import LinearStdWeight

import torch

import unittest


class TestLinear(unittest.TestCase):

    def test_linear_std_weight_stat_property(self):
        # This test may exceptionally fail as it checks
        # some statistical properties using uncontrolled randomness.
        batch_size = 200
        in_features = 100
        out_features = 80

        lin = LinearStdWeight(in_features, out_features, bias=True)
        input_tensor = torch.randn(batch_size, in_features)
        output_tensor = lin(input_tensor)

        self.assertListEqual(list(output_tensor.size()), [batch_size, out_features])

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
