from pathlib import Path
import json

import torch

import torch322.nn.functional as F

import unittest

from ..unittest_utils import myAssertAlmostEqual

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

        padded_input = F.pad322(input_, channel_dim, spatial_dims, spatial_paddings, padding_values)

        # print(input_.size())
        # print(padded_input.size())
        #
        # for i in range(input_.size(0)):
        #     print(f"i={i}")
        #     print(input_[i])
        #     print(padded_input[i])

        expected_padded_input_file_path = _DATA_FOLDER / "expected_pad322_result.json"

        if generate:
            with open(expected_padded_input_file_path, "w") as f:
                f.write(json.dumps(padded_input.tolist()))

        with open(expected_padded_input_file_path, "r") as f:
            expected_padded_input = json.load(f)

        self.assertListEqual(padded_input.tolist(), expected_padded_input)

    def test_pad_with_indicator_channels(self):
        generate = False  # To generate ref file, launch with generate = True

        input_ = torch.arange(4 * 5 * 2 * 3, dtype=torch.float).view(4, 5, 2, 3)
        channel_dim = 1
        paddings = [
            [3, "append", 2, 3.5],
            [2, "prepend", 3, 0.5],
            [3, "prepend", 1, 2.5],
        ]

        padded_input = F.pad_with_indicator_channels(input_, channel_dim, paddings)

        # print(input_.size())
        # print(padded_input.size())
        #
        # for i in range(input_.size(0)):
        #     print(f"i={i}")
        #     print(input_[i])
        #     print(padded_input[i])

        expected_padded_input_file_path = _DATA_FOLDER / "expected_pad_with_indicator_channels_result.json"

        if generate:
            with open(expected_padded_input_file_path, "w") as f:
                f.write(json.dumps(padded_input.tolist()))

        with open(expected_padded_input_file_path, "r") as f:
            expected_padded_input = json.load(f)

        self.assertListEqual(padded_input.tolist(), expected_padded_input)

        # Idem with negative indexes
        channel_dim = -3
        paddings = [
            [-1, "append", 2, 3.5],
            [-2, "prepend", 3, 0.5],
            [-1, "prepend", 1, 2.5],
        ]

        padded_input_neg = F.pad_with_indicator_channels(input_, channel_dim, paddings)

        self.assertListEqual(padded_input_neg.tolist(), expected_padded_input)

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

    def test_cat_pool_basic_2d(self):
        """Standard 2D case: [B, C, H, W] with 2x2 kernel."""
        x = torch.arange(1 * 2 * 4 * 4).reshape(1, 2, 4, 4).float()
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], kernel_size=[2, 2])
        self.assertEqual(out.shape, (1, 8, 2, 2))

    def test_cat_pool_basic_1d(self):
        """1D spatial case: [B, C, L] with kernel 2."""
        x = torch.arange(1 * 3 * 8).reshape(1, 3, 8).float()
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2], kernel_size=[2])
        self.assertEqual(out.shape, (1, 6, 4))

    def test_cat_pool_basic_3d(self):
        """3D spatial case: [B, C, D, H, W] with 2x2x2 kernel."""
        x = torch.arange(1 * 2 * 4 * 4 * 4).reshape(1, 2, 4, 4, 4).float()
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3, 4], kernel_size=[2, 2, 2])
        self.assertEqual(out.shape, (1, 16, 2, 2, 2))

    def test_cat_pool_no_information_loss(self):
        """All values must be preserved after cat_pool."""
        x = torch.randn(1, 2, 4, 4)
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], kernel_size=[2, 2])
        self.assertEqual(sorted(x.flatten().tolist()), sorted(out.flatten().tolist()))

    def test_cat_pool_asymmetric_kernel(self):
        """Different kernel sizes on each spatial dimension."""
        x = torch.randn(1, 2, 4, 6)
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], kernel_size=[2, 3])
        self.assertEqual(out.shape, (1, 12, 2, 2))

    def test_cat_pool_channel_dim_not_first(self):
        """Channel dimension at a non-standard position."""
        x = torch.randn(4, 4, 2)  # [H, W, C]
        out = F.cat_pool(x, channel_dim=2, spatial_dims=[0, 1], kernel_size=[2, 2])
        self.assertEqual(out.shape, (2, 2, 8))

    def test_cat_pool_redundant_dim_raises(self):
        """Using the same dimension as both channel and spatial should raise."""
        x = torch.randn(1, 2, 4, 4)
        with self.assertRaises(ValueError):
            F.cat_pool(x, channel_dim=1, spatial_dims=[1, 2], kernel_size=[2, 2])

    def test_cat_pool_kernel_spatial_mismatch_raises(self):
        """Mismatched kernel_size and spatial_dims lengths should raise."""
        x = torch.randn(1, 2, 4, 4)
        with self.assertRaises(ValueError):
            F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], kernel_size=[2])

    def test_cat_pool_not_divisible_raises(self):
        """Spatial dimension not divisible by kernel size should raise."""
        x = torch.randn(1, 2, 3, 4)
        with self.assertRaises(ValueError):
            F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], kernel_size=[2, 2])

    def test_cat_pool_negative_dims(self):
        """Negative channel_dim and spatial_dims should give the same result as positive ones."""
        x = torch.randn(1, 2, 4, 4)
        out_positive = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], kernel_size=[2, 2])
        out_negative = F.cat_pool(x, channel_dim=-3, spatial_dims=[-2, -1], kernel_size=[2, 2])
        self.assertTrue(torch.allclose(out_positive, out_negative))

    def test_cat_pool_complex(self):
        x = torch.randn(10, 5, 7, 12)
        y = F.cat_pool(x, channel_dim=1, spatial_dims=[-1, 0], kernel_size=[3, 2])

        x_split = x.reshape([5, 2, 5, 7, 4, 3])
        x_permuted = x_split.permute([0, 5, 1, 2, 3, 4])
        y_expected = x_permuted.reshape([5, 30, 7, 4])

        myAssertAlmostEqual(self, y_expected, y, relative_delta=0.0001)
