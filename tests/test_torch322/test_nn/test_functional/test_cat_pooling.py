import torch

import torch322.nn.functional as F

import unittest

from ...unittest_utils import myAssertAlmostEqual


class TestFunctional(unittest.TestCase):

    def test_cat_pool_basic_2d(self):
        """Standard 2D case: [B, C, H, W] with 2x2 block."""
        x = torch.arange(1 * 2 * 4 * 4).reshape(1, 2, 4, 4).float()
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2, 2])
        self.assertEqual(out.shape, (1, 8, 2, 2))

    def test_cat_pool_basic_1d(self):
        """1D spatial case: [B, C, L] with block 2."""
        x = torch.arange(1 * 3 * 8).reshape(1, 3, 8).float()
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2], block_size=[2])
        self.assertEqual(out.shape, (1, 6, 4))

    def test_cat_pool_basic_3d(self):
        """3D spatial case: [B, C, D, H, W] with 2x2x2 block."""
        x = torch.arange(1 * 2 * 4 * 4 * 4).reshape(1, 2, 4, 4, 4).float()
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3, 4], block_size=[2, 2, 2])
        self.assertEqual(out.shape, (1, 16, 2, 2, 2))

    def test_cat_pool_no_information_loss(self):
        """All values must be preserved after cat_pool."""
        x = torch.randn(1, 2, 4, 4)
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2, 2])
        self.assertEqual(sorted(x.flatten().tolist()), sorted(out.flatten().tolist()))

    def test_cat_pool_asymmetric_block(self):
        """Different block sizes on each spatial dimension."""
        x = torch.randn(1, 2, 4, 6)
        out = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2, 3])
        self.assertEqual(out.shape, (1, 12, 2, 2))

    def test_cat_pool_channel_dim_not_first(self):
        """Channel dimension at a non-standard position."""
        x = torch.randn(4, 4, 2)  # [H, W, C]
        out = F.cat_pool(x, channel_dim=2, spatial_dims=[0, 1], block_size=[2, 2])
        self.assertEqual(out.shape, (2, 2, 8))

    def test_cat_pool_redundant_dim_raises(self):
        """Using the same dimension as both channel and spatial should raise."""
        x = torch.randn(1, 2, 4, 4)
        with self.assertRaises(ValueError):
            F.cat_pool(x, channel_dim=1, spatial_dims=[1, 2], block_size=[2, 2])

    def test_cat_pool_block_spatial_mismatch_raises(self):
        """Mismatched block_size and spatial_dims lengths should raise."""
        x = torch.randn(1, 2, 4, 4)
        with self.assertRaises(ValueError):
            F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2])

    def test_cat_pool_not_divisible_raises(self):
        """Spatial dimension not divisible by block size should raise."""
        x = torch.randn(1, 2, 3, 4)
        with self.assertRaises(ValueError):
            F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2, 2])

    def test_cat_pool_negative_dims(self):
        """Negative channel_dim and spatial_dims should give the same result as positive ones."""
        x = torch.randn(1, 2, 4, 4)
        out_positive = F.cat_pool(x, channel_dim=1, spatial_dims=[2, 3], block_size=[2, 2])
        out_negative = F.cat_pool(x, channel_dim=-3, spatial_dims=[-2, -1], block_size=[2, 2])
        self.assertTrue(torch.allclose(out_positive, out_negative))

    def test_cat_pool_complex(self):
        x = torch.randn(10, 5, 7, 12)
        y = F.cat_pool(x, channel_dim=1, spatial_dims=[-1, 0], block_size=[3, 2])

        x_split = x.reshape([5, 2, 5, 7, 4, 3])
        x_permuted = x_split.permute([0, 5, 1, 2, 3, 4])
        y_expected = x_permuted.reshape([5, 30, 7, 4])

        myAssertAlmostEqual(self, y_expected, y, relative_delta=0.0001)

    def test_cat_unpool_complex(self):
        x = torch.randn(5, 30, 7, 4)
        y = F.cat_unpool(x, channel_dim=1, spatial_dims=[-1, 0], block_size=[3, 2])

        x_split = x.reshape([5, 3, 2, 5, 7, 4])
        x_permuted = x_split.permute([0, 2, 3, 4, 5, 1])
        y_expected = x_permuted.reshape([10, 5, 7, 12])

        myAssertAlmostEqual(self, y_expected, y, relative_delta=0.0001)
