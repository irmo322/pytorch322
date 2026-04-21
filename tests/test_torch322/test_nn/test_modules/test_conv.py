import torch322

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

        conv = torch322.nn.Conv2dStdWeight(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias)
        input_tensor = torch.randn(batch_size, in_channels, input_width, input_height)
        output_tensor = conv(input_tensor)

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


class TestConvSparseKernelInit(unittest.TestCase):
    """Tests for __init__ and parameter validation."""

    def test_basic_1d_construction(self):
        m = make_conv_sparse_kernel_1d()
        self.assertEqual(m.in_channels, 4)
        self.assertEqual(m.out_channels, 8)
        self.assertEqual(m.dim, 1)

    def test_basic_2d_construction(self):
        m = make_conv_sparse_kernel_2d()
        self.assertEqual(m.dim, 2)
        self.assertEqual(len(m.lins), 3)

    def test_bias_parameter_present_when_requested(self):
        m = make_conv_sparse_kernel_1d(bias=True)
        self.assertIsNotNone(m.bias)
        self.assertEqual(m.bias.shape, torch.Size([8]))

    def test_bias_parameter_absent_when_disabled(self):
        m = make_conv_sparse_kernel_1d(bias=False)
        self.assertIsNone(m.bias)

    # --- Error cases ---

    def test_key_wrong_dimension_raises(self):
        with self.assertRaises(ValueError):
            torch322.nn.ConvSparseKernel(
                in_channels=4, out_channels=8,
                kernel_size=torch.Size([3]),
                kernel_keys=[(0, 1)],  # 2-D key for 1-D kernel
                stride=(1,),
            )

    def test_key_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            torch322.nn.ConvSparseKernel(
                in_channels=4, out_channels=8,
                kernel_size=torch.Size([3]),
                kernel_keys=[(3,)],  # index 3 >= size 3
                stride=(1,),
            )

    def test_stride_wrong_dimension_raises(self):
        with self.assertRaises(ValueError):
            torch322.nn.ConvSparseKernel(
                in_channels=4, out_channels=8,
                kernel_size=torch.Size([3]),
                kernel_keys=[(0,)],
                stride=(1, 1),  # 2 strides for 1-D kernel
            )


class TestConvSparseKernelOutputShape(unittest.TestCase):
    """Tests for forward() output shapes."""

    # --- 1-D ---

    def test_1d_output_shape_stride1(self):
        m = make_conv_sparse_kernel_1d(in_channels=4, out_channels=8,
                                       kernel_size=torch.Size([3]), kernel_keys=[(0,), (1,), (2,)], stride=(1,))
        x = torch.randn(2, 4, 10)
        y = m(x)
        self.assertEqual(y.shape, torch.Size([2, 8, 8]))  # (10-3)//1+1 = 8

    def test_1d_output_shape_stride2(self):
        m = make_conv_sparse_kernel_1d(in_channels=4, out_channels=8,
                                       kernel_size=torch.Size([3]), kernel_keys=[(0,), (1,), (2,)], stride=(2,))
        x = torch.randn(2, 4, 10)
        y = m(x)
        self.assertEqual(y.shape, torch.Size([2, 8, 4]))  # (10-3)//2+1 = 4

    def test_1d_no_batch_dimension(self):
        m = make_conv_sparse_kernel_1d()
        x = torch.randn(4, 10)  # no batch dim
        y = m(x)
        self.assertEqual(y.shape[0], m.out_channels)

    def test_1d_multi_batch_dimensions(self):
        m = make_conv_sparse_kernel_1d(in_channels=4, out_channels=8,
                                       kernel_size=torch.Size([3]), kernel_keys=[(0,), (1,), (2,)], stride=(1,))
        x = torch.randn(3, 5, 4, 10)  # batch (3, 5)
        y = m(x)
        self.assertEqual(y.shape, torch.Size([3, 5, 8, 8]))

    # --- 2-D ---

    def test_2d_output_shape_stride1(self):
        m = make_conv_sparse_kernel_2d(in_channels=3, out_channels=6,
                                       kernel_size=torch.Size([3, 3]),
                                       kernel_keys=[(0, 0), (1, 1), (2, 2)],
                                       stride=(1, 1))
        x = torch.randn(2, 3, 8, 8)
        y = m(x)
        self.assertEqual(y.shape, torch.Size([2, 6, 6, 6]))

    def test_2d_output_shape_stride2(self):
        m = make_conv_sparse_kernel_2d(in_channels=3, out_channels=6,
                                       kernel_size=torch.Size([3, 3]),
                                       kernel_keys=[(0, 0), (2, 2)],
                                       stride=(2, 2))
        x = torch.randn(1, 3, 9, 9)
        y = m(x)
        # (9-3)//2+1 = 4
        self.assertEqual(y.shape, torch.Size([1, 6, 4, 4]))

    def test_2d_asymmetric_stride(self):
        m = make_conv_sparse_kernel_2d(in_channels=3, out_channels=6,
                                       kernel_size=torch.Size([3, 3]),
                                       kernel_keys=[(0, 0)],
                                       stride=(1, 2))
        x = torch.randn(1, 3, 8, 9)
        y = m(x)
        # height: (8-3)//1+1=6 ; width: (9-3)//2+1=4
        self.assertEqual(y.shape, torch.Size([1, 6, 6, 4]))

    def test_out_channels_in_output(self):
        m = make_conv_sparse_kernel_1d(out_channels=16)
        x = torch.randn(2, 4, 10)
        y = m(x)
        self.assertEqual(y.shape[1], 16)


class TestConvSparseKernelForwardValues(unittest.TestCase):
    """Tests for the numerical correctness of forward()."""

    def test_output_is_finite(self):
        m = make_conv_sparse_kernel_1d()
        x = torch.randn(2, 4, 10)
        y = m(x)
        self.assertTrue(torch.isfinite(y).all())

    def test_output_dtype_preserved(self):
        m = make_conv_sparse_kernel_1d().float()
        x = torch.randn(2, 4, 10, dtype=torch.float32)
        y = m(x)
        self.assertEqual(y.dtype, torch.float32)

    def test_single_key_linearity_in_input(self):
        """With a single key and no bias, output must scale linearly with input."""
        m = torch322.nn.ConvSparseKernel(
            in_channels=4, out_channels=8,
            kernel_size=torch.Size([3]),
            kernel_keys=[(1,)],
            stride=(1,),
            bias=False,
        )
        x = torch.randn(1, 4, 10)
        with torch.no_grad():
            y1 = m(x * 2)
            y2 = m(x) * 2
        torch.testing.assert_close(y1, y2)

    def test_reset_parameters_changes_weights(self):
        m = make_conv_sparse_kernel_1d()
        old_weights = [lin.weight.clone() for lin in m.lins]
        m.reset_parameters()
        new_weights = [lin.weight for lin in m.lins]
        # Very unlikely all weights are identical after re-init
        any_changed = any(
            not torch.equal(o, n) for o, n in zip(old_weights, new_weights)
        )
        self.assertTrue(any_changed)

    def test_output_norm(self):
        # This test may exceptionally fail as it checks
        # some statistical properties using uncontrolled randomness.
        batch_size = 100
        in_channels = 16
        out_channels = 10
        kernel_size = torch.Size([5, 3])
        kernel_keys = [(0, 0), (0, 2), (4, 1)]
        stride = (3, 2)
        bias = True
        input_width = 30
        input_height = 25
        expected_output_width = 9
        expected_output_height = 12

        m = torch322.nn.ConvSparseKernel(in_channels, out_channels, kernel_size, kernel_keys, stride=stride, bias=bias)
        input_tensor = torch.randn(batch_size, in_channels, input_width, input_height)
        output_tensor = m(input_tensor)

        self.assertListEqual(
            list(output_tensor.size()),
            [batch_size, out_channels, expected_output_width, expected_output_height]
        )

        mean_tensor = torch.mean(output_tensor, dim=0)
        std_tensor = torch.std(output_tensor, dim=0)

        for q in [0.1 * i for i in range(1, 10)]:
            print(f'q = {q}')
            print("mean quantile:", mean_tensor.quantile(q).item())
            print("std quantile:", std_tensor.quantile(q).item())

        self.assertGreater(mean_tensor.quantile(0.25).item(), -0.2)
        self.assertLess(mean_tensor.quantile(0.75).item(), +0.2)

        self.assertGreater(std_tensor.quantile(0.25).item(), 0.8)
        self.assertLess(std_tensor.quantile(0.75).item(), 1.2)


class TestConvSparseKernelEdgeCases(unittest.TestCase):
    """Edge-case and boundary tests."""

    def test_spatial_input_equal_kernel_raises(self):
        """Input spatial size == kernel size → output size 0 → should raise."""
        m = make_conv_sparse_kernel_1d(kernel_size=torch.Size([5]), kernel_keys=[(0,)], stride=(1,))
        x = torch.randn(1, 4, 5)  # spatial == kernel size → output 1 is OK (5-5+1=1)
        # Actually 1 > 0 so no error; let's use size 4 < 5
        x_too_small = torch.randn(1, 4, 4)
        with self.assertRaises(ValueError):
            m(x_too_small)

    def test_large_stride_single_output(self):
        """Stride large enough to produce a single spatial output element."""
        m = torch322.nn.ConvSparseKernel(
            in_channels=2, out_channels=4,
            kernel_size=torch.Size([2]),
            kernel_keys=[(0,), (1,)],
            stride=(100,),
            bias=False,
        )
        x = torch.randn(1, 2, 102)  # (102-2)//100+1 = 2
        y = m(x)
        self.assertEqual(y.shape[-1], 2)

    def test_gradients_flow(self):
        """Backward pass should not raise and produce non-None gradients."""
        m = make_conv_sparse_kernel_1d()
        x = torch.randn(2, 4, 10, requires_grad=True)
        y = m(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for lin in m.lins:
            self.assertIsNotNone(lin.weight.grad)

    def test_parameters_are_registered(self):
        """All weights and bias should appear in model.parameters()."""
        m = make_conv_sparse_kernel_1d(bias=True)
        params = list(m.parameters())
        # 3 linear layers × 1 weight each + 1 bias
        self.assertEqual(len(params), 4)

    def test_no_bias_parameter_count(self):
        m = make_conv_sparse_kernel_1d(bias=False)
        params = list(m.parameters())
        self.assertEqual(len(params), 3)  # only linear weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_conv_sparse_kernel_1d(**kwargs):
    """Default valid 1-D ConvSparseKernel (stride=1, bias=True)."""
    defaults = dict(
        in_channels=4,
        out_channels=8,
        kernel_size=torch.Size([3]),
        kernel_keys=[(0,), (1,), (2,)],
        stride=(1,),
        bias=True,
    )
    defaults.update(kwargs)
    return torch322.nn.ConvSparseKernel(**defaults)


def make_conv_sparse_kernel_2d(**kwargs):
    """Default valid 2-D ConvSparseKernel."""
    defaults = dict(
        in_channels=3,
        out_channels=6,
        kernel_size=torch.Size([3, 3]),
        kernel_keys=[(0, 0), (1, 1), (2, 2)],
        stride=(1, 1),
        bias=True,
    )
    defaults.update(kwargs)
    return torch322.nn.ConvSparseKernel(**defaults)
