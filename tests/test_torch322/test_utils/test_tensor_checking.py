import torch

import torch322

import unittest


class TestTensorChecking(unittest.TestCase):

    def test_check_tensor_sizes(self):
        tensors_a = [
            torch.zeros(3, 4, 5),
            torch.zeros(3, 10),
            torch.zeros(3)
        ]
        tensors_b = [
            torch.zeros(3, 4, 5),
            torch.zeros(4, 10),
            torch.zeros(3)
        ]
        true_dimensions_a = [("b", 4, 5), ("b", 10), ("b",)]
        bad_dimensions_a_list = [
            [("b", 4, 6), ("b", 10), ("b",)],
            [("b", 5, 5), ("b", 10), ("b",)],
            [("b", 4, 5), ("b", 9), ("b",)],
            [("b", 4, 5), ("b", 10), ("b", 1)],
            [("b", 4, 5), ("b", 10, 1), ("b",)],
            [("b", 4, 5, 1), ("b", 10), ("b",)],
            [("b", 4,), ("b", 10), ("b",)],
            [("b", 4, 5), ("b",), ("b",)],
        ]
        true_dimensions_b = [("b", 4, -1), ("c", None), ("b",)]

        result, _ = torch322.utils.check_tensor_sizes(tensors_a, true_dimensions_a)
        self.assertTrue(result)

        result, _ = torch322.utils.check_tensor_sizes(tensors_b, true_dimensions_b)
        self.assertTrue(result)

        result, _ = torch322.utils.check_tensor_sizes(tensors_b, true_dimensions_a)
        self.assertFalse(result)
        with self.assertRaises(torch322.utils.TensorSizeError):
            torch322.utils.check_tensor_sizes(tensors_b, true_dimensions_a, raise_exception=True)

        for bad_dimensions in bad_dimensions_a_list:
            result, _ = torch322.utils.check_tensor_sizes(tensors_a, bad_dimensions)
            # print(result, tensors_a, bad_dimensions)
            self.assertFalse(result)
            with self.assertRaises(torch322.utils.TensorSizeError):
                torch322.utils.check_tensor_sizes(tensors_a, bad_dimensions, raise_exception=True)

        result, _ = torch322.utils.check_tensor_sizes(tensors_a, true_dimensions_a[:-1])
        self.assertFalse(result)
        with self.assertRaises(ValueError):
            torch322.utils.check_tensor_sizes(tensors_a, true_dimensions_a[:-1], raise_exception=True)
