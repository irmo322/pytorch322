import torch

import torch322

import unittest


class TestTensorChecking(unittest.TestCase):

    def test_check_tensor_size(self):
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

        torch322.utils.check_tensor_size(tensors_a, true_dimensions_a)
        torch322.utils.check_tensor_size(tensors_b, true_dimensions_b)

        with self.assertRaises(torch322.utils.TensorSizeError):
            torch322.utils.check_tensor_size(tensors_b, true_dimensions_a)

        for bad_dimensions in bad_dimensions_a_list:
            with self.assertRaises(torch322.utils.TensorSizeError):
                torch322.utils.check_tensor_size(tensors_a, bad_dimensions)

        with self.assertRaises(torch322.utils.TensorSizeError):
            torch322.utils.check_tensor_size(tensors_a, true_dimensions_a[:-1])

    def test_tensor_size_checker(self):
        tsc = torch322.utils.TensorSizeChecker(values={"a": 10})

        tsc.check(torch.zeros(5, 10, 15), [5, "a", -1])
        tsc.check(torch.zeros(20), ["b"])
        tsc.check(torch.Size((10, 20)), ["a", "b"])

        with self.assertRaises(torch322.utils.TensorSizeError):
            tsc.check(torch.zeros(10, 21), ["a", "b"])

        tsc.check(
            {
                "tensor1": torch.Size((10, 20)),
                "tensor2": torch.Size((20, 30)),
            },
            {
                "tensor1": [10, 20],
                "tensor2": [20, 30],
            },
        )

        with self.assertRaises(torch322.utils.TensorSizeError):
            tsc.check(
                {
                    "tensor1": torch.Size((10, 20)),
                    "tensor2": torch.Size((20, 30)),
                },
                {
                    "tensor1": [10, 20],
                    "tensor3": [20, 30],
                },
            )
