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

    def test_star_captures_middle_dims(self):
        # *middle captures (2, 3)
        t = torch.zeros(4, 2, 3, 5)
        values = torch322.utils.check_tensor_size(t, [4, "*middle", 5])
        self.assertListEqual(list(values["*middle"]), [2, 3])

    def test_star_captures_zero_dims(self):
        # *x captures empty tuple
        t = torch.zeros(4, 5)
        values = torch322.utils.check_tensor_size(t, [4, "*x", 5])
        self.assertListEqual(list(values["*x"]), [])

    def test_star_captures_all_dims(self):
        # *all captures all
        t = torch.zeros(2, 3, 4)
        values = torch322.utils.check_tensor_size(t, ["*all"])
        self.assertListEqual(list(values["*all"]), [2, 3, 4])

    def test_star_at_start(self):
        t = torch.zeros(2, 3, 4)
        values = torch322.utils.check_tensor_size(t, ["*start", 4])
        self.assertListEqual(list(values["*start"]), [2, 3])

    def test_star_at_end(self):
        t = torch.zeros(2, 3, 4)
        values = torch322.utils.check_tensor_size(t, [2, "*end"])
        self.assertListEqual(list(values["*end"]), [3, 4])

    def test_star_consistent_across_tensors(self):
        t1 = torch.zeros(2, 3, 5)
        t2 = torch.zeros(2, 3, 7)
        values = torch322.utils.check_tensor_size(t1, ["*batch", 5])
        torch322.utils.check_tensor_size(t2, ["*batch", 7], values=values)
        self.assertListEqual(list(values["*batch"]), [2, 3])

    def test_star_with_other_named_keys(self):
        t = torch.zeros(4, 2, 3, 5)
        values = torch322.utils.check_tensor_size(t, ["n", "*mid", "d"])
        self.assertEqual(values["n"], 4)
        self.assertEqual(values["d"], 5)
        self.assertListEqual(list(values["*mid"]), [2, 3])

    def test_star_inconsistent_across_tensors(self):
        t1 = torch.zeros(2, 3, 5)
        t2 = torch.zeros(9, 9, 7)  # *batch != (2, 3)
        values = torch322.utils.check_tensor_size(t1, ["*batch", 5])
        with self.assertRaises(torch322.utils.TensorSizeError):
            torch322.utils.check_tensor_size(t2, ["*batch", 7], values=values)

    def test_multiple_stars_raises(self):
        t = torch.zeros(2, 3, 4)
        with self.assertRaises(ValueError):
            torch322.utils.check_tensor_size(t, ["*a", "*b"])

    def test_star_too_few_dims_raises(self):
        t = torch.zeros(2, 3)
        with self.assertRaises(torch322.utils.TensorSizeError):
            torch322.utils.check_tensor_size(t, [2, "*x", 3, 4])
