import torch322

import random

import unittest


class TestChunkSampler(unittest.TestCase):

    def test_no_shuffle(self):
        # Dataset of 10, chunk of 4
        sampler = torch322.utils.data.ChunkSampler(dataset_size=10, chunk_size=4, shuffle=False)

        assert list(sampler) == [0, 1, 2, 3]   # iteration 1
        assert list(sampler) == [4, 5, 6, 7]   # iteration 2
        assert list(sampler) == [8, 9, 0, 1]   # iteration 3: wrap-around
        assert list(sampler) == [2, 3, 4, 5]   # iteration 4

    def test_shuffle(self):
        random.seed(322)
        sampler = torch322.utils.data.ChunkSampler(dataset_size=5, chunk_size=4, shuffle=True)

        samples = []
        for _ in range(5):
            chunk = list(sampler)
            self.assertEqual(len(chunk), 4)
            samples.extend(chunk)

        for i in range(4):
            self.assertEqual(5, len(set(samples[5*i:5*i+5])))

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            torch322.utils.data.ChunkSampler(dataset_size=0, chunk_size=5)

        with self.assertRaises(ValueError):
            torch322.utils.data.ChunkSampler(dataset_size=10, chunk_size=0)

    def test_reset(self):
        sampler = torch322.utils.data.ChunkSampler(dataset_size=10, chunk_size=3, shuffle=False)

        first_pass = list(sampler)  # [0, 1, 2]
        list(sampler)               # [3, 4, 5]

        sampler.reset()
        assert list(sampler) == first_pass  # [0, 1, 2]
