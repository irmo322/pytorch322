from torch322.nn.utils.module_ext import call_method, call_method_recursive

import torch

import unittest

class TestModuleExt(unittest.TestCase):

    def test_call_method(self):
        module_a = SuperModuleA([])

        with self.assertRaises(RuntimeError):
            call_method(module_a, "toto", True)
        call_method(module_a, "toto", False)

        call_method(module_a, "plop", True, 2)
        self.assertEqual(module_a.a, 12)

        call_method(module_a, "plop", False)
        self.assertEqual(module_a.a, 13)

    def test_call_method_rec(self):

        module = SuperModuleA([
            SuperModuleA([
                SuperModuleA([]),
                SuperModuleB([]),
            ]),
            SuperModuleB([
                SuperModuleA([]),
                SuperModuleB([]),
            ]),
        ])

        call_method_recursive(module, "plop", False, 2)

        self.assertEqual(module.a, 12)
        self.assertEqual(module.sub_modules[0].a, 12)
        self.assertEqual(module.sub_modules[0].sub_modules[0].a, 12)
        self.assertEqual(module.sub_modules[1].sub_modules[0].a, 12)


class SuperModuleA(torch.nn.Module):
    def __init__(self, sub_modules):
        super().__init__()
        self.sub_modules = torch.nn.ModuleList(sub_modules)
        self.a = 10

    def plop(self, inc=1):
        self.a = self.a + inc

class SuperModuleB(torch.nn.Module):
    def __init__(self, sub_modules):
        super().__init__()
        self.sub_modules = torch.nn.ModuleList(sub_modules)
