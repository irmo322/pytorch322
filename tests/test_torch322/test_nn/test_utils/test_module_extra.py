import torch

import torch322

import unittest

class TestModuleExtra(unittest.TestCase):

    def test_call_module_method(self):
        module_a = SuperModuleA([])

        with self.assertRaises(RuntimeError):
            torch322.nn.utils.call_module_method(module_a, "toto", required=True)
        torch322.nn.utils.call_module_method(module_a, "toto", required=False)

        torch322.nn.utils.call_module_method(module_a, "plop", required=True, args=[2])
        self.assertEqual(module_a.a, 12)

        torch322.nn.utils.call_module_method(module_a, "plop", required=False)
        self.assertEqual(module_a.a, 13)

    def test_call_module_method_recursive(self):

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

        torch322.nn.utils.call_module_method_recursive(module, "plop", required=False, kwargs={'inc':2})

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
