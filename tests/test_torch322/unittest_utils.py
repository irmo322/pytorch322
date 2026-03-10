from pathlib import Path


def myAssertAlmostEqual(test_case, ref, value, relative_delta=None, **kwargs):
    t = type(ref)
    n = t.__name__
    if t == int or t == float:
        if relative_delta is not None:
            kwargs["delta"] = abs(ref) * relative_delta
        test_case.assertAlmostEqual(ref, value, **kwargs)
    elif t == list or t == tuple:
        test_case.assertEqual(len(ref), len(value))
        for ref_el, value_el in zip(ref, value):
            myAssertAlmostEqual(test_case, ref_el, value_el, relative_delta=relative_delta, **kwargs)
    elif t == dict:
        myAssertAlmostEqual(test_case, list(ref.keys()), list(value.keys()))
        for key in ref:
            myAssertAlmostEqual(test_case, ref[key], value[key], relative_delta=relative_delta, **kwargs)
    elif n in ['ndarray', 'Tensor']:
        myAssertAlmostEqual(test_case, ref.tolist(), value.tolist(), relative_delta=relative_delta, **kwargs)
    else:
        test_case.assertEqual(ref, value)
