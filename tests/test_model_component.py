import math
from unittest import TestCase

import numpy as np
import xarray as xr

from ocpy.oc import ModelComponent, Linear, Quadratic, Sinusoidal, Parameter


class TestSetMath(TestCase):
    def test_set_math_numpy(self):
        s = Sinusoidal(amp=1.0, P=2.0)
        result = s.set_math(np)
        self.assertIs(s.math_class, np)
        self.assertIs(result, s)

    def test_set_math_python_math(self):
        s = Sinusoidal(amp=1.0, P=2.0)
        s.set_math(math)
        self.assertIs(s.math_class, math)

    def test_set_math_preserves_model_func(self):
        s = Sinusoidal(amp=1.0, P=2.0)
        s.set_math(math)
        y = s.model_func(0.5, 1.0, 2.0)
        expected = math.sin(2 * math.pi * 0.5 / 2.0)
        self.assertAlmostEqual(y, expected, places=7)

    def test_set_math_returns_self(self):
        lin = Linear(a=1.0, b=0.0)
        result = lin.set_math(np)
        self.assertIs(result, lin)


class TestUpdateParameters(TestCase):
    def test_update_existing_params(self):
        lin = Linear(a=1.0, b=2.0)
        lin.update_parameters({"a": 5.0, "b": 10.0})
        self.assertEqual(lin.params["a"].value, 5.0)
        self.assertEqual(lin.params["b"].value, 10.0)

    def test_update_partial(self):
        lin = Linear(a=1.0, b=2.0)
        lin.update_parameters({"a": 99.0})
        self.assertEqual(lin.params["a"].value, 99.0)
        self.assertEqual(lin.params["b"].value, 2.0)

    def test_update_unknown_key_ignored(self):
        lin = Linear(a=1.0, b=2.0)
        lin.update_parameters({"nonexistent": 42.0})
        self.assertEqual(lin.params["a"].value, 1.0)
        self.assertEqual(lin.params["b"].value, 2.0)

    def test_update_returns_self(self):
        lin = Linear(a=1.0, b=2.0)
        result = lin.update_parameters({"a": 3.0})
        self.assertIs(result, lin)

    def test_update_quadratic(self):
        q = Quadratic(q=1.0)
        q.update_parameters({"q": 0.5})
        self.assertEqual(q.params["q"].value, 0.5)


class TestUpdateFromIdata(TestCase):
    def _make_idata(self, prefix, param_values):
        data_vars = {}
        for name, val in param_values.items():
            full_name = f"{prefix}_{name}"
            data = np.full((2, 100), val)
            data_vars[full_name] = xr.DataArray(
                data, dims=["chain", "draw"]
            )
        ds = xr.Dataset(data_vars)

        class FakeIdata:
            def __getitem__(self, key):
                return ds

        return FakeIdata()

    def test_update_from_idata_median(self):
        lin = Linear(a=0.0, b=0.0)
        idata = self._make_idata("linear", {"a": 5.0, "b": 3.0})
        lin.update_from_idata(idata, stat="median")
        self.assertAlmostEqual(lin.params["a"].value, 5.0)
        self.assertAlmostEqual(lin.params["b"].value, 3.0)

    def test_update_from_idata_mean(self):
        lin = Linear(a=0.0, b=0.0)
        idata = self._make_idata("linear", {"a": 7.0, "b": 2.0})
        lin.update_from_idata(idata, stat="mean")
        self.assertAlmostEqual(lin.params["a"].value, 7.0)
        self.assertAlmostEqual(lin.params["b"].value, 2.0)

    def test_update_from_idata_returns_self(self):
        lin = Linear(a=0.0, b=0.0)
        idata = self._make_idata("linear", {"a": 1.0, "b": 1.0})
        result = lin.update_from_idata(idata)
        self.assertIs(result, lin)

    def test_update_from_idata_ignores_unrelated(self):
        lin = Linear(a=1.0, b=2.0)
        idata = self._make_idata("sinusoidal", {"amp": 5.0, "P": 10.0})
        lin.update_from_idata(idata)
        self.assertEqual(lin.params["a"].value, 1.0)
        self.assertEqual(lin.params["b"].value, 2.0)


class TestModelFunction(TestCase):
    def test_model_function_returns_callable(self):
        lin = Linear(a=1.0, b=0.0)
        func = lin.model_function()
        self.assertTrue(callable(func))

    def test_model_function_is_model_func(self):
        lin = Linear(a=1.0, b=0.0)
        func = lin.model_function()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(func(x, 2.0, 1.0), lin.model_func(x, 2.0, 1.0))
