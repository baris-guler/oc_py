from random import uniform, randint
from unittest import TestCase

import numpy as np

from ocpy.oc import Parameter, Linear, Quadratic, Keplerian
from ocpy.oc_lmfit import _ensure_param, OCLMFit
from tests.utils import N


class TestOCLMFit(TestCase):
    def test_ensure_param_returns_same_if_parameter(self):
        value = uniform(-1000, 1000)
        p = Parameter(value=value)
        default_value = uniform(-1000, 1000)
        result = _ensure_param(p, default=Parameter(value=default_value))
        assert result is p  # should return the same object

    def test_ensure_param_returns_default_if_none(self):
        default_value = uniform(-1000, 1000)
        default_param = Parameter(value=default_value)
        result = _ensure_param(None, default=default_param)
        assert result is default_param  # should return default object

    def test_ensure_param_creates_new_parameter_if_value(self):
        value = uniform(-1000, 1000)
        default_value = uniform(-1000, 1000)
        default_param = Parameter(value=default_value)
        result = _ensure_param(value, default=default_param)
        assert isinstance(result, Parameter)
        assert result.value == value
        assert result is not default_param  # must be a new object

    def test_ensure_param_with_zero_value(self):
        default_value = uniform(-1000, 1000)
        default_param = Parameter(value=default_value)
        result = _ensure_param(0, default=default_param)
        assert isinstance(result, Parameter)
        assert result.value == 0

    def test_ensure_param_with_float_value(self):
        value = uniform(-1000, 1000)
        default_value = uniform(-1000, 1000)
        default_param = Parameter(value=default_value)
        result = _ensure_param(value, default=default_param)
        assert isinstance(result, Parameter)
        assert result.value == value

    def test_fit_sinusoidal(self):
        for _ in range(100):
            start = randint(-1000, -50)
            end = randint(50, 1000)

            cycle = np.linspace(start, end, 100)
            oc = np.sin(np.deg2rad(cycle))

            oc_lm = OCLMFit(oc=oc, cycle=cycle, weights=1.0)

            res = oc_lm.fit_sinusoidal(P=300, amp=0.6)

            y_fit_at_x = res.eval(x=cycle)
            resid = oc - y_fit_at_x
            self.assertAlmostEqual(resid.mean(), 0, places=1)

    def test_fit_keplerian(self):
        for _ in range(100):
            start = randint(-1000, -1)
            end = randint(1, 1000)

            cycle = np.linspace(start, end, 100)
            oc = 3 * cycle ** 2

            oc_lm = OCLMFit(oc=oc.tolist(), cycle=cycle.tolist(), weights=1)

            res = oc_lm.fit_quadratic()

            y_fit = res.eval(x=cycle)
            resid = oc - y_fit
            self.assertAlmostEqual(resid.mean(), 0, places=1)

    def test_fit_combination(self):
        for _ in range(N):
            start = randint(-5000, -1)
            end = randint(1, 5000)
            t = np.linspace(start, end, 100)

            lin = Linear(
                a=Parameter(value=1.2e-6),
                b=Parameter(value=-3.5e-3),
            )

            quad = Quadratic(
                q=Parameter(value=1e-10),
            )

            kep = Keplerian(
                amp=Parameter(value=8e-4),
                e=Parameter(value=0.25),
                omega=Parameter(value=110.0),
                P=Parameter(value=1200.0),
                T0=Parameter(value=300.0),
            )

            y_lin = lin.model_func(t, lin.params["a"].value, lin.params["b"].value)
            y_quad = quad.model_func(t, quad.params["q"].value)
            y_kep = kep.model_func(
                t,
                kep.params["amp"].value,
                kep.params["e"].value,
                kep.params["omega"].value,
                kep.params["P"].value,
                kep.params["T0"].value,
            )

            oc_true = y_lin + y_quad + y_kep

            noise = np.random.normal(0.0, 1e-4, size=t.size)
            oc_obs = oc_true + noise

            err = np.full_like(t, 1e-4)

            oc_data = OCLMFit(
                oc=oc_obs.tolist(),
                minimum_time=t.tolist(),
                minimum_time_error=err.tolist(),
                weights=np.ones_like(t).tolist(),
                minimum_type=[None] * t.size,
                labels=["sim"] * t.size,
                cycle=t.tolist(),
            )

            lin = Linear(
                a=Parameter(value=1e-6),
                b=Parameter(value=-3e-3),
            )
            quad = Quadratic(
                q=Parameter(value=3e-10),
            )
            kep = Keplerian(
                amp=Parameter(value=6e-4),
                e=Parameter(value=0.3, min=0, max=1),
                omega=Parameter(value=90.0),
                P=Parameter(value=1000.0),
                T0=Parameter(value=0.0),
            )
            res = oc_data.fit([lin, quad, kep])

            y_fit = res.eval(x=oc_obs)
            resid = oc_obs - y_fit
            self.assertAlmostEqual(resid.mean(), 0, places=1)
