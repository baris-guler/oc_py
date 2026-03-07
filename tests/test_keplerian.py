import numpy as np
from random import random, randint
from unittest import TestCase

from ocpy.oc import Keplerian
from ocpy.oc import Parameter
from tests.utils import maybe_none


class TestKeplerian(TestCase):

    def test_initialize_defaults(self):
        k = Keplerian()

        self.assertEqual(k.name, "keplerian")

        for p in ("amp", "e", "omega", "P", "T0"):
            self.assertIn(p, k.params)
            self.assertIsInstance(k.params[p], Parameter)

        self.assertIsNone(k.params["amp"].value)
        self.assertEqual(k.params["e"].value, 0.0)
        self.assertEqual(k.params["omega"].value, 0.0)
        self.assertIsNone(k.params["P"].value)
        self.assertIsNone(k.params["T0"].value)

    def test_initialize_custom_values(self):
        values = {
            "amp": random(),
            "e": random() * 0.9,
            "omega": random() * 360,
            "P": random() + 0.1,
            "T0": random(),
        }

        k = Keplerian(**values)

        for key, val in values.items():
            self.assertEqual(k.params[key].value, float(val))

    def test_initialize_with_parameters(self):
        vals = {
            "amp": Parameter(value=1.0),
            "e": Parameter(value=0.1),
            "omega": Parameter(value=45.0),
            "P": Parameter(value=5.0),
            "T0": Parameter(value=0.2),
        }

        k = Keplerian(**vals)

        for key, param in vals.items():
            self.assertIs(k.params[key], param)

    def test_initialize_with_none(self):
        k = Keplerian(amp=None, P=None, T0=None)

        self.assertIsNone(k.params["amp"].value)
        self.assertIsNone(k.params["P"].value)
        self.assertIsNone(k.params["T0"].value)

    def test_override_name(self):
        k = Keplerian(name="my_kepler")
        self.assertEqual(k.name, "my_kepler")

    def test_kepler_solve_basic(self):
        k = Keplerian()
        e = 0.1

        for _ in range(10):
            M = random() * 2 * np.pi - np.pi
            E = k._kepler_solve(M, e)
            lhs = E - e * np.sin(E)
            self.assertAlmostEqual(lhs, M, places=6)

    def test_model_function_simple(self):
        amp = 10.0
        e = 0.0
        omega = 0.0
        P = 2.0
        T0 = 0.0

        k = Keplerian(amp=amp, e=e, omega=omega, P=P, T0=T0)

        x = 0.25
        result = k.model_func(x, amp, e, omega, P, T0)
        expected = amp * np.sin(2 * np.pi * x / P)

        self.assertAlmostEqual(result, expected, places=6)

    def test_model_function_range(self):
        k = Keplerian(
            amp=5.0,
            e=0.5,
            omega=30.0,
            P=3.0,
            T0=0.25,
        )

        for x in np.linspace(0, 3, 20):
            y = k.model_func(x, 5.0, 0.5, 30.0, 3.0, 0.25)
            self.assertFalse(np.isnan(y))
            self.assertFalse(np.isinf(y))

    def test_randomized(self):
        for _ in range(30):

            vals = {
                "amp": maybe_none(random() * 5),
                "e": maybe_none(random() * 0.9),
                "omega": maybe_none(random() * 360),
                "P": maybe_none(random() + 0.1),
                "T0": maybe_none(random()),
            }

            for k2 in vals:
                if randint(0, 1) and vals[k2] is not None:
                    vals[k2] = Parameter(value=vals[k2])

            k = Keplerian(**vals)

            for key, orig in vals.items():
                if isinstance(orig, Parameter):
                    self.assertIs(k.params[key], orig)
                else:
                    expected = None if orig is None else float(orig)
                    self.assertEqual(k.params[key].value, expected)
