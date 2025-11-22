import math
import numpy as np
from random import random, randint
from unittest import TestCase

from ocpy.oc import Sinusoidal
from ocpy.oc import Parameter
from tests.utils import maybe_none, N


class TestSinusoidal(TestCase):

    def test_initialize_defaults(self):
        s = Sinusoidal()

        self.assertEqual(s.name, "sinusoidal")

        self.assertIn("amp", s.params)
        self.assertIn("P", s.params)

        self.assertIsInstance(s.params["amp"], Parameter)
        self.assertIsInstance(s.params["P"], Parameter)

        self.assertIsNone(s.params["amp"].value)
        self.assertIsNone(s.params["P"].value)

    def test_initialize_custom_values(self):
        amp = random()
        P = random() + 0.1

        s = Sinusoidal(amp=amp, P=P)

        self.assertEqual(s.params["amp"].value, float(amp))
        self.assertEqual(s.params["P"].value, float(P))

    def test_initialize_with_parameters(self):
        amp_param = Parameter(value=1.0)
        P_param = Parameter(value=5.0)

        s = Sinusoidal(amp=amp_param, P=P_param)

        self.assertIs(s.params["amp"], amp_param)
        self.assertIs(s.params["P"], P_param)

    def test_initialize_with_none(self):
        s = Sinusoidal(amp=None, P=None)

        self.assertIsNone(s.params["amp"].value)
        self.assertIsNone(s.params["P"].value)

    def test_override_name(self):
        s = Sinusoidal(name="my_sin")
        self.assertEqual(s.name, "my_sin")

    def test_model_function(self):
        amp = 3.0
        P = 2.0
        x = 0.5

        s = Sinusoidal(amp=amp, P=P)

        y = s.model_func(x, amp, P)
        expected = amp * np.sin(2 * np.pi * x / P)

        self.assertAlmostEqual(y, expected, places=7)

    def test_math_backend(self):
        s = Sinusoidal(amp=1.0, P=2.0)

        s.set_math(math)

        x = 0.25
        y = s.model_func(x, 1.0, 2.0)
        expected = math.sin(2 * math.pi * x / 2.0)

        self.assertAlmostEqual(y, expected, places=7)

    def test_randomized(self):
        for _ in range(N):
            amp_val = maybe_none(random())
            P_val = maybe_none(random() + 0.1)

            if randint(0, 1) and amp_val is not None:
                amp_val = Parameter(value=amp_val)
            if randint(0, 1) and P_val is not None:
                P_val = Parameter(value=P_val)

            s = Sinusoidal(amp=amp_val, P=P_val)

            if isinstance(amp_val, Parameter):
                self.assertIs(s.params["amp"], amp_val)
            else:
                self.assertEqual(
                    s.params["amp"].value,
                    None if amp_val is None else float(amp_val)
                )

            if isinstance(P_val, Parameter):
                self.assertIs(s.params["P"], P_val)
            else:
                self.assertEqual(
                    s.params["P"].value,
                    None if P_val is None else float(P_val)
                )
