from random import random, randint
from unittest import TestCase

from ocpy.oc import Quadratic
from ocpy.oc import Parameter
from tests.utils import maybe_none, N


class TestQuadratic(TestCase):

    def test_initialize_defaults(self):
        quad = Quadratic()

        self.assertEqual(quad.name, "quadratic")
        self.assertIn("q", quad.params)
        self.assertIsInstance(quad.params["q"], Parameter)
        self.assertEqual(quad.params["q"].value, 0.0)

    def test_initialize_custom_value(self):
        q = random()
        quad = Quadratic(q=q)

        self.assertEqual(quad.params["q"].value, float(q))

    def test_initialize_with_parameter(self):
        q_param = Parameter(value=4.0)
        quad = Quadratic(q=q_param)

        self.assertIs(quad.params["q"], q_param)

    def test_initialize_with_none(self):
        quad = Quadratic(q=None)

        self.assertIsInstance(quad.params["q"], Parameter)
        self.assertIsNone(quad.params["q"].value)

    def test_override_name(self):
        quad = Quadratic(name="my_quad")
        self.assertEqual(quad.name, "my_quad")

    def test_model_function(self):
        quad = Quadratic(q=3.0)

        result = quad.model_func(5, 3.0)  # q * x^2 = 3 * 25
        self.assertEqual(result, 75.0)

    def test_randomized(self):
        for _ in range(N):
            q_val = maybe_none(random())

            if randint(0, 1) and q_val is not None:
                q_val = Parameter(value=q_val)

            quad = Quadratic(q=q_val)

            if isinstance(q_val, Parameter):
                self.assertIs(quad.params["q"], q_val)
            else:
                self.assertEqual(
                    quad.params["q"].value,
                    None if q_val is None else float(q_val)
                )
