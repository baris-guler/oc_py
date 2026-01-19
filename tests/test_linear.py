from random import random, randint
from unittest import TestCase

from ocpy.oc import Linear
from ocpy.oc import Parameter
from tests.utils import maybe_none, N


class TestLinear(TestCase):
    def test_initialize_defaults(self):
        linear = Linear()
        self.assertEqual(linear.name, "linear")

        self.assertIsInstance(linear.params["a"], Parameter)
        self.assertIsInstance(linear.params["b"], Parameter)

        self.assertEqual(linear.params["a"].value, 1.0)
        self.assertEqual(linear.params["b"].value, 0.0)

    def test_initialize_custom_values(self):
        a = random()
        b = random()

        linear = Linear(a=a, b=b)

        self.assertEqual(linear.params["a"].value, float(a))
        self.assertEqual(linear.params["b"].value, float(b))

    def test_initialize_with_parameters(self):
        a = Parameter(value=5.0)
        b = Parameter(value=-2.0)

        linear = Linear(a=a, b=b)

        self.assertIs(linear.params["a"], a)
        self.assertIs(linear.params["b"], b)

    def test_initialize_with_none(self):
        linear = Linear(a=None, b=None)

        self.assertIsInstance(linear.params["a"], Parameter)
        self.assertIsInstance(linear.params["b"], Parameter)
        self.assertIsNone(linear.params["a"].value)
        self.assertIsNone(linear.params["b"].value)

    def test_override_name(self):
        linear = Linear(name="my_linear")
        self.assertEqual(linear.name, "my_linear")

    def test_model_function(self):
        linear = Linear(a=2.0, b=3.0)

        y = linear.model_func(5, 2.0, 3.0)
        self.assertEqual(y, 13.0)

    def test_full_randomized(self):
        for _ in range(N):
            a_val = maybe_none(random())
            b_val = maybe_none(random())

            if randint(0, 1) and a_val is not None:
                a_val = Parameter(value=a_val)
            if randint(0, 1) and b_val is not None:
                b_val = Parameter(value=b_val)

            linear = Linear(a=a_val, b=b_val)

            if isinstance(a_val, Parameter):
                self.assertIs(linear.params["a"], a_val)
            else:
                self.assertEqual(
                    linear.params["a"].value,
                    None if a_val is None else float(a_val)
                )

            if isinstance(b_val, Parameter):
                self.assertIs(linear.params["b"], b_val)
            else:
                self.assertEqual(
                    linear.params["b"].value,
                    None if b_val is None else float(b_val)
                )
