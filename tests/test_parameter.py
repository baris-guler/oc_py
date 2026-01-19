from random import random, randint, choices
from string import ascii_letters
from unittest import TestCase

from ocpy.oc import Parameter
from tests.utils import maybe_none, N


class TestParameter(TestCase):

    def test_ini_without_data(self):
        parameter = Parameter()
        self.assertIsNone(parameter.value)
        self.assertIsNone(parameter.min)
        self.assertIsNone(parameter.max)
        self.assertIsNone(parameter.std)
        self.assertFalse(parameter.fixed)
        self.assertEqual(parameter.distribution, "truncatednormal")

    def test_all_data(self):
        value = random()
        min_value = random()
        max_value = random()
        std = random()
        fixed = bool(randint(0, 1))
        distribution = "".join(choices(ascii_letters, k=10))

        parameter = Parameter(
            value=value,
            min=min_value,
            max=max_value,
            std=std,
            fixed=fixed,
            distribution=distribution,
        )

        self.assertEqual(parameter.value, value)
        self.assertEqual(parameter.min, min_value)
        self.assertEqual(parameter.max, max_value)
        self.assertEqual(parameter.std, std)
        self.assertEqual(parameter.fixed, fixed)
        self.assertEqual(parameter.distribution, distribution)

    def test_randomize(self):
        for _ in range(N):
            value = maybe_none(random())
            min_value = maybe_none(random())
            max_value = maybe_none(random())
            std = maybe_none(random())
            fixed = maybe_none(bool(randint(0, 1)))
            distribution = maybe_none("".join(choices(ascii_letters, k=10)))

            parameter = Parameter(
                value=value,
                min=min_value,
                max=max_value,
                std=std,
                fixed=fixed,
                distribution=distribution,
            )

            self.assertEqual(parameter.value, value)
            self.assertEqual(parameter.min, min_value)
            self.assertEqual(parameter.max, max_value)
            self.assertEqual(parameter.std, std)
            self.assertEqual(parameter.fixed, fixed)
            self.assertEqual(parameter.distribution, distribution)
