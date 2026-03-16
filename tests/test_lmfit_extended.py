from random import randint
from unittest import TestCase

import numpy as np

from ocpy.oc import Parameter, Linear, Sinusoidal
from ocpy.oc_lmfit import OCLMFit
from tests.utils import N


class TestOCLMFitResidue(TestCase):
    def _make_data(self, n=100):
        cycle = np.linspace(-500, 500, n)
        oc = np.sin(np.deg2rad(cycle))
        return OCLMFit(oc=oc.tolist(), cycle=cycle.tolist(), weights=np.ones(n).tolist())

    def test_residue_returns_oclmfit(self):
        data = self._make_data()
        result = data.fit_sinusoidal(P=300, amp=0.6)
        residual = data.residue(result)
        self.assertIsInstance(residual, OCLMFit)

    def test_residue_length(self):
        data = self._make_data()
        result = data.fit_sinusoidal(P=300, amp=0.6)
        residual = data.residue(result)
        self.assertEqual(len(residual), len(data))

    def test_residue_smaller_than_original(self):
        data = self._make_data()
        result = data.fit_sinusoidal(P=300, amp=0.6)
        residual = data.residue(result)

        original_rms = np.sqrt(np.mean(np.array(data["oc"]) ** 2))
        residual_rms = np.sqrt(np.mean(np.array(residual["oc"]) ** 2))
        self.assertLess(residual_rms, original_rms)

    def test_residue_preserves_cycle(self):
        data = self._make_data()
        result = data.fit_sinusoidal(P=300, amp=0.6)
        residual = data.residue(result)
        np.testing.assert_array_almost_equal(
            residual["cycle"].to_numpy(),
            data["cycle"].to_numpy(),
        )

    def test_residue_preserves_weights(self):
        data = self._make_data()
        result = data.fit_sinusoidal(P=300, amp=0.6)
        residual = data.residue(result)
        np.testing.assert_array_almost_equal(
            residual["weights"].to_numpy(),
            data["weights"].to_numpy(),
        )

    def test_residue_after_linear_fit(self):
        n = 100
        cycle = np.linspace(-100, 100, n)
        oc = 2.0 * cycle + 5.0
        data = OCLMFit(oc=oc.tolist(), cycle=cycle.tolist(), weights=np.ones(n).tolist())

        result = data.fit_linear(a=1.5, b=3.0)
        residual = data.residue(result)

        residual_oc = residual["oc"].to_numpy()
        self.assertAlmostEqual(np.mean(np.abs(residual_oc)), 0.0, places=3)

    def test_residue_randomized(self):
        for _ in range(N):
            n = randint(30, 200)
            data = self._make_data(n)
            result = data.fit_sinusoidal(P=300, amp=0.6)
            residual = data.residue(result)
            self.assertEqual(len(residual), n)
