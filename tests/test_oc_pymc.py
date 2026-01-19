import unittest
import numpy as np
import pymc as pm
from ocpy.oc_pymc import OCPyMC
from ocpy.oc import Parameter, Linear

class TestOCPyMC(unittest.TestCase):
    def setUp(self):
        self.cycle = np.linspace(0, 100, 20)
        self.oc = 0.5 * self.cycle + 10 + np.random.normal(0, 0.1, 20)
        self.err = np.ones_like(self.cycle) * 0.1
        
        self.oc_pymc = OCPyMC(
            cycle=self.cycle,
            oc=self.oc,
            minimum_time=self.cycle, 
            minimum_time_error=self.err,
            weights=np.ones_like(self.cycle)
        )

    def test_initialization(self):
        self.assertIsInstance(self.oc_pymc, OCPyMC)
        self.assertTrue("cycle" in self.oc_pymc.data.columns)

    def test_fit_linear_runs(self):
        idata = self.oc_pymc.fit_linear(
            a=Parameter(value=0.5, min=0.4, max=0.6),
            b=Parameter(value=10, min=9, max=11),
            draws=2,
            tune=2,
            chains=1,
            progressbar=False,
            random_seed=42
        )
        self.assertIsNotNone(idata)
        self.assertTrue(hasattr(idata, "posterior"))
        self.assertIn("linear_a", idata.posterior.data_vars)

    def test_fit_generic_runs(self):
        lin = Linear(
            a=Parameter(value=0.5, min=0.4, max=0.6),
            b=Parameter(value=10, min=9, max=11)
        )
        idata = self.oc_pymc.fit(
            model_components=[lin],
            draws=2,
            tune=2,
            chains=1,
            progressbar=False,
            random_seed=42
        )
        self.assertIsNotNone(idata)
        self.assertTrue(hasattr(idata, "posterior"))
        self.assertIn("linear_a", idata.posterior.data_vars)

    def test_fit_keplerian_runs(self):
        idata = self.oc_pymc.fit_keplerian(
            P=Parameter(value=50, fixed=True),
            e=Parameter(value=0.1, fixed=True),
            draws=2,
            tune=2,
            chains=1,
            progressbar=False,
            random_seed=42
        )
        self.assertIsNotNone(idata)
        self.assertIn("keplerian1_amp", idata.posterior.data_vars)

    def test_residue(self):
        idata = self.oc_pymc.fit_linear(
            draws=2, tune=2, chains=1, progressbar=False
        )
        oc_res = self.oc_pymc.residue(idata)
        self.assertIsInstance(oc_res, OCPyMC)
        self.assertEqual(len(oc_res.data), len(self.oc_pymc.data))
        self.assertTrue(np.all(np.isfinite(oc_res.data["oc"])))

    def test_fit_lite_runs(self):
        idata = self.oc_pymc.fit_lite(
            P=Parameter(value=50, fixed=True),
            e=Parameter(value=0.1, fixed=True),
            draws=2,
            tune=2,
            chains=1,
            progressbar=False,
            random_seed=42
        )
        self.assertIsNotNone(idata)
        self.assertIn("keplerian1_amp", idata.posterior.data_vars)
