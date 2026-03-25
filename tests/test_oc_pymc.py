import unittest
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pymc as pm
import arviz as az
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


class TestOCPyMCRemoveBadSamples(unittest.TestCase):
    def setUp(self):
        self.cycle = np.linspace(0, 100, 20)
        self.oc = 0.5 * self.cycle + 10 + np.random.normal(0, 0.1, 20)
        self.err = np.ones_like(self.cycle) * 0.1

        self.oc_pymc = OCPyMC(
            cycle=self.cycle,
            oc=self.oc,
            minimum_time=self.cycle,
            minimum_time_error=self.err,
            weights=np.ones_like(self.cycle),
        )
        self.idata = self.oc_pymc.fit_linear(
            a=Parameter(value=0.5, min=0.4, max=0.6),
            b=Parameter(value=10, min=9, max=11),
            draws=20,
            tune=10,
            chains=2,
            progressbar=False,
            random_seed=42,
        )

    def test_remove_bad_samples_returns_inference_data(self):
        cleaned = self.oc_pymc.remove_bad_samples(self.idata, verbose=False)
        self.assertIsInstance(cleaned, az.InferenceData)
        self.assertTrue(hasattr(cleaned, "posterior"))

    def test_remove_bad_samples_preserves_attrs(self):
        cleaned = self.oc_pymc.remove_bad_samples(self.idata, verbose=False)
        self.assertIn("_model_components", cleaned.attrs)
        self.assertIn("_model_prefixes", cleaned.attrs)

    def test_remove_bad_samples_filter_outliers(self):
        cleaned = self.oc_pymc.remove_bad_samples(self.idata, filter_outliers=True, verbose=False)
        self.assertTrue(hasattr(cleaned, "posterior"))

    def test_remove_bad_samples_no_divergent_filtering(self):
        cleaned = self.oc_pymc.remove_bad_samples(self.idata, remove_divergent=False, verbose=False)
        self.assertTrue(hasattr(cleaned, "posterior"))

    def test_remove_bad_samples_drop_chains(self):
        cleaned = self.oc_pymc.remove_bad_samples(self.idata, drop_chains=1, verbose=False)
        n_chains = cleaned.posterior.sizes["chain"]
        self.assertEqual(n_chains, 1)

    def test_remove_bad_samples_drop_too_many_chains_raises(self):
        with self.assertRaises(ValueError):
            self.oc_pymc.remove_bad_samples(self.idata, drop_chains=2, verbose=False)

    def test_remove_bad_samples_drop_chains_and_filter(self):
        cleaned = self.oc_pymc.remove_bad_samples(
            self.idata,
            drop_chains=1,
            filter_outliers=True,
            remove_divergent=True,
            verbose=False
        )
        self.assertTrue(hasattr(cleaned, "posterior"))

    def test_remove_bad_samples_with_quality_checks(self):
        cleaned = self.oc_pymc.remove_bad_samples(
            self.idata,
            check_ess=True,
            min_ess=10,
            check_rhat=True,
            max_rhat=1.1,
            verbose=False
        )
        self.assertTrue(hasattr(cleaned, "posterior"))


class TestOCPyMCCornerTrace(unittest.TestCase):
    def setUp(self):
        self.cycle = np.linspace(0, 100, 20)
        self.oc = 0.5 * self.cycle + 10 + np.random.normal(0, 0.1, 20)
        self.err = np.ones_like(self.cycle) * 0.1

        self.oc_pymc = OCPyMC(
            cycle=self.cycle,
            oc=self.oc,
            minimum_time=self.cycle,
            minimum_time_error=self.err,
            weights=np.ones_like(self.cycle),
        )
        self.idata = self.oc_pymc.fit_linear(
            a=Parameter(value=0.5, min=0.4, max=0.6),
            b=Parameter(value=10, min=9, max=11),
            draws=20,
            tune=10,
            chains=2,
            progressbar=False,
            random_seed=42,
        )

    def test_corner_default(self):
        fig = self.oc_pymc.corner(self.idata)
        self.assertIsNotNone(fig)

    def test_corner_arviz_style(self):
        axes = self.oc_pymc.corner(self.idata, cornerstyle="arviz")
        self.assertIsNotNone(axes)

    def test_corner_invalid_style(self):
        with self.assertRaises(ValueError):
            self.oc_pymc.corner(self.idata, cornerstyle="invalid")

    def test_corner_with_units(self):
        fig = self.oc_pymc.corner(self.idata, units={"linear_a": "d/cycle"})
        self.assertIsNotNone(fig)

    def test_trace(self):
        axes = self.oc_pymc.trace(self.idata)
        self.assertIsNotNone(axes)
