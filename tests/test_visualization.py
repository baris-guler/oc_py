import unittest
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from ocpy.visualization import Plot
from ocpy.oc import OC, Linear, Quadratic, Sinusoidal, Keplerian, Parameter
from ocpy.oc_lmfit import OCLMFit
from ocpy.oc_pymc import OCPyMC


class TestPlotData(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_data_with_labels(self):
        oc = OC(
            cycle=np.arange(10).tolist(),
            oc=np.sin(np.arange(10)).tolist(),
            minimum_time_error=[0.1] * 10,
            minimum_time=np.arange(10).tolist(),
            weights=[1.0] * 10,
            labels=["A"] * 5 + ["B"] * 5,
        )
        ax = Plot.plot_data(oc)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_data_no_labels(self):
        oc = OC(
            cycle=np.arange(10).tolist(),
            oc=np.sin(np.arange(10)).tolist(),
            minimum_time_error=[0.1] * 10,
            minimum_time=np.arange(10).tolist(),
            weights=[1.0] * 10,
        )
        ax = Plot.plot_data(oc)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_data_no_error(self):
        oc = OC(
            cycle=np.arange(10).tolist(),
            oc=np.sin(np.arange(10)).tolist(),
            minimum_time=np.arange(10).tolist(),
            weights=[1.0] * 10,
        )
        ax = Plot.plot_data(oc)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_data_with_existing_ax(self):
        fig, ax = plt.subplots()
        oc = OC(
            cycle=np.arange(10).tolist(),
            oc=np.sin(np.arange(10)).tolist(),
            minimum_time=np.arange(10).tolist(),
            weights=[1.0] * 10,
        )
        result = Plot.plot_data(oc, ax=ax)
        self.assertIs(result, ax)

    def test_plot_data_with_nan_labels(self):
        labels = ["A"] * 5 + [None] * 5
        oc = OC(
            cycle=np.arange(10).tolist(),
            oc=np.sin(np.arange(10)).tolist(),
            minimum_time_error=[0.1] * 10,
            minimum_time=np.arange(10).tolist(),
            weights=[1.0] * 10,
            labels=labels,
        )
        ax = Plot.plot_data(oc)
        self.assertIsInstance(ax, plt.Axes)


class TestPlotModelLmfit(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def setUp(self):
        n = 50
        cycle = np.linspace(-100, 100, n)
        oc = 2.0 * cycle + 5.0 + np.random.normal(0, 0.1, n)
        self.data = OCLMFit(
            oc=oc.tolist(), cycle=cycle.tolist(),
            minimum_time=cycle.tolist(),
            minimum_time_error=[0.1] * n,
            weights=np.ones(n).tolist(),
        )
        self.result = self.data.fit_linear(a=1.5, b=3.0)

    def test_plot_model_lmfit(self):
        ax = Plot.plot_model_lmfit(self.result, self.data)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_model_lmfit_with_ax(self):
        fig, ax = plt.subplots()
        result = Plot.plot_model_lmfit(self.result, self.data, ax=ax)
        self.assertIs(result, ax)


class TestPlotModelComponents(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_single_component(self):
        lin = Linear(a=Parameter(value=2.0), b=Parameter(value=1.0))
        xline = np.linspace(-10, 10, 100)
        ax = Plot.plot_model_components([lin], xline)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_multiple_components(self):
        lin = Linear(a=Parameter(value=1.0), b=Parameter(value=0.0))
        quad = Quadratic(q=Parameter(value=0.01))
        xline = np.linspace(-10, 10, 100)
        ax = Plot.plot_model_components([lin, quad], xline)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_uncertainty_band(self):
        lin = Linear(a=Parameter(value=1.0), b=Parameter(value=0.0))
        xline = np.linspace(-10, 10, 100)
        y = lin.model_func(xline, 1.0, 0.0)
        band = (xline, y - 0.5, y + 0.5)
        ax = Plot.plot_model_components([lin], xline, uncertainty_band=band)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_empty_components(self):
        xline = np.linspace(-10, 10, 100)
        ax = Plot.plot_model_components([], xline)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_existing_ax(self):
        fig, ax = plt.subplots()
        lin = Linear(a=Parameter(value=1.0), b=Parameter(value=0.0))
        xline = np.linspace(-10, 10, 100)
        result = Plot.plot_model_components([lin], xline, ax=ax)
        self.assertIs(result, ax)


class TestPlotWrapper(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def setUp(self):
        n = 50
        self.cycle = np.linspace(-100, 100, n)
        oc = 2.0 * self.cycle + 5.0 + np.random.normal(0, 0.1, n)
        self.data = OCLMFit(
            oc=oc.tolist(), cycle=self.cycle.tolist(),
            minimum_time=self.cycle.tolist(),
            minimum_time_error=[0.1] * n,
            weights=np.ones(n).tolist(),
            labels=["CCD"] * n,
        )

    def test_plot_no_model(self):
        ax = Plot.plot(self.data, res=False)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_lmfit_model(self):
        result = self.data.fit_linear(a=1.5, b=3.0)
        ax = Plot.plot(self.data, model=result, res=True)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_component_list(self):
        lin = Linear(a=Parameter(value=2.0), b=Parameter(value=5.0))
        ax = Plot.plot(self.data, model=[lin], res=True)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_single_component(self):
        lin = Linear(a=Parameter(value=2.0), b=Parameter(value=5.0))
        ax = Plot.plot(self.data, model=lin, res=True)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_title(self):
        ax = Plot.plot(self.data, res=False, title="Test Plot")
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_with_existing_ax_no_res(self):
        fig, ax = plt.subplots()
        result = Plot.plot(self.data, ax=ax, res=True)
        self.assertIsInstance(result, plt.Axes)


class TestPlotModelPyMC(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def setUp(self):
        n = 20
        self.cycle = np.linspace(0, 100, n)
        oc = 0.5 * self.cycle + 10 + np.random.normal(0, 0.1, n)
        self.data = OCPyMC(
            cycle=self.cycle,
            oc=oc,
            minimum_time=self.cycle,
            minimum_time_error=np.ones(n) * 0.1,
            weights=np.ones(n),
        )
        self.idata = self.data.fit_linear(
            a=Parameter(value=0.5, min=0.4, max=0.6),
            b=Parameter(value=10, min=9, max=11),
            draws=10, tune=5, chains=1, progressbar=False, random_seed=42,
        )

    def test_plot_model_pymc(self):
        fig, ax = plt.subplots()
        result = Plot.plot_model_pymc(self.idata, self.data, ax=ax)
        self.assertIsInstance(result, plt.Axes)

    def test_plot_with_pymc_model(self):
        ax = Plot.plot(self.data, model=self.idata, res=True)
        self.assertIsInstance(ax, plt.Axes)


class TestFormatLabel(unittest.TestCase):
    def test_known_param(self):
        result = Plot._format_label("linear_omega")
        self.assertIn("omega", result)

    def test_with_index(self):
        result = Plot._format_label("keplerian1_amp")
        self.assertIn("1", result)

    def test_unknown_param(self):
        result = Plot._format_label("unknown_xyz")
        self.assertEqual(result, "unknown_xyz")

    def test_with_unit(self):
        result = Plot._format_label("linear_a", unit="d/cycle")
        self.assertIn("d/cycle", result)

    def test_direct_mapping(self):
        result = Plot._format_label("omega")
        self.assertIn("omega", result)

    def test_no_underscore(self):
        result = Plot._format_label("something")
        self.assertEqual(result, "something")
