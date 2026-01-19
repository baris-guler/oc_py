import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ocpy.visualization import Plot
from ocpy.oc import OC

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.cycle = np.arange(10)
        self.oc_vals = np.sin(self.cycle)
        self.oc_err = np.ones(10) * 0.1
        self.labels = ["A"] * 5 + ["B"] * 5
        
        self.data_dict = {
            "cycle": self.cycle,
            "oc": self.oc_vals,
            "minimum_time_error": self.oc_err,
            "labels": self.labels,
            "minimum_time": self.cycle,
            "weights": np.ones(10),
            "minimum_type": ["p"] * 10
        }
        self.oc_obj = OC(**self.data_dict)

    @patch("matplotlib.pyplot.subplots")
    def test_plot_data_basic(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        ax = Plot.plot_data(self.oc_obj)
        
        self.assertEqual(ax, mock_ax)
        mock_ax.errorbar.assert_called()
        mock_ax.set_ylabel.assert_called_with("O−C")
        mock_ax.set_xlabel.assert_called_with("Cycle")

    @patch("matplotlib.pyplot.subplots")
    def test_plot_data_with_labels(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        Plot.plot_data(self.oc_obj)
        
        self.assertTrue(mock_ax.errorbar.call_count >= 2)
        mock_ax.legend.assert_called()

    @patch("matplotlib.pyplot.subplots")
    def test_plot_wrapper(self, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        Plot.plot(self.oc_obj, residuals=False)
        
        mock_ax.errorbar.assert_called()

    @patch("matplotlib.pyplot.subplots")
    def test_plot_residuals(self, mock_subplots):
        mock_fig = MagicMock()
        mock_main_ax = MagicMock()
        mock_resid_ax = MagicMock()
        
        mock_subplots.return_value = (mock_fig, (mock_main_ax, mock_resid_ax))
        
        mock_subplots.return_value = (mock_fig, mock_main_ax)
        Plot.plot(self.oc_obj, residuals=True)
        mock_main_ax.errorbar.assert_called()

    @patch("matplotlib.pyplot.subplots")
    @patch("ocpy.visualization.Plot.plot_model_components")
    def test_plot_with_model(self, mock_plot_comps, mock_subplots):
        mock_fig = MagicMock()
        mock_main_ax = MagicMock()
        mock_resid_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_main_ax, mock_resid_ax))
        
        model = [] 
        
        Plot.plot(self.oc_obj, model=model, residuals=True)
        
        mock_plot_comps.assert_called()
        self.assertTrue(mock_main_ax.errorbar.call_count > 0)
        mock_resid_ax.set_ylabel.assert_called_with("Resid")

    @patch("arviz.extract")
    @patch("corner.corner")
    def test_plot_corner(self, mock_corner, mock_az_extract):
        mock_idata = MagicMock()
        
        mock_var = MagicMock()
        mock_var.values = np.random.rand(100, 4) 
        mock_var.ndim = 2
        
        mock_idata.posterior.data_vars = {"a": mock_var}
        mock_idata.posterior.__getitem__.return_value = mock_var
        
        mock_az_extract.return_value = {"a": mock_var}

        from ocpy.visualization import Plot
        Plot.plot_corner(mock_idata, cornerstyle="corner")
        
        mock_corner.assert_called()

    @patch("arviz.plot_trace")
    def test_plot_trace(self, mock_az_plot_trace):
        mock_idata = MagicMock()
        
        mock_var = MagicMock()
        mock_var.values = np.random.rand(100, 4)
        
        mock_idata.posterior.data_vars = {"a": mock_var}
        mock_idata.posterior.__getitem__.return_value = mock_var
        
        mock_axes = np.array([MagicMock()])
        mock_az_plot_trace.return_value = mock_axes
        
        from ocpy.visualization import Plot
        Plot.plot_trace(mock_idata)
        
        mock_az_plot_trace.assert_called()
