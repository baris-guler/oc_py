from unittest import TestCase

import numpy as np
import pytensor.tensor as pt
from astropy import constants as const
from astropy import units as u

from ocpy.newtonian import _c_for_time_unit, NewtonianModel, NewtonianOp, _NewtonianGradOp
from ocpy.oc import Parameter


class TestCForTimeUnit(TestCase):
    def test_day(self):
        expected = const.c.to(u.au / u.day).value
        self.assertAlmostEqual(_c_for_time_unit("day"), expected, places=4)

    def test_d_alias(self):
        self.assertAlmostEqual(_c_for_time_unit("d"), _c_for_time_unit("day"))

    def test_year(self):
        expected = const.c.to(u.au / u.yr).value
        self.assertAlmostEqual(_c_for_time_unit("year"), expected, places=2)

    def test_yr_alias(self):
        self.assertAlmostEqual(_c_for_time_unit("yr"), _c_for_time_unit("year"))

    def test_sec(self):
        expected = const.c.to(u.au / u.s).value
        self.assertAlmostEqual(_c_for_time_unit("s"), expected, places=6)

    def test_sec_alias(self):
        self.assertAlmostEqual(_c_for_time_unit("sec"), _c_for_time_unit("s"))

    def test_case_insensitive(self):
        self.assertAlmostEqual(_c_for_time_unit("DAY"), _c_for_time_unit("day"))
        self.assertAlmostEqual(_c_for_time_unit("  Day  "), _c_for_time_unit("day"))

    def test_invalid_unit(self):
        with self.assertRaises(ValueError):
            _c_for_time_unit("hour")


class TestNewtonianModelInit(TestCase):
    def test_default_init(self):
        model = NewtonianModel()
        self.assertEqual(model.name, "newtonian")
        self.assertEqual(model.integrator, "ias15")
        self.assertAlmostEqual(model.dt, 0.01)
        self.assertEqual(model.units, {"m": "msun", "t": "day", "l": "au"})
        self.assertIn("central_mass", model.params)
        self.assertEqual(model.central_mass.value, 1.0)

    def test_custom_name(self):
        model = NewtonianModel(name="custom")
        self.assertEqual(model.name, "custom")

    def test_custom_units(self):
        model = NewtonianModel(units={"t": "yr"})
        self.assertEqual(model.units["t"], "yr")
        self.assertEqual(model.units["m"], "msun")
        expected_c = const.c.to(u.au / u.yr).value
        self.assertAlmostEqual(model._c_light, expected_c, places=2)

    def test_bodies(self):
        bodies = [
            {"m": 0.001, "a": 1.0, "e": 0.1, "inc": 90.0, "omega": 0.0, "Omega": 0.0, "M": 0.0},
        ]
        model = NewtonianModel(bodies=bodies)
        self.assertIn("b1_m", model.params)
        self.assertIn("b1_a", model.params)
        self.assertIn("b1_e", model.params)
        self.assertEqual(model.params["b1_m"].value, 0.001)
        self.assertEqual(model.params["b1_e"].min, 0.0)
        self.assertEqual(model.params["b1_e"].max, 1.0)

    def test_multiple_bodies(self):
        bodies = [
            {"m": 0.001, "a": 1.0, "e": 0.0},
            {"m": 0.002, "a": 2.0, "e": 0.1},
        ]
        model = NewtonianModel(bodies=bodies)
        self.assertIn("b1_m", model.params)
        self.assertIn("b2_m", model.params)
        self.assertEqual(model.params["b2_m"].value, 0.002)

    def test_central_mass_parameter(self):
        p = Parameter(value=2.0, min=0.5, max=5.0)
        model = NewtonianModel(central_mass=p)
        self.assertIs(model.central_mass, p)
        self.assertEqual(model.central_mass.value, 2.0)

    def test_integration_grid(self):
        grid = [0.0, 1.0, 2.0, 3.0]
        model = NewtonianModel(integration_grid=grid)
        np.testing.assert_array_equal(model.integration_grid, np.array(grid))

    def test_no_integration_grid(self):
        model = NewtonianModel()
        self.assertIsNone(model.integration_grid)


class TestNewtonianModelIntegration(TestCase):
    def setUp(self):
        self.bodies = [
            {"m": 0.001, "a": 1.0, "e": 0.0, "inc": 90.0, "omega": 0.0, "Omega": 0.0, "M": 0.0},
        ]
        self.model = NewtonianModel(
            bodies=self.bodies,
            central_mass=1.0,
            dt=0.01,
        )

    def test_setup_rebound(self):
        params = {k: p.value for k, p in self.model.params.items()}
        sim = self.model._setup_rebound(params)
        self.assertEqual(sim.N, 2)  # central + 1 body

    def test_integrate_returns_dict(self):
        times = np.linspace(0, 10, 20)
        result = self.model.integrate(times)
        self.assertIn("D", result)
        self.assertIn("E", result)
        self.assertIn("F", result)
        self.assertIn("G", result)

    def test_integrate_xyz_shape(self):
        times = np.linspace(0, 10, 20)
        result = self.model.integrate(times)
        self.assertEqual(result["D"].shape, (20, 2, 3))

    def test_integrate_orbital_shape(self):
        times = np.linspace(0, 10, 20)
        result = self.model.integrate(times)
        self.assertEqual(result["E"].shape, (20, 1, 7))

    def test_integrate_no_xyz(self):
        model = NewtonianModel(bodies=self.bodies, compute_xyz=False)
        times = np.linspace(0, 10, 10)
        result = model.integrate(times)
        self.assertIsNone(result["D"])

    def test_integrate_no_orbital(self):
        times = np.linspace(0, 10, 10)
        result = self.model.integrate(times, compute_orbital=False)
        self.assertIsNone(result["E"])

    def test_integrate_time_bounds(self):
        model = NewtonianModel(bodies=self.bodies, t_start=2.0, t_end=8.0)
        times = np.linspace(0, 10, 50)
        result = model.integrate(times)
        # filtered times should produce NaN for out-of-range
        self.assertTrue(result["F"])

    def test_integrate_energy_conservation(self):
        times = np.linspace(0, 100, 200)
        result = self.model.integrate(times)
        self.assertTrue(result["F"])
        self.assertAlmostEqual(result["G"]["delta_E"], 0.0, places=5)

    def test_calculate_etv(self):
        x = np.arange(0, 20, dtype=float)
        params = {k: p.value for k, p in self.model.params.items()}
        etv = self.model._calculate_etv(x, params)
        self.assertEqual(len(etv), len(x))
        self.assertFalse(np.isnan(etv).all())

    def test_calculate_etv_nan_param(self):
        x = np.arange(0, 10, dtype=float)
        params = {k: p.value for k, p in self.model.params.items()}
        params["central_mass"] = np.nan
        etv = self.model._calculate_etv(x, params)
        self.assertTrue(np.isnan(etv).all())

    def test_model_func_numpy(self):
        x = np.arange(0, 10, dtype=float)
        result = self.model.model_func(x, central_mass=1.0, b1_m=0.001)
        self.assertEqual(len(result), len(x))

    def test_setup_rebound_a_and_P_conflict(self):
        bodies = [{"m": 0.001, "a": 1.0, "P": 1.0}]
        model = NewtonianModel(bodies=bodies)
        params = {k: p.value for k, p in model.params.items()}
        with self.assertRaises(ValueError):
            model._setup_rebound(params)

    def test_jacobi_orbit_type(self):
        model = NewtonianModel(bodies=self.bodies, orbit_type="jacobi")
        params = {k: p.value for k, p in model.params.items()}
        sim = model._setup_rebound(params)
        self.assertEqual(sim.N, 2)

    def test_precision_integration_steps(self):
        model = NewtonianModel(
            bodies=self.bodies,
            precision_integration_steps=100,
        )
        x = np.arange(0, 10, dtype=float)
        params = {k: p.value for k, p in model.params.items()}
        etv = model._calculate_etv(x, params)
        self.assertEqual(len(etv), len(x))

    def test_integration_grid_etv(self):
        grid = np.linspace(0, 30, 500)
        model = NewtonianModel(
            bodies=self.bodies,
            integration_grid=grid,
        )
        x = np.arange(0, 20, dtype=float)
        params = {k: p.value for k, p in model.params.items()}
        etv = model._calculate_etv(x, params)
        self.assertEqual(len(etv), len(x))

    def test_integrator_params(self):
        model = NewtonianModel(
            bodies=self.bodies,
            integrator_params={"dt": 0.005},
        )
        params = {k: p.value for k, p in model.params.items()}
        sim = model._setup_rebound(params)
        self.assertAlmostEqual(sim.dt, 0.005)

    def test_T_large_with_T0_ref(self):
        bodies = [{"m": 0.001, "P": 365.25, "e": 0.0, "T": 2_500_000.0}]
        model = NewtonianModel(bodies=bodies, T0_ref=2_400_000.0)
        params = {k: p.value for k, p in model.params.items()}
        sim = model._setup_rebound(params)
        self.assertEqual(sim.N, 2)

    def test_model_func_without_central_mass_kwarg(self):
        x = np.arange(0, 5, dtype=float)
        result = self.model.model_func(x, b1_m=0.001)
        self.assertEqual(len(result), len(x))

    def test_precision_steps_empty_times(self):
        model = NewtonianModel(
            bodies=self.bodies,
            precision_integration_steps=100,
            t_start=100.0,
            t_end=200.0,
        )
        x = np.arange(0, 5, dtype=float)
        params = {k: p.value for k, p in model.params.items()}
        etv = model._calculate_etv(x, params)
        self.assertEqual(len(etv), len(x))

    def test_jacobi_output_type(self):
        model = NewtonianModel(
            bodies=self.bodies,
            orbit_output_type="jacobi",
        )
        times = np.linspace(0, 10, 10)
        result = model.integrate(times)
        self.assertIsNotNone(result["E"])


class TestNewtonianOp(TestCase):
    def setUp(self):
        self.bodies = [
            {"m": 0.001, "a": 1.0, "e": 0.0, "inc": 90.0, "omega": 0.0, "Omega": 0.0, "M": 0.0},
        ]
        self.model = NewtonianModel(bodies=self.bodies, central_mass=1.0)

    def test_newtonian_op_perform(self):
        op = NewtonianOp(self.model)
        x = np.arange(0, 5, dtype=float)
        param_keys = sorted(self.model.params.keys())
        param_vals = [float(self.model.params[k].value) for k in param_keys]

        x_tensor = pt.as_tensor_variable(x)
        param_tensors = [pt.as_tensor_variable(v) for v in param_vals]
        out = op(x_tensor, *param_tensors)

        result = out.eval()
        self.assertEqual(result.shape, (5,))
        self.assertFalse(np.isnan(result).all())

    def test_newtonian_op_make_node(self):
        op = NewtonianOp(self.model)
        x_tensor = pt.as_tensor_variable(np.array([0.0, 1.0]))
        param_keys = sorted(self.model.params.keys())
        param_tensors = [pt.as_tensor_variable(float(self.model.params[k].value)) for k in param_keys]
        node = op.make_node(x_tensor, *param_tensors)
        self.assertEqual(len(node.outputs), 1)

    def test_grad_op_perform(self):
        grad_op = _NewtonianGradOp(self.model, sorted(self.model.params.keys()))
        x = np.arange(0, 5, dtype=float)
        gz = np.ones(5, dtype=float)
        param_keys = sorted(self.model.params.keys())
        param_vals = [float(self.model.params[k].value) for k in param_keys]

        x_tensor = pt.as_tensor_variable(x)
        gz_tensor = pt.as_tensor_variable(gz)
        param_tensors = [pt.as_tensor_variable(v) for v in param_vals]
        grads = grad_op(x_tensor, gz_tensor, *param_tensors)

        if not isinstance(grads, (list, tuple)):
            grads = [grads]
        for g in grads:
            val = g.eval()
            self.assertTrue(np.isfinite(val))

    def test_model_func_symbolic(self):
        x = np.arange(0, 5, dtype=float)
        kwargs = {}
        for k, p in self.model.params.items():
            kwargs[k] = pt.as_tensor_variable(float(p.value))

        result = self.model.model_func(x, **kwargs)
        evaluated = result.eval()
        self.assertEqual(evaluated.shape, (5,))

    def test_op_grad(self):
        op = NewtonianOp(self.model)
        x_tensor = pt.as_tensor_variable(np.arange(0, 3, dtype=float))
        param_keys = sorted(self.model.params.keys())
        param_tensors = [pt.as_tensor_variable(float(self.model.params[k].value)) for k in param_keys]

        out = op(x_tensor, *param_tensors)
        gz = pt.ones_like(out)
        grad_results = op.grad([x_tensor] + param_tensors, [gz])
        self.assertEqual(len(grad_results), 1 + len(param_keys))

    def test_op_grad_single_param(self):
        model = NewtonianModel(central_mass=1.0)
        op = NewtonianOp(model)
        x_tensor = pt.as_tensor_variable(np.array([0.0, 1.0]))
        param_tensors = [pt.as_tensor_variable(1.0)]

        out = op(x_tensor, *param_tensors)
        gz = pt.ones_like(out)
        grad_results = op.grad([x_tensor] + param_tensors, [gz])
        self.assertEqual(len(grad_results), 2)
