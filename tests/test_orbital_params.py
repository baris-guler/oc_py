import unittest
import numpy as np
from ocpy.orbital_params import (
    period_to_years,
    a12_sini,
    mass_function,
    m3_sini,
    a3_sini,
    msun_to_mjup,
    compute_orbital_params,
    compute_orbital_params_newtonian,
    compute_orbital_params_from_fit,
    forward_model,
    forward_model_from_fit,
    OrbitalParamsResult,
    OrbitalParamsCollection,
    _C_AU_PER_DAY,
    _MSUN_TO_MJUP,
    _MSUN_TO_MEARTH,
    _AU_TO_RSUN,
    _DAYS_PER_YEAR,
)


class TestConstants(unittest.TestCase):
    def test_c_au_per_day(self):
        self.assertAlmostEqual(_C_AU_PER_DAY, 173.145, places=2)

    def test_msun_to_mjup(self):
        self.assertAlmostEqual(_MSUN_TO_MJUP, 1047.35, places=0)

    def test_msun_to_mearth(self):
        self.assertAlmostEqual(_MSUN_TO_MEARTH, 332946, places=-1)

    def test_au_to_rsun(self):
        self.assertAlmostEqual(_AU_TO_RSUN, 215.03, places=0)


class TestPeriodToYears(unittest.TestCase):
    def test_basic(self):
        P_yr = period_to_years(10000, 0.1)
        expected = (10000 * 0.1) / 365.242199
        self.assertAlmostEqual(P_yr, expected)

    def test_one_year(self):
        P_yr = period_to_years(365.242199, 1.0)
        self.assertAlmostEqual(P_yr, 1.0, places=5)


class TestA12Sini(unittest.TestCase):
    def test_circular(self):
        result = a12_sini(0.001, e=0.0, omega_deg=0.0)
        self.assertAlmostEqual(result, 0.001 * _C_AU_PER_DAY, places=8)

    def test_eccentric(self):
        result = a12_sini(0.001, e=0.5, omega_deg=0.0)
        expected = 0.001 * _C_AU_PER_DAY / np.sqrt(0.75)
        self.assertAlmostEqual(result, expected, places=8)

    def test_omega_90(self):
        result = a12_sini(0.001, e=0.5, omega_deg=90.0)
        self.assertAlmostEqual(result, 0.001 * _C_AU_PER_DAY, places=8)


class TestMassFunction(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(mass_function(1.0, 1.0), 1.0)

    def test_scaling(self):
        f1 = mass_function(2.0, 1.0)
        f2 = mass_function(1.0, 2.0)
        self.assertAlmostEqual(f2 / f1, 32.0)


class TestM3Sini(unittest.TestCase):
    def test_known_solution(self):
        result = m3_sini(1e-9, 1.0)
        approx = (1e-9 * 1.0**2) ** (1.0 / 3.0)
        self.assertAlmostEqual(result, approx, places=4)

    def test_consistency(self):
        M, x_true = 0.6, 0.003
        f = x_true**3 / (M + x_true)**2
        self.assertAlmostEqual(m3_sini(f, M), x_true, places=8)

    def test_array_input(self):
        f_arr = np.array([1e-9, 1e-8, 1e-7])
        result = m3_sini(f_arr, 0.6)
        self.assertEqual(result.shape, (3,))
        for i in range(3):
            f_check = result[i]**3 / (0.6 + result[i])**2
            self.assertAlmostEqual(f_check, f_arr[i], places=12)


class TestA3Sini(unittest.TestCase):
    def test_basic(self):
        result = a3_sini(a12sini_au=0.01, m_total=0.6, m3sini_val=0.003)
        self.assertAlmostEqual(result, 0.01 * 0.6 / 0.003)


class TestMsunToMjup(unittest.TestCase):
    def test_one_msun(self):
        self.assertAlmostEqual(msun_to_mjup(1.0), _MSUN_TO_MJUP, places=1)

    def test_jupiter(self):
        self.assertAlmostEqual(msun_to_mjup(1.0 / _MSUN_TO_MJUP), 1.0, places=5)


class TestComputeOrbitalParams(unittest.TestCase):
    def test_returns_all_keys(self):
        result = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        expected_keys = {
            "amp_day", "amp_s", "e", "omega_deg", "omega_rad",
            "P_day", "P_yr", "a12_sini_au", "a12_sini_rsun",
            "f_mass_msun", "m3_sini_msun", "m3_sini_mjup",
            "m3_sini_mearth", "a3_sini_au",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_unit_conversions(self):
        result = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        self.assertAlmostEqual(result["amp_day"], 0.001)
        self.assertAlmostEqual(result["amp_s"], 0.001 * 86400)
        self.assertAlmostEqual(result["e"], 0.0)
        self.assertAlmostEqual(result["omega_deg"], 0.0)
        self.assertAlmostEqual(result["omega_rad"], 0.0)
        self.assertAlmostEqual(result["P_day"], 30000 * 0.1)
        self.assertAlmostEqual(result["P_yr"], result["P_day"] / 365.242199)
        self.assertAlmostEqual(
            result["a12_sini_rsun"], result["a12_sini_au"] * _AU_TO_RSUN, places=4
        )
        self.assertAlmostEqual(
            result["m3_sini_mjup"], result["m3_sini_msun"] * _MSUN_TO_MJUP, places=4
        )
        self.assertAlmostEqual(
            result["m3_sini_mearth"], result["m3_sini_msun"] * _MSUN_TO_MEARTH, places=0
        )

    def test_ny_vir_like(self):
        amp_days = 35.0 / 86400.0
        result = compute_orbital_params(
            amp=amp_days, e=0.0, omega_deg=0.0,
            P_cycles=8127.0 / 0.101, ref_period=0.101,
            m1=0.471, m2=0.14,
        )
        self.assertGreater(result["P_yr"], 20)
        self.assertGreater(result["m3_sini_mjup"], 1.0)
        self.assertLess(result["m3_sini_mjup"], 10.0)

    def test_array_input_posterior(self):
        np.random.seed(42)
        n = 100
        result = compute_orbital_params(
            amp=np.random.normal(0.0004, 0.00002, n),
            e=np.abs(np.random.normal(0.05, 0.01, n)),
            omega_deg=np.random.normal(270, 10, n),
            P_cycles=np.random.normal(80000, 1000, n),
            ref_period=0.101, m1=0.471, m2=0.14,
        )
        self.assertEqual(result["m3_sini_mjup"].shape, (n,))
        self.assertTrue(np.all(np.isfinite(result["m3_sini_mjup"])))


class _FakeParam:
    def __init__(self, value):
        self.value = value


class _FakeLMFitParams:
    def __init__(self, param_dict):
        self._params = param_dict

    def __getitem__(self, key):
        return self._params[key]

    def keys(self):
        return self._params.keys()

    def valuesdict(self):
        return {k: v.value for k, v in self._params.items()}


class _FakeLMFitResult:
    def __init__(self, params_dict):
        self.params = _FakeLMFitParams(
            {k: _FakeParam(v) for k, v in params_dict.items()}
        )


class _FakeDataArray:
    def __init__(self, values):
        self._values = np.asarray(values)

    @property
    def values(self):
        return self._values

    def flatten(self):
        return self._values.flatten()


class _FakePosterior:
    def __init__(self, data_vars_dict):
        self._vars = {k: _FakeDataArray(v) for k, v in data_vars_dict.items()}

    def __getitem__(self, key):
        return self._vars[key]

    @property
    def data_vars(self):
        return self._vars.keys()


class _FakePyMCResult:
    def __init__(self, data_vars_dict):
        self.posterior = _FakePosterior(data_vars_dict)


class TestComputeOrbitalParamsFromFitLMFit(unittest.TestCase):
    def test_lmfit_single_keplerian(self):
        res = _FakeLMFitResult({
            "keplerian_amp": 0.001, "keplerian_e": 0.0,
            "keplerian_omega": 0.0, "keplerian_P": 30000, "keplerian_T0": 0.0,
        })
        result = compute_orbital_params_from_fit(res, ref_period=0.1, m1=0.47, m2=0.14)
        expected = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        for key in expected:
            self.assertAlmostEqual(result[key], expected[key])

    def test_lmfit_numbered_prefix(self):
        res = _FakeLMFitResult({
            "keplerian1_amp": 0.001, "keplerian1_e": 0.0,
            "keplerian1_omega": 0.0, "keplerian1_P": 30000, "keplerian1_T0": 0.0,
        })
        result = compute_orbital_params_from_fit(res, ref_period=0.1, m1=0.47, m2=0.14)
        self.assertIn("P_yr", result)

    def test_lmfit_two_keplerians(self):
        res = _FakeLMFitResult({
            "keplerian1_amp": 0.001, "keplerian1_e": 0.0,
            "keplerian1_omega": 0.0, "keplerian1_P": 30000, "keplerian1_T0": 0.0,
            "keplerian2_amp": 0.0005, "keplerian2_e": 0.1,
            "keplerian2_omega": 90.0, "keplerian2_P": 50000, "keplerian2_T0": 0.0,
        })
        result = compute_orbital_params_from_fit(res, ref_period=0.1, m1=0.47, m2=0.14)
        self.assertIsInstance(result, OrbitalParamsCollection)
        self.assertIn("keplerian1", result)
        self.assertIn("keplerian2", result)
        self.assertIn("P_yr", result["keplerian1"])
        self.assertIn("P_yr", result["keplerian2"])

    def test_lmfit_explicit_prefix(self):
        res = _FakeLMFitResult({
            "keplerian1_amp": 0.001, "keplerian1_e": 0.0,
            "keplerian1_omega": 0.0, "keplerian1_P": 30000, "keplerian1_T0": 0.0,
            "keplerian2_amp": 0.0005, "keplerian2_e": 0.1,
            "keplerian2_omega": 90.0, "keplerian2_P": 50000, "keplerian2_T0": 0.0,
        })
        result = compute_orbital_params_from_fit(
            res, ref_period=0.1, m1=0.47, m2=0.14, prefix="keplerian2"
        )
        self.assertIn("P_yr", result)


class TestComputeOrbitalParamsFromFitPyMC(unittest.TestCase):
    def test_pymc_single_keplerian(self):
        n = 50
        res = _FakePyMCResult({
            "keplerian_amp": np.full(n, 0.001), "keplerian_e": np.full(n, 0.0),
            "keplerian_omega": np.full(n, 0.0), "keplerian_P": np.full(n, 30000),
        })
        result = compute_orbital_params_from_fit(res, ref_period=0.1, m1=0.47, m2=0.14)
        self.assertEqual(result["m3_sini_mjup"].shape, (n,))
        self.assertAlmostEqual(result["P_yr"][0], result["P_yr"][-1])

    def test_pymc_posterior_spread(self):
        np.random.seed(42)
        n = 100
        res = _FakePyMCResult({
            "keplerian1_amp": np.random.normal(0.0004, 0.00002, n),
            "keplerian1_e": np.abs(np.random.normal(0.05, 0.01, n)),
            "keplerian1_omega": np.random.normal(270, 10, n),
            "keplerian1_P": np.random.normal(80000, 1000, n),
        })
        result = compute_orbital_params_from_fit(
            res, ref_period=0.101, m1=0.471, m2=0.14
        )
        self.assertEqual(result["m3_sini_mjup"].shape, (n,))
        self.assertTrue(np.all(np.isfinite(result["m3_sini_mjup"])))


class TestComputeOrbitalParamsFromFitErrors(unittest.TestCase):
    def test_invalid_result_type(self):
        with self.assertRaises(TypeError):
            compute_orbital_params_from_fit("not_a_result", 0.1, 0.47, 0.14)

    def test_no_keplerian_params(self):
        res = _FakeLMFitResult({"linear_a": 0.0, "linear_b": 0.0})
        with self.assertRaises(ValueError):
            compute_orbital_params_from_fit(res, 0.1, 0.47, 0.14)

    def test_keplerian_requires_ref_period(self):
        res = _FakeLMFitResult({
            "keplerian_amp": 0.001, "keplerian_e": 0.0,
            "keplerian_omega": 0.0, "keplerian_P": 30000, "keplerian_T0": 0.0,
        })
        with self.assertRaises(ValueError):
            compute_orbital_params_from_fit(res)


class TestComputeOrbitalParamsNewtonian(unittest.TestCase):
    def test_with_a(self):
        result = compute_orbital_params_newtonian(
            m3=0.001, e=0.0, omega_deg=0.0, m_central=0.6, a3_au=5.0,
        )
        self.assertIsInstance(result, OrbitalParamsResult)
        self.assertAlmostEqual(result["a3_sini_au"], 5.0)
        self.assertGreater(result["P_yr"], 0)
        P_yr_check = np.sqrt(5.0**3 / (0.6 + 0.001))
        self.assertAlmostEqual(result["P_yr"], P_yr_check, places=4)

    def test_with_P(self):
        result = compute_orbital_params_newtonian(
            m3=0.001, e=0.0, omega_deg=0.0, m_central=0.6, P_day=3000.0,
        )
        self.assertIsInstance(result, OrbitalParamsResult)
        self.assertAlmostEqual(result["P_day"], 3000.0)
        self.assertGreater(result["a3_sini_au"], 0)
        P_yr = 3000.0 / _DAYS_PER_YEAR
        a_check = ((0.6 + 0.001) * P_yr**2) ** (1.0 / 3.0)
        self.assertAlmostEqual(result["a3_sini_au"], a_check, places=4)

    def test_both_a_and_P_raises(self):
        with self.assertRaises(ValueError):
            compute_orbital_params_newtonian(
                m3=0.001, e=0.0, omega_deg=0.0, m_central=0.6,
                a3_au=5.0, P_day=3000.0,
            )

    def test_neither_a_nor_P_raises(self):
        with self.assertRaises(ValueError):
            compute_orbital_params_newtonian(
                m3=0.001, e=0.0, omega_deg=0.0, m_central=0.6,
            )

    def test_inclination(self):
        r90 = compute_orbital_params_newtonian(
            m3=0.001, e=0.0, omega_deg=0.0, m_central=0.6,
            a3_au=5.0, inc_deg=90.0,
        )
        r60 = compute_orbital_params_newtonian(
            m3=0.001, e=0.0, omega_deg=0.0, m_central=0.6,
            a3_au=5.0, inc_deg=60.0,
        )
        self.assertAlmostEqual(r90["m3_sini_msun"], 0.001)
        self.assertAlmostEqual(r60["m3_sini_msun"], 0.001 * np.sin(np.deg2rad(60)))
        self.assertAlmostEqual(r60["a3_sini_au"], 5.0 * np.sin(np.deg2rad(60)))


class TestComputeOrbitalParamsFromFitNewtonian(unittest.TestCase):
    def test_lmfit_newtonian_with_a(self):
        res = _FakeLMFitResult({
            "newtonian_central_mass": 0.6,
            "newtonian_b1_m": 0.001,
            "newtonian_b1_a": 5.0,
            "newtonian_b1_e": 0.1,
            "newtonian_b1_omega": 90.0,
        })
        result = compute_orbital_params_from_fit(res)
        self.assertIn("P_yr", result)
        self.assertGreater(result["a3_sini_au"], 0)

    def test_lmfit_newtonian_with_P(self):
        res = _FakeLMFitResult({
            "newtonian_central_mass": 0.6,
            "newtonian_b1_m": 0.001,
            "newtonian_b1_P": 3000.0,
            "newtonian_b1_e": 0.0,
            "newtonian_b1_omega": 0.0,
        })
        result = compute_orbital_params_from_fit(res)
        self.assertAlmostEqual(result["P_day"], 3000.0)

    def test_lmfit_newtonian_two_bodies(self):
        res = _FakeLMFitResult({
            "newtonian_central_mass": 0.6,
            "newtonian_b1_m": 0.001, "newtonian_b1_a": 5.0,
            "newtonian_b1_e": 0.0, "newtonian_b1_omega": 0.0,
            "newtonian_b2_m": 0.003, "newtonian_b2_P": 5000.0,
            "newtonian_b2_e": 0.1, "newtonian_b2_omega": 45.0,
        })
        result = compute_orbital_params_from_fit(res)
        self.assertIsInstance(result, OrbitalParamsCollection)
        self.assertIn("newtonian_b1", result)
        self.assertIn("newtonian_b2", result)
        self.assertIn("P_yr", result["newtonian_b1"])
        self.assertIn("a3_sini_au", result["newtonian_b2"])

    def test_pymc_newtonian(self):
        n = 50
        res = _FakePyMCResult({
            "newtonian_central_mass": np.full(n, 0.6),
            "newtonian_b1_m": np.full(n, 0.001),
            "newtonian_b1_a": np.full(n, 5.0),
            "newtonian_b1_e": np.full(n, 0.0),
            "newtonian_b1_omega": np.full(n, 0.0),
        })
        result = compute_orbital_params_from_fit(res)
        self.assertEqual(result["m3_sini_mjup"].shape, (n,))

    def test_explicit_newtonian_prefix(self):
        res = _FakeLMFitResult({
            "newtonian_central_mass": 0.6,
            "newtonian_b1_m": 0.001, "newtonian_b1_a": 5.0,
            "newtonian_b1_e": 0.0, "newtonian_b1_omega": 0.0,
        })
        result = compute_orbital_params_from_fit(res, prefix="newtonian")
        self.assertIn("P_yr", result)

    def test_custom_name_newtonian(self):
        res = _FakeLMFitResult({
            "nbody_central_mass": 0.6,
            "nbody_b1_m": 0.001, "nbody_b1_P": 3000.0,
            "nbody_b1_e": 0.0, "nbody_b1_omega": 0.0,
        })
        result = compute_orbital_params_from_fit(res)
        self.assertIn("P_yr", result)
        self.assertAlmostEqual(result["P_day"], 3000.0)

    def test_custom_name_newtonian_pymc(self):
        n = 50
        res = _FakePyMCResult({
            "nbody_central_mass": np.full(n, 0.611),
            "nbody_b1_m": np.full(n, 0.0021),
            "nbody_b1_P": np.full(n, 3170.0),
            "nbody_b1_e": np.full(n, 0.05),
            "nbody_b1_omega": np.full(n, 269.0),
        })
        result = compute_orbital_params_from_fit(res)
        self.assertEqual(result["m3_sini_mjup"].shape, (n,))

    def test_custom_name_newtonian_two_bodies(self):
        res = _FakeLMFitResult({
            "nbody_central_mass": 0.611,
            "nbody_b1_m": 0.0021, "nbody_b1_P": 3170.0,
            "nbody_b1_e": 0.05, "nbody_b1_omega": 269.0,
            "nbody_b2_m": 0.0038, "nbody_b2_P": 8260.0,
            "nbody_b2_e": 0.02, "nbody_b2_omega": 140.0,
        })
        result = compute_orbital_params_from_fit(res)
        self.assertIsInstance(result, OrbitalParamsCollection)
        self.assertIn("nbody_b1", result)
        self.assertIn("nbody_b2", result)


class TestOrbitalParamsResult(unittest.TestCase):
    def test_is_result_type(self):
        result = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        self.assertIsInstance(result, OrbitalParamsResult)

    def test_dict_access(self):
        result = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        self.assertIn("P_yr", result)
        self.assertIsInstance(result["P_yr"], float)
        self.assertEqual(len(list(result.keys())), 14)

    def test_to_dict(self):
        result = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(set(d.keys()), set(result.keys()))

    def test_repr_scalar(self):
        result = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        text = repr(result)
        self.assertIn("OrbitalParamsResult", text)
        self.assertIn("m_3 sin i", text)
        self.assertNotIn("N=", text)

    def test_repr_posterior(self):
        np.random.seed(42)
        n = 100
        result = compute_orbital_params(
            amp=np.random.normal(0.0004, 0.00002, n),
            e=np.abs(np.random.normal(0.05, 0.01, n)),
            omega_deg=np.random.normal(270, 10, n),
            P_cycles=np.random.normal(80000, 1000, n),
            ref_period=0.101, m1=0.471, m2=0.14,
        )
        text = repr(result)
        self.assertIn("N=100", text)
        self.assertTrue(result.is_posterior)

    def test_repr_html(self):
        result = compute_orbital_params(
            amp=0.001, e=0.0, omega_deg=0.0,
            P_cycles=30000, ref_period=0.1, m1=0.47, m2=0.14,
        )
        html = result._repr_html_()
        self.assertIn("<table", html)
        self.assertIn("m_3 sin i", html)

    def test_repr_html_posterior(self):
        np.random.seed(42)
        n = 50
        result = compute_orbital_params(
            amp=np.full(n, 0.001), e=np.full(n, 0.0),
            omega_deg=np.full(n, 0.0), P_cycles=np.full(n, 30000),
            ref_period=0.1, m1=0.47, m2=0.14,
        )
        html = result._repr_html_()
        self.assertIn("Posterior samples", html)
        self.assertIn("N = 50", html)


class TestOrbitalParamsCollection(unittest.TestCase):
    def test_collection_from_two_keplerians(self):
        res = _FakeLMFitResult({
            "keplerian1_amp": 0.001, "keplerian1_e": 0.0,
            "keplerian1_omega": 0.0, "keplerian1_P": 30000, "keplerian1_T0": 0.0,
            "keplerian2_amp": 0.0005, "keplerian2_e": 0.1,
            "keplerian2_omega": 90.0, "keplerian2_P": 50000, "keplerian2_T0": 0.0,
        })
        col = compute_orbital_params_from_fit(res, ref_period=0.1, m1=0.47, m2=0.14)
        self.assertIsInstance(col, OrbitalParamsCollection)
        self.assertEqual(len(col), 2)

    def test_collection_repr(self):
        res = _FakeLMFitResult({
            "keplerian1_amp": 0.001, "keplerian1_e": 0.0,
            "keplerian1_omega": 0.0, "keplerian1_P": 30000, "keplerian1_T0": 0.0,
            "keplerian2_amp": 0.0005, "keplerian2_e": 0.1,
            "keplerian2_omega": 90.0, "keplerian2_P": 50000, "keplerian2_T0": 0.0,
        })
        col = compute_orbital_params_from_fit(res, ref_period=0.1, m1=0.47, m2=0.14)
        text = repr(col)
        self.assertIn("keplerian1", text)
        self.assertIn("keplerian2", text)

    def test_collection_repr_html(self):
        res = _FakeLMFitResult({
            "keplerian1_amp": 0.001, "keplerian1_e": 0.0,
            "keplerian1_omega": 0.0, "keplerian1_P": 30000, "keplerian1_T0": 0.0,
            "keplerian2_amp": 0.0005, "keplerian2_e": 0.1,
            "keplerian2_omega": 90.0, "keplerian2_P": 50000, "keplerian2_T0": 0.0,
        })
        col = compute_orbital_params_from_fit(res, ref_period=0.1, m1=0.47, m2=0.14)
        html = col._repr_html_()
        self.assertIn("keplerian1", html)
        self.assertIn("keplerian2", html)
        self.assertIn("<table", html)

    def test_collection_iter(self):
        res = _FakeLMFitResult({
            "newtonian_central_mass": 0.6,
            "newtonian_b1_m": 0.001, "newtonian_b1_a": 5.0,
            "newtonian_b1_e": 0.0, "newtonian_b1_omega": 0.0,
            "newtonian_b2_m": 0.003, "newtonian_b2_P": 5000.0,
            "newtonian_b2_e": 0.1, "newtonian_b2_omega": 45.0,
        })
        col = compute_orbital_params_from_fit(res)
        keys = list(col)
        self.assertEqual(len(keys), 2)
        for key in keys:
            self.assertIsInstance(col[key], OrbitalParamsResult)


class TestForwardModel(unittest.TestCase):
    def test_circular_orbit(self):
        """Circular orbit (e=0) should produce a sinusoidal O-C curve."""
        cycles = np.linspace(0, 30000, 500)
        oc = forward_model(cycles, amp=0.001, e=0.0, omega_deg=0.0,
                           P_cycles=30000, T0=0.0)
        self.assertEqual(oc.shape, (500,))
        # Peak-to-peak should be ~2*amp
        self.assertAlmostEqual(np.max(oc) - np.min(oc), 2 * 0.001, places=4)

    def test_zero_at_T0(self):
        """At T0 the O-C for circular orbit should pass through ~0."""
        cycles = np.array([0.0])
        oc = forward_model(cycles, amp=0.001, e=0.0, omega_deg=0.0,
                           P_cycles=30000, T0=0.0)
        self.assertAlmostEqual(float(oc[0]), 0.0, places=8)

    def test_matches_keplerian_class(self):
        """forward_model should match Keplerian.model_func output."""
        from ocpy.oc import Keplerian
        kep = Keplerian()
        cycles = np.linspace(0, 50000, 200)
        amp, e, omega, P, T0 = 0.0005, 0.3, 45.0, 40000.0, 5000.0
        oc_fm = forward_model(cycles, amp, e, omega, P, T0)
        oc_kep = kep.model_func(cycles, amp, e, omega, P, T0)
        np.testing.assert_allclose(oc_fm, oc_kep, atol=1e-12)

    def test_posterior_shape(self):
        """Array params should return (N_samples, N_cycles)."""
        n = 200
        cycles = np.linspace(0, 50000, 100)
        oc = forward_model(
            cycles,
            amp=np.full(n, 0.001),
            e=np.full(n, 0.0),
            omega_deg=np.full(n, 0.0),
            P_cycles=np.full(n, 30000),
            T0=np.full(n, 0.0),
        )
        self.assertEqual(oc.shape, (n, 100))

    def test_posterior_spread(self):
        """Different posterior samples should produce different curves."""
        np.random.seed(42)
        n = 50
        cycles = np.linspace(0, 50000, 100)
        oc = forward_model(
            cycles,
            amp=np.random.normal(0.001, 0.0001, n),
            e=np.abs(np.random.normal(0.1, 0.02, n)),
            omega_deg=np.random.normal(90, 10, n),
            P_cycles=np.random.normal(30000, 500, n),
            T0=np.random.normal(0, 100, n),
        )
        self.assertEqual(oc.shape, (n, 100))
        # Rows should not all be identical
        self.assertGreater(np.std(oc[:, 50]), 0)

    def test_eccentric_vs_circular(self):
        """Eccentric orbit curve should differ from circular."""
        cycles = np.linspace(0, 30000, 500)
        oc_circ = forward_model(cycles, amp=0.001, e=0.0, omega_deg=0.0,
                                P_cycles=30000, T0=0.0)
        oc_ecc = forward_model(cycles, amp=0.001, e=0.5, omega_deg=45.0,
                               P_cycles=30000, T0=0.0)
        # Curves should not be identical
        self.assertGreater(np.max(np.abs(oc_ecc - oc_circ)), 1e-5)


class TestForwardModelFromFit(unittest.TestCase):
    def test_lmfit_single_keplerian(self):
        res = _FakeLMFitResult({
            "keplerian_amp": 0.001, "keplerian_e": 0.0,
            "keplerian_omega": 0.0, "keplerian_P": 30000, "keplerian_T0": 0.0,
        })
        cycles = np.linspace(0, 30000, 100)
        oc = forward_model_from_fit(res, cycles)
        self.assertEqual(oc.shape, (100,))
        expected = forward_model(cycles, 0.001, 0.0, 0.0, 30000, 0.0)
        np.testing.assert_allclose(oc, expected, atol=1e-14)

    def test_pymc_posterior(self):
        n = 50
        res = _FakePyMCResult({
            "keplerian_amp": np.full(n, 0.001),
            "keplerian_e": np.full(n, 0.0),
            "keplerian_omega": np.full(n, 0.0),
            "keplerian_P": np.full(n, 30000),
            "keplerian_T0": np.full(n, 0.0),
        })
        cycles = np.linspace(0, 30000, 80)
        oc = forward_model_from_fit(res, cycles)
        self.assertEqual(oc.shape, (n, 80))

    def test_two_keplerians_sum(self):
        """Two keplerian components should be summed."""
        res = _FakeLMFitResult({
            "keplerian1_amp": 0.001, "keplerian1_e": 0.0,
            "keplerian1_omega": 0.0, "keplerian1_P": 30000, "keplerian1_T0": 0.0,
            "keplerian2_amp": 0.0005, "keplerian2_e": 0.0,
            "keplerian2_omega": 0.0, "keplerian2_P": 15000, "keplerian2_T0": 0.0,
        })
        cycles = np.linspace(0, 30000, 100)
        oc = forward_model_from_fit(res, cycles)
        oc1 = forward_model(cycles, 0.001, 0.0, 0.0, 30000, 0.0)
        oc2 = forward_model(cycles, 0.0005, 0.0, 0.0, 15000, 0.0)
        np.testing.assert_allclose(oc, oc1 + oc2, atol=1e-14)

    def test_newtonian_raises(self):
        res = _FakeLMFitResult({
            "newtonian_central_mass": 0.6,
            "newtonian_b1_m": 0.001, "newtonian_b1_P": 3000.0,
            "newtonian_b1_e": 0.0, "newtonian_b1_omega": 0.0,
        })
        with self.assertRaises(ValueError):
            forward_model_from_fit(res, np.linspace(0, 1000, 10))

    def test_invalid_result_type(self):
        with self.assertRaises(TypeError):
            forward_model_from_fit("not_a_result", np.array([0.0]))

    def test_no_keplerian_params(self):
        res = _FakeLMFitResult({"linear_a": 0.0, "linear_b": 0.0})
        with self.assertRaises(ValueError):
            forward_model_from_fit(res, np.array([0.0]))
