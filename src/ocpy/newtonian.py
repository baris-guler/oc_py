import numpy as np
import rebound
from typing import Optional, Dict, List, Any, Union
from ocpy.oc import ModelComponent, Parameter
from ocpy.custom_types import NumberOrParam

try:
    import pytensor
    import pytensor.tensor as pt
    HAS_PYTENSOR = True
except ImportError:
    HAS_PYTENSOR = False


_C_LIGHT: Dict[str, float] = {
    "day": 173.1446,
    "d":   173.1446,
    "yr":  63241.077,
    "year": 63241.077,
    "s":   0.00200398,
    "sec": 0.00200398,
}

def _c_for_time_unit(time_unit: str) -> float:
    key = time_unit.strip().lower()
    if key not in _C_LIGHT:
        raise ValueError(
            f"Bilinmeyen zaman birimi: '{time_unit}'. "
            f"Geçerli değerler: {list(_C_LIGHT.keys())}"
        )
    return _C_LIGHT[key]


class NewtonianModel(ModelComponent):
    name = "newtonian"
    _expensive = True

    def __init__(
        self,
        *,
        integrator: str = "ias15",
        dt: float = 0.01,
        integrator_params: Optional[Dict[str, Any]] = None,
        units: Optional[Dict[str, str]] = None,
        reference_time: float = 0.0,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        stop_at_exact_time: bool = True,
        escape_radius: Optional[float] = 100000.0,
        min_distance: Optional[float] = 0.0001,
        precision_integration_steps: int = 0,
        integration_grid: Optional[Union[np.ndarray, List[float]]] = None,
        central_mass: NumberOrParam = 1.0,
        bodies: Optional[List[Dict[str, Any]]] = None,
        orbit_type: str = "heliocentric",
        orbit_output_type: str = "heliocentric",
        T0_ref: float = 0.0,
        P_ref: float = 1.0,
        compute_xyz: bool = True,
        compute_orbital: bool = True,
        name: Optional[str] = None,
    ) -> None:
        if name is not None:
            self.name = name

        self.integrator = integrator
        self.dt = dt
        self.integrator_params = integrator_params or {}

        _default_units: Dict[str, str] = {"m": "msun", "t": "day", "l": "au"}
        if units:
            self.units = {**_default_units, **units}
        else:
            self.units = _default_units.copy()

        self._c_light: float = _c_for_time_unit(self.units["t"])

        self.reference_time = reference_time
        self.t_start = t_start
        self.t_end = t_end
        self.stop_at_exact_time = stop_at_exact_time
        self.escape_radius = escape_radius
        self.min_distance = min_distance

        self.precision_integration_steps = precision_integration_steps
        self.integration_grid = (
            np.array(integration_grid) if integration_grid is not None else None
        )

        self.central_mass = self._param(central_mass)
        if self.central_mass.min is None:
            self.central_mass.min = 0.0

        self.bodies_data = bodies or []
        self.orbit_type = orbit_type
        self.orbit_output_type = orbit_output_type

        self.T0_ref = T0_ref
        self.P_ref = P_ref

        self.compute_xyz = compute_xyz
        self.compute_orbital = compute_orbital

        self.params = {"central_mass": self.central_mass}
        for i, body in enumerate(self.bodies_data):
            prefix = f"b{i + 1}_"
            m_val = body.get("m", 0.0)
            self.params[f"{prefix}m"] = self._param(m_val)

            for element in ["a", "P", "e", "inc", "Omega", "omega", "M", "T"]:
                if element in body:
                    p = self._param(body[element])

                    if element in ["m", "a", "P"] and p.min is None:
                        p.min = 0.0
                    elif element == "e":
                        if p.min is None:
                            p.min = 0.0
                        if p.max is None:
                            p.max = 1.0

                    self.params[f"{prefix}{element}"] = p

    def _setup_rebound(self, params_dict: Dict[str, float]) -> rebound.Simulation:
        sim = rebound.Simulation()
        sim.integrator = self.integrator
        sim.dt = self.dt

        l_unit = self.units.get("l", "au")
        t_unit = self.units.get("t", "day")
        m_unit = self.units.get("m", "msun")
        sim.units = (l_unit, t_unit, m_unit)

        for k, v in self.integrator_params.items():
            setattr(sim, k, v)

        if self.escape_radius:
            sim.exit_max_distance = self.escape_radius
        if self.min_distance:
            sim.exit_min_distance = self.min_distance

        m_central = params_dict.get("central_mass", self.central_mass.value)
        sim.add(m=m_central)

        for i, _ in enumerate(self.bodies_data):
            prefix = f"b{i + 1}_"
            m = params_dict.get(
                f"{prefix}m", self.params.get(f"{prefix}m").value
            )

            orb_params: Dict[str, Any] = {}

            inc_key = f"{prefix}inc"
            if inc_key in params_dict:
                orb_params["inc"] = np.deg2rad(params_dict[inc_key])
            elif inc_key in self.params:
                orb_params["inc"] = np.deg2rad(self.params[inc_key].value)
            else:
                orb_params["inc"] = np.deg2rad(90.0)

            for element in ["a", "P", "e", "Omega", "omega", "M", "T"]:
                key = f"{prefix}{element}"
                if key in params_dict:
                    val = params_dict[key]
                elif key in self.params:
                    val = self.params[key].value
                else:
                    continue

                if val is not None and np.isfinite(val):
                    if element == "e":
                        val = min(max(val, 0.0), 0.99999)

                    if element in ["Omega", "omega", "M"]:
                        val = np.deg2rad(val)

                    if (
                        element == "T"
                        and val > 1_000_000
                        and self.T0_ref != 0
                    ):
                        val = val - self.T0_ref

                    orb_params[element] = val

            if "a" in orb_params and "P" in orb_params:
                raise ValueError(
                    f"Body {i + 1}: 'a' ve 'P' aynı anda verilemez."
                )

            if self.orbit_type == "jacobi":
                sim.add(m=m, **orb_params)
            else:
                sim.add(m=m, primary=sim.particles[0], **orb_params)

        sim.move_to_com()
        return sim

    def integrate(
        self,
        times: np.ndarray,
        params_dict: Optional[Dict[str, float]] = None,
        *,
        compute_orbital: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if params_dict is None:
            params_dict = {k: p.value for k, p in self.params.items()}

        if self.t_start is not None or self.t_end is not None:
            t_lo = self.t_start if self.t_start is not None else -np.inf
            t_hi = self.t_end if self.t_end is not None else np.inf
            times = times[(times >= t_lo) & (times <= t_hi)]

        _compute_orbital = compute_orbital if compute_orbital is not None else self.compute_orbital

        sim = self._setup_rebound(params_dict)
        num_bodies = sim.N
        num_times = len(times)

        outputs: Dict[str, Any] = {
            "D": (
                np.full((num_times, num_bodies, 3), np.nan)
                if self.compute_xyz
                else None
            ),
            "E": (
                np.full((num_times, num_bodies - 1, 7), np.nan)
                if _compute_orbital
                else None
            ),
            "F": True,
            "G": {"delta_E": 0.0, "delta_L": 0.0},
        }

        try:
            E0 = sim.energy()
            L0 = np.linalg.norm(sim.angular_momentum())
        except Exception:
            E0, L0 = 0.0, 0.0

        idx = np.argsort(times)
        sorted_times = times[idx]

        for i, t in enumerate(sorted_times):
            try:
                sim.integrate(t, exact_finish_time=self.stop_at_exact_time)
                orig_i = idx[i]

                if self.compute_xyz:
                    for j in range(num_bodies):
                        p = sim.particles[j]
                        outputs["D"][orig_i, j] = [p.x, p.y, p.z]

                if _compute_orbital:
                    jacobi = self.orbit_output_type == "jacobi"
                    orbits = (
                        sim.orbits(jacobi=True)
                        if jacobi
                        else sim.orbits()
                    )
                    for j in range(1, num_bodies):
                        orb = orbits[j - 1]
                        outputs["E"][orig_i, j - 1] = [
                            orb.a,
                            orb.P,
                            orb.e,
                            np.rad2deg(orb.inc),
                            np.rad2deg(orb.Omega),
                            np.rad2deg(orb.omega),
                            np.rad2deg(orb.M),
                        ]
            except Exception:
                outputs["F"] = False
                continue

        try:
            Ef = sim.energy()
            Lf = np.linalg.norm(sim.angular_momentum())
            outputs["G"]["delta_E"] = (
                (Ef - E0) / E0 if E0 != 0 else Ef
            )
            outputs["G"]["delta_L"] = (
                (Lf - L0) / L0 if L0 != 0 else Lf
            )
        except Exception:
            pass

        return outputs

    def _calculate_etv(
        self, x: np.ndarray, params_float: Dict[str, float]
    ) -> np.ndarray:
        for v in params_float.values():
            if not np.isfinite(v):
                return np.full_like(x, np.nan, dtype=float)

        times = x * self.P_ref

        try:
            if (
                self.integration_grid is not None
                and len(self.integration_grid) > 0
            ):
                res = self.integrate(self.integration_grid, params_float, compute_orbital=False)
                if not res["F"] or res["D"] is None:
                    return np.full_like(x, np.nan, dtype=float)
                grid_z = res["D"][:, 0, 2]
                if np.isnan(grid_z).any():
                    return np.full_like(x, np.nan, dtype=float)
                z_primary = np.interp(times, self.integration_grid, grid_z)

            elif self.precision_integration_steps > 0:
                if len(times) == 0:
                    return np.zeros_like(x, dtype=float)
                grid_times = np.linspace(
                    np.min(times), np.max(times), self.precision_integration_steps
                )
                res = self.integrate(grid_times, params_float, compute_orbital=False)
                if not res["F"] or res["D"] is None:
                    return np.full_like(x, np.nan, dtype=float)
                grid_z = res["D"][:, 0, 2]
                if np.isnan(grid_z).any():
                    return np.full_like(x, np.nan, dtype=float)
                z_primary = np.interp(times, grid_times, grid_z)

            else:
                res = self.integrate(times, params_float, compute_orbital=False)
                if not res["F"] or res["D"] is None:
                    return np.full_like(x, np.nan, dtype=float)
                z_primary = res["D"][:, 0, 2]
                if np.isnan(z_primary).any():
                    return np.full_like(x, np.nan, dtype=float)

            return -z_primary / self._c_light
        except Exception:
            return np.full_like(x, np.nan, dtype=float)

    def model_func(self, x: np.ndarray, **kwargs) -> np.ndarray:
        is_symbolic = False
        if HAS_PYTENSOR:
            for v in kwargs.values():
                if isinstance(v, pt.TensorVariable):
                    is_symbolic = True
                    break

        if is_symbolic:
            op = NewtonianOp(self)
            all_kwargs: Dict[str, Any] = {}
            for k, p in self.params.items():
                all_kwargs[k] = (
                    kwargs[k]
                    if k in kwargs
                    else pt.as_tensor_variable(float(p.value))
                )
            keys = sorted(all_kwargs.keys())
            inputs = [pt.as_tensor_variable(x)] + [all_kwargs[k] for k in keys]
            return op(*inputs)

        params_float: Dict[str, float] = {}
        for k, v in kwargs.items():
            params_float[k] = float(v.value) if hasattr(v, "value") else float(v)

        if "central_mass" not in params_float:
            params_float["central_mass"] = float(self.central_mass.value)

        return self._calculate_etv(x, params_float)


if HAS_PYTENSOR:

    class _NewtonianGradOp(pt.Op):
        EPS = 1e-7

        def __init__(self, model: NewtonianModel, param_keys: List[str]) -> None:
            self.model = model
            self.param_keys = param_keys
            self.n_params = len(param_keys)

        def make_node(self, x, gz, *args):
            x = pt.as_tensor_variable(x)
            gz = pt.as_tensor_variable(gz)
            args = [pt.as_tensor_variable(a) for a in args]
            output_types = [a.type.make_variable() for a in args]
            return pytensor.graph.basic.Apply(
                self, [x, gz] + list(args), output_types
            )

        def perform(self, node, inputs, outputs):
            x = inputs[0]
            gz = inputs[1]
            param_vals = inputs[2:]

            base_params = {
                k: float(v) for k, v in zip(self.param_keys, param_vals)
            }
            f0 = self.model._calculate_etv(x, base_params)

            for i, key in enumerate(self.param_keys):
                perturbed = base_params.copy()
                h = max(abs(base_params[key]) * self.EPS, self.EPS)
                perturbed[key] += h
                f1 = self.model._calculate_etv(x, perturbed)
                df_dp = (f1 - f0) / h
                outputs[i][0] = np.asarray(
                    np.sum(gz * df_dp), dtype=node.outputs[i].dtype
                )

    class NewtonianOp(pt.Op):

        def __init__(self, model: NewtonianModel) -> None:
            self.model = model
            self.param_keys = sorted(model.params.keys())
            self._grad_op = _NewtonianGradOp(model, self.param_keys)

        def make_node(self, x, *args):
            x = pt.as_tensor_variable(x)
            args = [pt.as_tensor_variable(a) for a in args]
            return pytensor.graph.basic.Apply(
                self, [x] + list(args), [x.type.make_variable()]
            )

        def perform(self, node, inputs, outputs):
            x = inputs[0]
            param_vals = inputs[1:]
            params_float = {
                k: float(v) for k, v in zip(self.param_keys, param_vals)
            }
            result = self.model._calculate_etv(x, params_float)
            outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

        def grad(self, inputs, output_gradients):
            gz = output_gradients[0]
            x = inputs[0]
            param_inputs = inputs[1:]
            param_grads = self._grad_op(x, gz, *param_inputs)
            if not isinstance(param_grads, (list, tuple)):
                param_grads = [param_grads]
            return [pytensor.gradient.DisconnectedType()()] + list(param_grads)
