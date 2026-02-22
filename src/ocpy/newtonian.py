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

class NewtonianModel(ModelComponent):
    """
    Newtonian n-body model component using REBOUND for ETV modeling.
    """
    name = "newtonian"

    def __init__(
        self,
        *,
        integrator: str = "ias15",
        dt: float = 0.01,
        integrator_params: Optional[Dict[str, Any]] = None,
        units: Optional[Dict[str, str]] = None,
        
        reference_time: float = 0.0,
        stop_at_exact_time: bool = True,
        escape_radius: Optional[float] = None,
        min_distance: Optional[float] = None,
        
        precision_integration_steps: int = 0,
        integration_grid: Optional[Union[np.ndarray, List[float]]] = None,
        
        central_mass: NumberOrParam = 1.0,
        bodies: Optional[List[Dict[str, Any]]] = None,
        orbit_type: str = "heliocentric", 
        
        T0_ref: float = 0.0,
        P_ref: float = 1.0,

        compute_xyz: bool = True,
        compute_orbital: bool = True,
        name: Optional[str] = None
    ) -> None:
        if name is not None:
            self.name = name
        
        self.integrator = integrator
        self.dt = dt
        self.integrator_params = integrator_params or {}
        
        default_units = {"m": "msun", "t": "day", "l": "au"}
        if units:
            self.units = default_units.copy()
            self.units.update(units)
        else:
            self.units = default_units
        
        self.reference_time = reference_time
        self.stop_at_exact_time = stop_at_exact_time
        self.escape_radius = escape_radius
        self.min_distance = min_distance
        
        self.precision_integration_steps = precision_integration_steps
        self.integration_grid = np.array(integration_grid) if integration_grid is not None else None
        
        self.central_mass = self._param(central_mass)
        if self.central_mass.min is None:
            self.central_mass.min = 0.0
        self.bodies_data = bodies or []
        self.orbit_type = orbit_type
        
        self.T0_ref = T0_ref
        self.P_ref = P_ref

        self.compute_xyz = compute_xyz
        self.compute_orbital = compute_orbital
        
        self.params = {"central_mass": self.central_mass}
        for i, body in enumerate(self.bodies_data):
            prefix = f"b{i+1}_"
            m_val = body.get("m", 0.0)
            self.params[f"{prefix}m"] = self._param(m_val)
            
            for element in ["a", "P", "e", "Omega", "omega", "M", "T"]:
                if element in body:
                    p = self._param(body[element])
                    
                    # Automate bounds if not provided
                    if element in ["m", "a", "P"] and p.min is None:
                        p.min = 0.0
                    elif element == "e":
                        if p.min is None: p.min = 0.0
                        if p.max is None: p.max = 1.0
                    
                    self.params[f"{prefix}{element}"] = p

    def _setup_rebound(self, params_dict: Dict[str, float]) -> rebound.Simulation:
        sim = rebound.Simulation()
        sim.integrator = self.integrator
        sim.dt = self.dt
        
        if self.units:
            sim.units = (self.units.get("l", "au"), self.units.get("t", "yr"), self.units.get("m", "msun"))
        
        for k, v in self.integrator_params.items():
            setattr(sim, k, v)
        
        if self.escape_radius:
            sim.exit_max_distance = self.escape_radius
        if self.min_distance:
            sim.exit_min_distance = self.min_distance
            
        m_central = params_dict.get("central_mass", self.central_mass.value)
        sim.add(m=m_central)
        
        for i, _ in enumerate(self.bodies_data):
            prefix = f"b{i+1}_"
            m = params_dict.get(f"{prefix}m", self.params.get(f"{prefix}m").value)
            
            orb_params = {}
            # Default inclination is 90 degrees (pi/2 radians) for edge-on
            orb_params["inc"] = np.deg2rad(90.0)

            for element in ["a", "P", "e", "Omega", "omega", "M", "T"]:
                key = f"{prefix}{element}"
                if key in params_dict:
                    val = params_dict[key]
                elif key in self.params:
                    val = self.params[key].value
                else:
                    continue
                
                if val is not None:
                    if element in ["Omega", "omega", "M"]:
                        val = np.deg2rad(val)
                    
                    if element == "T" and val > 1_000_000 and self.T0_ref != 0:
                        val = val - self.T0_ref
                        
                    orb_params[element] = val
            
            if "a" in orb_params and "P" in orb_params:
                raise ValueError(f"Body {i+1}: Cannot specify both semi-major axis 'a' and period 'P'.")

            if self.orbit_type == "jacobi":
                sim.add(m=m, **orb_params)
            else: 
                sim.add(m=m, primary=sim.particles[0], **orb_params)
        
        sim.move_to_com()
        return sim

    def integrate(self, times: np.ndarray, params_dict: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Runs the full integration and returns the requested outputs.
        """
        if params_dict is None:
            params_dict = {k: p.value for k, p in self.params.items()}
            
        sim = self._setup_rebound(params_dict)
        num_bodies = sim.N
        num_times = len(times)
        
        outputs = {
            "D": np.full((num_times, num_bodies, 3), np.nan) if self.compute_xyz else None,
            "E": np.full((num_times, num_bodies - 1, 7), np.nan) if self.compute_orbital else None,
            "F": True,
            "G": {"delta_E": 0.0, "delta_L": 0.0}
        }
        
        try:
            E0 = sim.energy()
            L0 = np.linalg.norm(sim.angular_momentum())
        except:
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
                
                if self.compute_orbital:
                    orbits = sim.orbits()
                    for j in range(1, num_bodies):
                        orb = orbits[j-1]
                        outputs["E"][orig_i, j-1] = [
                            orb.a, orb.P, orb.e, 
                            np.rad2deg(orb.inc), np.rad2deg(orb.Omega), 
                            np.rad2deg(orb.omega), np.rad2deg(orb.M)
                        ]
            except Exception as e:
                outputs["F"] = False
                continue
            
        try:
            Ef = sim.energy()
            Lf = np.linalg.norm(sim.angular_momentum())
            outputs["G"]["delta_E"] = (Ef - E0) / E0 if E0 != 0 else Ef
            outputs["G"]["delta_L"] = (Lf - L0) / L0 if L0 != 0 else Lf
        except:
            pass
        
        return outputs

    def _calculate_etv(self, x, params_float) -> np.ndarray:
        """
        Core ETV calculation logic using rebound.
        """
        times = x * self.P_ref
        
        # Check if a custom integration grid is provided
        if self.integration_grid is not None and len(self.integration_grid) > 0:
             # Integrate on the custom grid
            res = self.integrate(self.integration_grid, params_float)
            
            if res["D"] is not None:
                # Extract result on grid
                grid_z = res["D"][:, 0, 2]
                
                # Interpolate to actual observation times
                z_primary = np.interp(times, self.integration_grid, grid_z)
            else:
                return np.zeros_like(x)

        # Check if precision integration is requested via steps
        elif self.precision_integration_steps > 0:
            if len(times) > 0:
                t_min = np.min(times)
                t_max = np.max(times)
                # Create a uniform grid spanning the observation range
                grid_times = np.linspace(t_min, t_max, self.precision_integration_steps)
                
                # Integrate on the grid
                res = self.integrate(grid_times, params_float)
                
                if res["D"] is not None:
                    # Extract result on grid
                    grid_z = res["D"][:, 0, 2]
                    
                    # Interpolate to actual observation times
                    z_primary = np.interp(times, grid_times, grid_z)
                else:
                    return np.zeros_like(x)
            else:
                return np.zeros_like(x)
        else:
            # Standard integration at exact observation times
            res = self.integrate(times, params_float)
            
            if res["D"] is not None:
                z_primary = res["D"][:, 0, 2] 
            else:
                return np.zeros_like(x)
        
        time_unit = self.units.get("t", "yr").lower()
        if time_unit in ["day", "d"]:
            c = 173.1446 
        elif time_unit in ["s", "sec"]:
            c = 0.00200398 
        else: 
            c = 63241.077 
            
        # Flip sign to match analytical LiTE convention
        return -z_primary / c

    def model_func(self, x, **kwargs) -> np.ndarray:
        """
        Function for ocpy fitting. Calculates ETVs (T_obs - T_calc).
        x can be cycle (E) or time (BJD).
        """
        # Check if any input is a symbolic pytensor variable
        is_symbolic = False
        if HAS_PYTENSOR:
            for v in kwargs.values():
                if isinstance(v, pt.TensorVariable):
                    is_symbolic = True
                    break
        
        if is_symbolic:
            # Use NewtonianOp to handle symbolic variables
            op = NewtonianOp(self)
            
            # Ensure all params are present even if not in kwargs (use fixed values)
            all_kwargs = {}
            for k, p in self.params.items():
                if k in kwargs:
                    all_kwargs[k] = kwargs[k]
                else:
                    all_kwargs[k] = pt.as_tensor_variable(float(p.value))
            
            # Sort keys to match Op.perform
            keys = sorted(all_kwargs.keys())
            inputs = [pt.as_tensor_variable(x)] + [all_kwargs[k] for k in keys]
            return op(*inputs)

        params_float = {}
        for k, v in kwargs.items():
            if hasattr(v, "value"):
                params_float[k] = float(v.value)
            else:
                params_float[k] = float(v)
        
        if "central_mass" not in params_float:
            params_float["central_mass"] = float(self.central_mass.value)

        return self._calculate_etv(x, params_float)

if HAS_PYTENSOR:
    class NewtonianOp(pt.Op):
        """
        PyTensor Op wrapper for NewtonianModel.
        """
        def __init__(self, model: NewtonianModel):
            self.model = model
            self.param_keys = sorted(model.params.keys())

        def make_node(self, x, *args):
            x = pt.as_tensor_variable(x)
            args = [pt.as_tensor_variable(a) for a in args]
            # Output is a double vector
            return pytensor.graph.basic.Apply(self, [x] + args, [x.type.make_variable()])

        def perform(self, node, inputs, outputs):
            x = inputs[0]
            param_vals = inputs[1:]
            
            params_float = {k: float(v) for k, v in zip(self.param_keys, param_vals)}
            
            result = self.model._calculate_etv(x, params_float)
            outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

