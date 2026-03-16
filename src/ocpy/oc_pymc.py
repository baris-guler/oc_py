from __future__ import annotations
from typing import Dict, List, Optional, Literal
import warnings

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt

from .oc import OC, Linear, Quadratic, Keplerian, Sinusoidal, Parameter, ModelComponent
from .visualization import Plot


class OCPyMC(OC):
    math = pm.math

    def _to_param(self, x, *, default: float = 0.0, min_: float | None = None, max_: float | None = None, fixed: bool = False, std: float | None = None) -> Parameter:
        if isinstance(x, Parameter):
            return x
        return Parameter(value=default if x is None else x, min=min_, max=max_, fixed=fixed, std=std)

    def fit(
        self, 
        model_components: List[ModelComponent], 
        *, 
        draws: int = 2000, 
        tune: int = 2000, 
        chains: int = 4, 
        cores: Optional[int] = None,
        target_accept: Optional[float] = None, 
        random_seed: Optional[int] = None, 
        progressbar: bool = True, 
        return_model: bool = False,
        **kwargs
    ) -> az.InferenceData | pm.Model:
        
        x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
        y = np.asarray(self.data["oc"].to_numpy(), dtype=float)
        sigma_i = np.asarray(self.data["minimum_time_error"].to_numpy(), dtype=float)

        if np.isnan(sigma_i).any():
            raise ValueError("Found NaN in 'minimum_time_error'.")

        for c in model_components:
            if hasattr(c, "set_math"):
                c.set_math(self.math)

        def _rv(name: str, par: Parameter):
            val = float(getattr(par, "value", 0.0) or 0.0)
            sd = getattr(par, "std", None)
            lo = getattr(par, "min", None)
            hi = getattr(par, "max", None)
            fix = bool(getattr(par, "fixed", False))

            if fix:

                return pm.Deterministic(name, pt.as_tensor_variable(val))

            if sd is None or sd <= 0:
                sd = max(abs(val) * 0.1, 1e-6)

            if (lo is not None and np.isfinite(lo)) or (hi is not None and np.isfinite(hi)):
                lower = float(lo) if lo is not None else None
                upper = float(hi) if hi is not None else None
                
                safe_val = val
                eps = 1e-5
                if lower is not None and safe_val <= lower:
                    safe_val = lower + eps
                if upper is not None and safe_val >= upper:
                    safe_val = upper - eps
                    
                return pm.TruncatedNormal(name, mu=val, sigma=float(sd), lower=lower, upper=upper, initval=safe_val)
            
            return pm.Normal(name, mu=val, sigma=float(sd), initval=val)

        with pm.Model() as model:
            base_names = [getattr(c, 'name', c.__class__.__name__.lower()) for c in model_components]
            counts = {name: base_names.count(name) for name in base_names}
            seen = {name: 0 for name in base_names}
            
            prefixes = []
            for name in base_names:
                seen[name] += 1
                if counts[name] > 1:
                    prefixes.append(f"{name}{seen[name]}_")
                else:
                    prefixes.append(f"{name}_")
            comp_rvs = {}

            for comp, pref in zip(model_components, prefixes):
                rvs = {}
                for pname, par in getattr(comp, "params", {}).items():
                    rvs[pname] = _rv(pref + pname, par)
                comp_rvs[pref] = rvs

            mus = []
            for comp, pref in zip(model_components, prefixes):
                mus.append(comp.model_func(x, **comp_rvs[pref]))
            
            mu_total = mus[0] if len(mus) == 1 else sum(mus)

            pm.Deterministic("y_model", mu_total)
            pm.Normal("y_obs", mu=mu_total, sigma=sigma_i, observed=y)

            has_expensive = any(getattr(c, "_expensive", False) for c in model_components)

            if not has_expensive:
                xmin, xmax = np.min(x), np.max(x)
                margin = (xmax - xmin) * 0.05
                dense_x_vals = np.linspace(xmin - margin, xmax + margin, 500)

                mus_dense = []
                for comp, pref in zip(model_components, prefixes):
                    mus_dense.append(comp.model_func(dense_x_vals, **comp_rvs[pref]))

                mu_total_dense = mus_dense[0] if len(mus_dense) == 1 else sum(mus_dense)
                pm.Deterministic("y_model_dense", mu_total_dense)
                pm.Deterministic("dense_x", pt.as_tensor_variable(dense_x_vals))

            if return_model:
                return model

            # Pack sample arguments
            sample_kwargs = kwargs.copy()

            if cores is not None:
                sample_kwargs["cores"] = min(cores, chains)
            elif "cores" not in sample_kwargs:
                sample_kwargs["cores"] = chains

            if target_accept is not None:
                sample_kwargs["target_accept"] = target_accept

            if has_expensive and "step" not in sample_kwargs:
                sample_kwargs["step"] = pm.DEMetropolisZ()

            if "step" in sample_kwargs and callable(sample_kwargs["step"]) and not isinstance(sample_kwargs["step"], pm.step_methods.arraystep.ArrayStep):
                sample_kwargs["step"] = sample_kwargs["step"]()

            inference_data = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                random_seed=random_seed, 
                return_inferencedata=True, 
                progressbar=progressbar,
                **sample_kwargs
            )

        inference_data.attrs["_model_components"] = model_components
        inference_data.attrs["_model_prefixes"] = prefixes

        return inference_data

    def clean(
        self, 
        inference_data: az.InferenceData, 
        drop_chains: int = 0, 
        filter_outliers: bool = True, 
        iqr_multiplier: float = 4.0
    ) -> az.InferenceData:
        posterior_data = inference_data.posterior
        chain_coords = posterior_data.coords["chain"].values
        chains_to_keep = list(chain_coords)
        
        if drop_chains > 0:
            var_names = [var_name for var_name in posterior_data.data_vars if getattr(posterior_data[var_name], "ndim", 0) == 2 and var_name not in {"y_model", "y_model_dense", "y_obs", "dense_x"}]
            if drop_chains >= len(chain_coords):
                raise ValueError("drop_chains must be less than the total number of chains.")

            chain_distances = []
            for chain in chain_coords:
                distance = 0.0
                for var_name in var_names:
                    overall_median = float(posterior_data[var_name].median())
                    chain_median = float(posterior_data[var_name].sel(chain=chain).median())
                    std = float(posterior_data[var_name].std())
                    if std > 1e-10:
                         distance += abs(chain_median - overall_median) / std
                chain_distances.append((chain, distance))
            
            chain_distances.sort(key=lambda x: x[1], reverse=True)
            chains_to_drop = [chain for chain, dist_val in chain_distances[:drop_chains]]
            chains_to_keep = [chain for chain in chain_coords if chain not in chains_to_drop]
            posterior_sub = posterior_data.sel(chain=chains_to_keep)
        else:
            posterior_sub = posterior_data
            
        mask = None
        if filter_outliers:
            var_names = [var_name for var_name in posterior_sub.data_vars if getattr(posterior_sub[var_name], "ndim", 0) == 2 and var_name not in {"y_model", "y_model_dense", "y_obs", "dense_x"}]
            stacked = posterior_sub.stack(sample=("chain", "draw"))
            mask = np.ones(stacked.sizes["sample"], dtype=bool)
            
            for var_name in var_names:
                values_array = stacked[var_name].values
                quartile_1 = np.percentile(values_array, 25)
                quartile_3 = np.percentile(values_array, 75)
                interquartile_range = quartile_3 - quartile_1
                if interquartile_range > 1e-10:
                    lower_bound = quartile_1 - iqr_multiplier * interquartile_range
                    upper_bound = quartile_3 + iqr_multiplier * interquartile_range
                    mask = mask & (values_array >= lower_bound) & (values_array <= upper_bound)
                    
        new_groups = {}
        for group_name in inference_data._groups:
            group_dataset = getattr(inference_data, group_name)
            if "chain" in group_dataset.dims and "draw" in group_dataset.dims:
                if drop_chains > 0:
                    try:
                        group_dataset = group_dataset.sel(chain=chains_to_keep)
                    except KeyError:
                        pass
                
                if filter_outliers and mask is not None:
                    try:
                        stacked = group_dataset.stack(sample=("chain", "draw"))
                        filtered = stacked.isel(sample=mask)
                        n_draws = filtered.sizes["sample"]
                        
                        # Prepare the dataset for dimensionality reshaping cleanly
                        filtered = filtered.drop_vars(["chain", "draw", "sample"], errors="ignore")
                        filtered = filtered.rename({"sample": "draw"})
                        filtered = filtered.assign_coords({"draw": np.arange(n_draws)})
                        
                        # Re-add chain dimension
                        group_dataset = filtered.expand_dims({"chain": [0]}).transpose("chain", "draw", ...)
                    except Exception as error_msg:
                        warnings.warn(f"clean() encountered an issue on {group_name}: {error_msg}")
                        
            new_groups[group_name] = group_dataset
            
        cleaned = az.InferenceData(**new_groups)
        for attr_key in ("_model_components", "_model_prefixes"):
            if attr_key in getattr(inference_data, "attrs", {}):
                cleaned.attrs[attr_key] = inference_data.attrs[attr_key]
        return cleaned


    def residue(self, inference_data: az.InferenceData, *, x_col: str = "cycle", y_col: str = "oc") -> "OCPyMC":
        y_model = inference_data.posterior["y_model"]
        y_fit = y_model.median(dim=("chain", "draw")).values
        
        return OCPyMC(
            minimum_time=self.data["minimum_time"].to_list() if "minimum_time" in self.data else None,
            minimum_time_error=self.data["minimum_time_error"].to_list() if "minimum_time_error" in self.data else None,
            weights=self.data["weights"].to_list() if "weights" in self.data else None,
            minimum_type=self.data["minimum_type"].to_list() if "minimum_type" in self.data else None,
            labels=self.data["labels"].to_list() if "labels" in self.data else None,
            cycle=self.data["cycle"].to_list() if "cycle" in self.data else None,
            oc=(self.data[y_col].to_numpy(dtype=float) - y_fit).tolist() if y_col in self.data else None,
        )

    def fit_linear(self, *, a: float | Parameter | None = None, b: float | Parameter | None = None, cores: Optional[int] = None, **kwargs):
        lin = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([lin], cores=cores, **kwargs)

    def fit_quadratic(self, *, q: float | Parameter | None = None, cores: Optional[int] = None, **kwargs) -> az.InferenceData:
        comp = Quadratic(q=self._to_param(q, default=0.0))
        return self.fit([comp], cores=cores, **kwargs)

    def fit_sinusoidal(self, *, amp: float | Parameter | None = None, P: float | Parameter | None = None, cores: Optional[int] = None, **kwargs) -> az.InferenceData:
        comp = Sinusoidal(amp=self._to_param(amp, default=1e-3), P=self._to_param(P, default=1000.0))
        return self.fit([comp], cores=cores, **kwargs)



    def fit_keplerian(self, *, amp: float | Parameter | None = None, e: float | Parameter | None = None, omega: float | Parameter | None = None, P: float | Parameter | None = None, T0: float | Parameter | None = None, name: Optional[str] = None, cores: Optional[int] = None, **kwargs) -> az.InferenceData:
        comp = Keplerian(
            amp=self._to_param(amp, default=0.001),
            e=self._to_param(e, default=0.1),
            omega=self._to_param(omega, default=90.0),
            P=self._to_param(P, default=1000.0),
            T0=self._to_param(T0, default=0.0),
            name=name or "keplerian1",
        )
        return self.fit([comp], cores=cores, **kwargs)

    def fit_lite(self, **kwargs) -> az.InferenceData:
        return self.fit_keplerian(**kwargs)
    
    def fit_parabola(self, *, q: float | Parameter | None = None, a: float | Parameter | None = None, b: float | Parameter | None = None, cores: Optional[int] = None, **kwargs) -> az.InferenceData:
        quad = Quadratic(q=self._to_param(q, default=0.0))
        lin  = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([quad, lin], cores=cores, **kwargs)

    def corner(self, inference_data: az.InferenceData, cornerstyle: Literal["corner", "arviz"] = "corner", units: Optional[Dict[str, str]] = None, **kwargs):
        return Plot.plot_corner(inference_data, cornerstyle=cornerstyle, units=units, **kwargs)

    def trace(self, inference_data: az.InferenceData, **kwargs):
        return Plot.plot_trace(inference_data, **kwargs)
