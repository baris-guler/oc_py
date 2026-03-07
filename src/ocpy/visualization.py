from typing import Optional, List, Union, Tuple, Dict, Literal
import numpy as np
import matplotlib.pyplot as plt
import re
import inspect
import arviz as az
try:
    import corner
except ImportError:
    corner = None

from .oc import Linear, Quadratic, Keplerian, Sinusoidal, Parameter

class Plot:


    @staticmethod
    def plot_data(
        data: "OC", 
        *, 
        ax: Optional[plt.Axes] = None, 
        x_col: str = "cycle", 
        y_col: str = "oc",
        plot_kwargs: Optional[dict] = None
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10.0, 5.4))
        
        
        # ax = ax # Not needed anymore
        
        plot_kwargs = dict(fmt="o", markersize=4.5, color="tab:blue", alpha=0.8, capsize=2, label="Data") | (plot_kwargs or {})

        x_values = np.asarray(data.data[x_col].to_numpy(), dtype=float)
        y_values = np.asarray(data.data[y_col].to_numpy(), dtype=float)
        
        if "labels" in data.data.columns:
            labels_data = data.data["labels"]
            unique_labels = sorted(list(set(labels_data.dropna().unique())))
            
            if len(unique_labels) > 0:
                colormap = plt.get_cmap("tab10")
                for index, label in enumerate(unique_labels):
                    mask = (labels_data == label).to_numpy(dtype=bool)
                    if not np.any(mask): continue
                    
                    color = colormap(index % 10)
                    local_kwargs = plot_kwargs.copy()
                    local_kwargs["color"] = color
                    local_kwargs["label"] = f"Data ({label})"
                    
                    y_error = None
                    if "minimum_time_error" in data.data.columns:
                        y_error = np.asarray(data.data["minimum_time_error"].to_numpy(), dtype=float)[mask]
                    
                    ax.errorbar(x_values[mask], y_values[mask], yerr=y_error, **local_kwargs)
                
                # Unlabeled data
                mask_unlabeled = labels_data.isna().to_numpy(dtype=bool)
                if np.any(mask_unlabeled):
                    local_kwargs = plot_kwargs.copy()
                    local_kwargs["color"] = "gray"
                    local_kwargs["label"] = "Data (unlabeled)"
                    y_error = None
                    if "minimum_time_error" in data.data.columns:
                        y_error = np.asarray(data.data["minimum_time_error"].to_numpy(), dtype=float)[mask_unlabeled]
                    ax.errorbar(x_values[mask_unlabeled], y_values[mask_unlabeled], yerr=y_error, **local_kwargs)
                
                ax.legend()
            else:
                 ax.errorbar(x_values, y_values, yerr=(np.asarray(data.data["minimum_time_error"].to_numpy(), dtype=float) if "minimum_time_error" in data.data.columns else None), **plot_kwargs)
        else:
            y_error = None
            if "minimum_time_error" in data.data.columns:
                y_error = np.asarray(data.data["minimum_time_error"].to_numpy(), dtype=float)
                
            ax.errorbar(x_values, y_values, yerr=y_error, **plot_kwargs)
        
        ax.set_ylabel("O−C")
        ax.set_xlabel(x_col.capitalize())
        ax.grid(True, alpha=0.25)
        
        return ax

    @classmethod
    def plot_model_pymc(
        cls,
        inference_data: az.InferenceData,
        data: "OCPyMC",
        *,
        ax: Optional[plt.Axes] = None,
        x_col: str = "cycle",
        n_points: int = 800,
        sum_kwargs: Optional[dict] = None,
        comp_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        plot_band: bool = True,
        extension_factor: float = 0.05
    ) -> plt.Axes:
        if ax is None:
            ax = plt.gca()


        def split_name(variable_name: str):
            underscore_index = variable_name.rfind("_")
            return (variable_name[:underscore_index], variable_name[underscore_index + 1 :]) if underscore_index != -1 else (None, None)

        def parse_prefix(prefix_str: str):
            match = re.match(r"^([A-Za-z_]+?)(\d+)?$", prefix_str)
            if not match:
                return (prefix_str, 0)
            base = match.group(1)
            index = int(match.group(2)) if match.group(2) is not None else 0
            return (base, index)

        scalars = [variable_name for variable_name, data_array in inference_data.posterior.data_vars.items() if getattr(data_array, "ndim", 0) == 2 and variable_name not in {"y_model", "y_model_dense", "y_obs"}]
        
        if not scalars:
            return ax

        medians_dict: dict[str, float] = {}
        for variable_name in scalars:
            data_array = inference_data.posterior[variable_name]
            value = data_array.median(dim=("chain", "draw")).item()
            medians_dict[variable_name] = float(value)

        groups: dict[str, dict[str, float]] = {}
        for variable_name, value in medians_dict.items():
            prefix, param_name = split_name(variable_name)
            if prefix is None:
                continue
            groups.setdefault(prefix, {})[param_name] = value

        order = sorted(groups.keys(), key=lambda p: parse_prefix(p))
        components = []
        valid_order = []
        
        for prefix in order:
            base, _ = parse_prefix(prefix)
            fields = groups[prefix]
            component = None

            if base in ("linear", "lin"):
                component = Linear(
                    a=Parameter(value=fields.get("a", 0.0), fixed=True),
                    b=Parameter(value=fields.get("b", 0.0), fixed=True)
                )
            elif base in ("quadratic", "quad"):
                component = Quadratic(
                    q=Parameter(value=fields.get("q", 0.0), fixed=True)
                )
            elif base in ("keplerian", "kep", "lite", "LiTE"):
                t0_value = fields.get("T0", fields.get("T", 0.0))
                component = Keplerian(
                    amp=Parameter(value=fields.get("amp", 0.0), fixed=True),
                    e=Parameter(value=fields.get("e", 0.0), fixed=True),
                    omega=Parameter(value=fields.get("omega", 0.0), fixed=True),
                    P=Parameter(value=fields.get("P", 1.0), fixed=True),
                    T0=Parameter(value=t0_value, fixed=True),
                    name=prefix,
                )
            elif base in ("sinusoidal", "sin"):
                component = Sinusoidal(
                    amp=Parameter(value=fields.get("amp", 0.0), fixed=True),
                    P=Parameter(value=fields.get("P", 1.0), fixed=True)
                )
            
            if component is not None:
                components.append(component)
                valid_order.append(prefix)
        
        order = valid_order

        x = np.asarray(data.data[x_col].to_numpy(), dtype=float)
        xmin, xmax = (float(np.min(x)), float(np.max(x))) if x.size else (0.0, 1.0)
        margin = (xmax - xmin) * extension_factor
        xline = np.linspace(xmin - margin, xmax + margin, n_points)

        band = None
        y_total_best = np.zeros_like(xline)

        # 1. Best Fallback: Use y_model_dense if it exists in inference_data
        if "y_model_dense" in inference_data.posterior and "dense_x" in inference_data.posterior:
            y_dense_post = inference_data.posterior["y_model_dense"]
            x_dense_vals = inference_data.posterior["dense_x"].values[0, 0] # Constant across chains/draws
            
            y_fit = y_dense_post.median(dim=("chain", "draw")).values
            
            if not components:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(10.0, 5.4))
                
                fit_color = (plot_kwargs or {}).get("color", "red")
                ax.plot(x_dense_vals, y_fit, color=fit_color, lw=2.6, label="Fit (Median)")
                
                if plot_band:
                    low = y_dense_post.quantile(0.16, dim=("chain", "draw")).values
                    high = y_dense_post.quantile(0.84, dim=("chain", "draw")).values
                    ax.fill_between(x_dense_vals, low, high, color=fit_color, alpha=0.3, linewidth=0, label=r"Uncertainty (1$\sigma$)")
                return ax

        # 2. Secondary Fallback: Interpolate y_model at observation points
        if "y_model" in inference_data.posterior and len(x) == inference_data.posterior["y_model"].shape[-1]:
            y_model_post = inference_data.posterior["y_model"]
            y_total_obs = y_model_post.median(dim=("chain", "draw")).values
            
            # If we reconstructed no components, we use y_model points as the fit line
            if not components:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(10.0, 5.4))
                
                # Handle duplicates and sort using pandas for robustness
                import pandas as pd
                df_temp = pd.DataFrame({'x': x, 'y': y_total_obs})
                
                # Check for uncertainty band data
                if plot_band:
                    df_temp['low'] = y_model_post.quantile(0.16, dim=("chain", "draw")).values
                    df_temp['high'] = y_model_post.quantile(0.84, dim=("chain", "draw")).values
                
                # Group by x and take mean to handle multiple observations at the same cycle
                df_average = df_temp.groupby('x').mean().sort_index()
                xs_clean = df_average.index.values
                ys_clean = df_average['y'].values
                
                fit_color = (plot_kwargs or {}).get("color", "red")
                try:
                    from scipy.interpolate import make_interp_spline
                    # Create a smooth grid for plotting (longer than the data range if requested)
                    x_smooth = np.linspace(xs_clean.min(), xs_clean.max(), 1000)
                    spl = make_interp_spline(xs_clean, ys_clean, k=3)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=fit_color, lw=2.6, label="Fit (Median)")
                    
                    if plot_band:
                        spl_low = make_interp_spline(xs_clean, df_average['low'].values, k=3)
                        spl_high = make_interp_spline(xs_clean, df_average['high'].values, k=3)
                        ax.fill_between(x_smooth, spl_low(x_smooth), spl_high(x_smooth), color=fit_color, alpha=0.3, linewidth=0, label=r"Uncertainty (1$\sigma$)")
                except Exception:
                    # Fallback to simple line if scipy.interpolate fails
                    ax.plot(xs_clean, ys_clean, color=fit_color, lw=2.6, label="Fit (Median)")
                    if plot_band:
                        ax.fill_between(xs_clean, df_average['low'].values, df_average['high'].values, color=fit_color, alpha=0.3, linewidth=0, label=r"Uncertainty (1$\sigma$)")
                
                return ax

        if plot_band and components:
            subset = az.extract(inference_data, num_samples=200)
            y_samples = []
            n_draws = subset.sample.size
            
            for sample_index in range(n_draws):
                y_total = np.zeros_like(xline)
                for index, prefix in enumerate(order):
                    component = components[index]
                    kwargs = {}
                    for param_name in groups[prefix].keys():
                        variable_name = f"{prefix}_{param_name}"
                        if variable_name in subset:
                            value = subset[variable_name].values[sample_index]
                            kwargs[param_name] = float(value)
                    y_total += component.model_func(xline, **kwargs)
                y_samples.append(y_total)
            
            y_samples = np.array(y_samples)
            low = np.percentile(y_samples, 16, axis=0)
            high = np.percentile(y_samples, 84, axis=0)
            band = (xline, low, high)

        return cls.plot_model_components(
            components,
            xline,
            ax=ax,
            sum_kwargs=sum_kwargs,
            comp_kwargs=comp_kwargs,
            uncertainty_band=band
        )

    @classmethod
    def plot_model_lmfit(
        cls,
        result,
        data: "OCLMFit",
        *,
        ax: Optional[plt.Axes] = None,
        x_col: str = "cycle",
        n_points: int = 500,
        plot_kwargs: Optional[dict] = None,
        extension_factor: float = 0.05
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10.0, 5.4))

        x = np.asarray(data.data[x_col].to_numpy(), dtype=float)
        xmin, xmax = (float(np.min(x)), float(np.max(x))) if x.size else (0.0, 1.0)
        margin = (xmax - xmin) * extension_factor
        x_dense = np.linspace(xmin - margin, xmax + margin, n_points)
        y_fit_dense = result.eval(x=x_dense)

        plot_kwargs = dict(color="red", label="Fit", zorder=3) | (plot_kwargs or {})
        
        try:
            dely = result.eval_uncertainty(x=x_dense, sigma=1)
            ax.fill_between(x_dense, y_fit_dense - dely, y_fit_dense + dely, color="red", alpha=0.3, linewidth=0, label=r"Uncertainty (1$\sigma$)", zorder=2)
        except Exception:
            pass

        ax.plot(x_dense, y_fit_dense, **plot_kwargs)
        
        return ax

    @classmethod
    def plot_model_components(
        cls,
        model_components: list,
        xline: np.ndarray,
        *,
        ax: Optional[plt.Axes] = None,
        sum_kwargs: Optional[dict] = None,
        comp_kwargs: Optional[dict] = None,
        uncertainty_band: Optional[tuple] = None
    ) -> plt.Axes:

        def _comp_name(comp):
            return getattr(comp, "name", comp.__class__.__name__.lower())

        def _sig_param_names(comp):
            sig = inspect.signature(comp.model_func)
            params = list(sig.parameters.values())[1:] # skip 'x'
            names = [p.name for p in params if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
            # If the function uses **kwargs, we assume it takes everything in comp.params
            if any(p.kind == p.VAR_KEYWORD for p in params):
                names = list(getattr(comp, "params", {}).keys())
            return names

        def _param_value(v):
            return getattr(v, "value", v)

        def _eval_component(comp, xvals):
            pnames = _sig_param_names(comp)
            params_dict = getattr(comp, "params", {}) or {}
            kwargs = {}
            for pname in pnames:
                if pname not in params_dict:
                    raise KeyError(f"Component '{_comp_name(comp)}' missing parameter '{pname}'")
                kwargs[pname] = float(_param_value(params_dict[pname]))
            return comp.model_func(xvals, **kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10.0, 5.4))

        sum_color = (sum_kwargs or {}).get("color", "red")
        sum_kwargs = dict(lw=2.6, alpha=0.95, label="Sum of selected components", color=sum_color, zorder=3) | (sum_kwargs or {})
        comp_kwargs = dict(lw=1.5, alpha=0.9, linestyle="--") | (comp_kwargs or {})

        comp_curves = []
        for comp in model_components:
            y_comp = _eval_component(comp, xline)
            comp_curves.append((comp, y_comp))
        y_sum = np.sum([yc for _, yc in comp_curves], axis=0) if comp_curves else np.zeros_like(xline)

        if uncertainty_band is not None:
            bx, blow, bhigh = uncertainty_band
            ax.fill_between(bx, blow, bhigh, color=sum_color, alpha=0.3, linewidth=0, label=r"Uncertainty (1$\sigma$)", zorder=2)

        ax.plot(xline, y_sum, **sum_kwargs)
        for component, y_comp in comp_curves:
            ax.plot(xline, y_comp, label=_comp_name(component), **comp_kwargs)
        
        return ax

    @classmethod
    def plot(
        cls,
        data: "OC",
        model: Union["InferenceData", "ModelResult", List["ModelComponent"]] = None,
        *,
        ax: Optional[plt.Axes] = None,
        res_ax: Optional[plt.Axes] = None,
        res: bool = True,
        title: Optional[str] = None,
        x_col: str = "cycle",
        y_col: str = "oc",
        fig_size: tuple = (10, 7),
        plot_kwargs: Optional[dict] = None,
        extension_factor: float = 0.05
    ) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
        x = np.asarray(data.data[x_col].to_numpy(), dtype=float)
        y = np.asarray(data.data[y_col].to_numpy(), dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        yerr = None
        if "minimum_time_error" in data.data.columns:
            yerr = np.asarray(data.data["minimum_time_error"].to_numpy(), dtype=float)
            yerr_clean = yerr[mask] if yerr is not None else None
        
        labels = data.data.get("labels", None)
        labels_clean = labels[mask] if labels is not None else None

        # ax_main = ax
        # res_ax_internal = res_ax
        
        if ax is None:
            if model is not None and res:
                fig, (ax, res_ax) = plt.subplots(2, 1, figsize=fig_size, sharex=True, 
                                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.04})
            else:
                fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]*0.75))
                res_ax = None
        else:
            if res and res_ax is None:
                res = False
        
        cls.plot_data(data, ax=ax, x_col=x_col, y_col=y_col, plot_kwargs=plot_kwargs)

        def _plot_resid(ax_r, x_r, resid_r, yerr_r, labels_r):
             # scatter_kwargs = dict(fmt='o', markersize=3, alpha=0.8, elinewidth=0.8, capsize=1) # This was unused
             
             resid_kwargs = dict(fmt='o', markersize=3, alpha=0.8, elinewidth=0.8, capsize=1)

             if labels_r is not None:
                 unique_labels = sorted(list(set(labels_r.dropna().unique())))
                 if len(unique_labels) > 0:
                     cmap = plt.get_cmap("tab10")
                     for i, lbl in enumerate(unique_labels):
                          m = (labels_r == lbl).to_numpy(dtype=bool)
                          if not np.any(m): continue
                          c = cmap(i % 10)
                          ax_r.errorbar(x_r[m], resid_r[m], yerr=(yerr_r[m] if yerr_r is not None else None), color=c, **resid_kwargs)
                     
                     # Check for unlabeled data (NaN in labels)
                     m_nan = labels_r.isna().to_numpy(dtype=bool)
                     if np.any(m_nan):
                          ax_r.errorbar(x_r[m_nan], resid_r[m_nan], yerr=(yerr_r[m_nan] if yerr_r is not None else None), color="gray", **resid_kwargs)
                     return

             ax_r.errorbar(x_r, resid_r, yerr=yerr_r, color="tab:blue", **resid_kwargs)

        if model is not None:
             is_pymc = hasattr(model, "posterior")
             is_lmfit = hasattr(model, "eval")
             is_list = isinstance(model, (list, tuple))
             is_component = hasattr(model, "model_func") and hasattr(model, "params")

             if is_component:
                 model = [model]
                 is_list = True
             
             if is_pymc:
                 cls.plot_model_pymc(inference_data=model, data=data, ax=ax, x_col=x_col, plot_kwargs=plot_kwargs, extension_factor=extension_factor)
                 if res and res_ax is not None:
                     y_model_post = model.posterior["y_model"]
                     yfit = y_model_post.median(dim=("chain", "draw")).values
                     if yfit.shape == y.shape:
                        resid = y - yfit
                        _plot_resid(res_ax, x, resid, yerr, labels)
                        res_ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6)
             elif is_lmfit:
                 cls.plot_model_lmfit(result=model, data=data, ax=ax, x_col=x_col, plot_kwargs=plot_kwargs, extension_factor=extension_factor)
                 if res and res_ax is not None:
                     y_fit_at_x = model.eval(x=x_clean)
                     resid = y_clean - y_fit_at_x
                     _plot_resid(res_ax, x_clean, resid, yerr_clean, labels_clean)
                     res_ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6)
             elif is_list:
                 xmin, xmax = (float(np.min(x)), float(np.max(x))) if x.size else (0.0, 1.0)
                 margin = (xmax - xmin) * extension_factor
                 xline = np.linspace(xmin - margin, xmax + margin, 800)
                 cls.plot_model_components(model, xline=xline, ax=ax)
                 
                 if res and res_ax is not None:
                     y_model_at_obs = np.zeros_like(x)
                     # ... internal logic omitted for brevity, but I need to make sure I don't break it
                     def _sig_param_names(comp):
                        sig = inspect.signature(comp.model_func)
                        params = list(sig.parameters.values())[1:]
                        names = [p.name for p in params if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
                        if any(p.kind == p.VAR_KEYWORD for p in params):
                            names = list(getattr(comp, "params", {}).keys())
                        return names
                     def _param_value(v):
                        return getattr(v, "value", v)
                        
                     for comp in model:
                         pnames = _sig_param_names(comp)
                         params_dict = getattr(comp, "params", {}) or {}
                         kwargs = {}
                         for pname in pnames:
                             if pname in params_dict:
                                 kwargs[pname] = float(_param_value(params_dict[pname]))
                         y_model_at_obs += comp.model_func(x, **kwargs)
                     
                     resid = y - y_model_at_obs
                     _plot_resid(res_ax, x, resid, yerr, labels)
                     res_ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6)

        if res_ax:
            res_ax.set_ylabel("Resid")
            res_ax.set_xlabel(x_col.capitalize())
            res_ax.grid(True, alpha=0.25)
            ax.set_xlabel("")
        
        if title:
            ax.set_title(title)
            
        ax.legend(loc="best")
        if ax is None: # If we created the figure internally
            if res_ax is None: # If no residuals subplot was created
                try:
                    fig.tight_layout()
                except Exception:
                    pass
        
        return ax

    @staticmethod
    def _format_label(name: str, unit: Optional[str] = None) -> str:

        
        mapping = {
            "omega": r"$\omega$",
            "e": r"$e$",
            "P": r"$P$",
            "T0": r"$T_0$",
            "T": r"$T$",
            "m": r"$m$",
            "a": r"$a$",
            "b": r"$b$",
            "q": r"$q$",
            "sigma": r"$\sigma$",
            "gamma": r"$\gamma$",
            "tau": r"$\tau$",
            "amp": r"$A$"
        }
        
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in mapping:
             sym = mapping[parts[1]]
             formatted = sym 
             pre = parts[0]
             m = re.match(r".*?(\d+)$", pre)
             if m:
                 formatted = fr"{sym}_{{{m.group(1)}}}"
             else:
                 pass
        elif name in mapping:
            formatted = mapping[name]
        else:
            formatted = name
            
        if unit:
            return fr"{formatted} [{unit}]"
        return formatted

    @staticmethod
    def plot_corner(
        inference_data: az.InferenceData, 
        var_names: Optional[List[str]] = None, 
        cornerstyle: Literal["corner", "arviz"] = "corner", 
        units: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        
        if var_names is None:
            variable_candidates = [var_name for var_name in inference_data.posterior.data_vars
                            if getattr(inference_data.posterior[var_name], "ndim", 0) == 2
                            and var_name not in {"y_model", "y_model_dense", "y_obs", "dense_x"}]
        else:
            variable_candidates = var_names

        selected_variables = []
        for var_name in variable_candidates:
            values_array = inference_data.posterior[var_name].values
            # Check if there's actual variation. ptp is peak-to-peak (max - min)
            if np.ptp(values_array) > 1e-12: 
                selected_variables.append(var_name)
        
        if not selected_variables:
             # If everything is fixed, we can't really do a corner plot, 
             # but let's at least not crash or warn cryptically.
             if variable_candidates:
                  selected_variables = [variable_candidates[0]]
             else:
                  raise ValueError("No suitable parameters found for corner plot.")
        
        if cornerstyle == "corner":
            if corner is None:
                raise ImportError("Corner plot requires 'corner' library. Please install it with `pip install corner`.")

            extracted_samples = az.extract(inference_data, var_names=selected_variables)
            samples = np.vstack([extracted_samples[var_name].values for var_name in selected_variables]).T

            # Map the 'range' list if its length doesn't match selected_variables
            if "range" in kwargs and isinstance(kwargs["range"], (list, np.ndarray)):
                range_list = list(kwargs["range"])
                if len(range_list) != len(selected_variables):
                    all_variables = list(inference_data.posterior.data_vars)
                    if len(range_list) == len(all_variables):
                        indices = [all_variables.index(v) for v in selected_variables]
                        kwargs["range"] = [range_list[idx] for idx in indices]
                    elif len(range_list) == len(variable_candidates):
                        indices = [variable_candidates.index(v) for v in selected_variables]
                        kwargs["range"] = [range_list[idx] for idx in indices]
                    elif len(set(range_list)) == 1:
                        # If all values are the same, corner accepts a single float
                        kwargs["range"] = range_list[0]

            # Calculate medians for truth lines
            medians = [float(inference_data.posterior[var_name].median(dim=("chain", "draw"))) for var_name in selected_variables]

            plot_labels = [Plot._format_label(var_name, (units or {}).get(var_name)) for var_name in selected_variables]

            if "quantiles" not in kwargs:
                kwargs["quantiles"] = [0.16, 0.5, 0.84]
            if "show_titles" not in kwargs:
                kwargs["show_titles"] = True
            if "title_fmt" not in kwargs:
                kwargs["title_fmt"] = ".4f"
            
            # Add truths if not already provided
            if "truths" not in kwargs:
                kwargs["truths"] = medians
                kwargs.setdefault("truth_color", "red")
            
            figure = corner.corner(samples, labels=plot_labels, **kwargs)
            return figure
            
        elif cornerstyle == "arviz":
            if "marginals" not in kwargs:
                kwargs["marginals"] = True
            if "kind" not in kwargs:
                kwargs["kind"] = "kde"
            
            with az.rc_context({"plot.max_subplots": 200}):
                return az.plot_pair(inference_data, var_names=selected_variables, **kwargs)
        else:
            raise ValueError(f"Unknown cornerstyle: {cornerstyle}. Use 'corner' or 'arviz'.")

    @staticmethod
    def plot_trace(inference_data, var_names=None, **kwargs):
        
        if var_names is None:
             # Only take variables with 2 dimensions (chain, draw) - these are usually parameters
             variable_candidates = [var_name for var_name in inference_data.posterior.data_vars 
                           if var_name not in {"y_model", "y_model_dense", "y_obs", "dense_x"}
                           and getattr(inference_data.posterior[var_name], "ndim", 0) == 2]
        else:
             variable_candidates = var_names

        selected_variables = []
        for var_name in variable_candidates:
            values_array = inference_data.posterior[var_name].values
            # Exclude variables with near-zero variance (e.g. fixed parameters)
            if values_array.std() > 1e-11:
                selected_variables.append(var_name)
        
        if not selected_variables:
             selected_variables = variable_candidates

        axes = az.plot_trace(inference_data, var_names=selected_variables, **kwargs)
        
        try:
            fig = axes.flatten()[0].figure
            fig.tight_layout()
        except Exception:
            pass
            
        return axes
