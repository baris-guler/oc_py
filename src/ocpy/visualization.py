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


    def plot_data(
        data: "OC", 
        *, 
        ax=None, 
        x_col: str = "cycle", 
        y_col: str = "oc",
        plot_kwargs: Optional[dict] = None
    ):
        draw_ax = ax
        if draw_ax is None:
            fig, draw_ax = plt.subplots(figsize=(10.0, 5.4))

        x = np.asarray(data.data[x_col].to_numpy(), dtype=float)
        y = np.asarray(data.data[y_col].to_numpy(), dtype=float)
        
        yerr = None
        if "minimum_time_error" in data.data.columns:
             yerr = np.asarray(data.data["minimum_time_error"].to_numpy(), dtype=float)

        user_kwargs = (plot_kwargs or {}).copy()
        if "s" in user_kwargs:
            user_kwargs["markersize"] = user_kwargs.pop("s")
        if "c" in user_kwargs:
            user_kwargs["color"] = user_kwargs.pop("c")

        default_kwargs = dict(fmt='o', markersize=3, alpha=0.8, elinewidth=0.8, capsize=1, zorder=1)
        plot_kwargs = default_kwargs | user_kwargs

        labels = data.data.get("labels", None)
        if labels is not None:
            unique_labels = sorted(list(set(labels.dropna().unique())))
            if len(unique_labels) > 0:
                cmap = plt.get_cmap("tab10")
                for i, lbl in enumerate(unique_labels):
                    mask = (labels == lbl).to_numpy(dtype=bool)
                    if not np.any(mask):
                        continue
                    
                    color = cmap(i % 10)
                    

                    
                    label_kwargs = plot_kwargs.copy()
                    label_kwargs.update({"color": color, "label": str(lbl)})
                    
                    # Filter data
                    xi = x[mask]
                    yi = y[mask]
                    yerri = yerr[mask] if yerr is not None else None
                    
                    draw_ax.errorbar(xi, yi, yerr=yerri, **label_kwargs)
                
                mask_nan = labels.isna().to_numpy(dtype=bool)
                if np.any(mask_nan):
                     label_kwargs = plot_kwargs.copy()
                     label_kwargs.update({"color": "gray", "label": "Unlabeled"})
                     draw_ax.errorbar(x[mask_nan], y[mask_nan], yerr=(yerr[mask_nan] if yerr is not None else None), **label_kwargs)

                draw_ax.legend()
            else:
                 plot_kwargs.setdefault("color", "tab:blue")
                 draw_ax.errorbar(x, y, yerr=yerr, **plot_kwargs)
        else:
            plot_kwargs.setdefault("color", "tab:blue")
            draw_ax.errorbar(x, y, yerr=yerr, **plot_kwargs)
        
        draw_ax.set_ylabel("O−C")
        draw_ax.set_xlabel(x_col.capitalize())
        draw_ax.grid(True, alpha=0.25)
        
        return draw_ax

    @classmethod
    def plot_model_pymc(
        cls,
        idata,
        data: "OCPyMC",
        *,
        ax=None,
        x_col: str = "cycle",
        n_points: int = 800,
        sum_kwargs: Optional[dict] = None,
        comp_kwargs: Optional[dict] = None,
        plot_band: bool = True,
        extension_factor: float = 0.05
    ):

        def split_name(vn: str):
            i = vn.rfind("_")
            return (vn[:i], vn[i + 1 :]) if i != -1 else (None, None)

        def parse_prefix(pref: str):
            m = re.match(r"^([A-Za-z_]+?)(\d+)?$", pref)
            if not m:
                return (pref, 0)
            base = m.group(1)
            idx = int(m.group(2)) if m.group(2) is not None else 0
            return (base, idx)

        scalars = [vn for vn, da in idata.posterior.data_vars.items() if getattr(da, "ndim", 0) == 2 and vn not in {"y_model", "y_model_dense", "y_obs"}]
        
        if not scalars:
            return ax

        med: dict[str, float] = {}
        for vn in scalars:
            da = idata.posterior[vn]
            val = da.median(dim=("chain", "draw")).item()
            med[vn] = float(val)

        groups: dict[str, dict[str, float]] = {}
        for vn, val in med.items():
            pref, pname = split_name(vn)
            if pref is None:
                continue
            groups.setdefault(pref, {})[pname] = val

        order = sorted(groups.keys(), key=lambda p: parse_prefix(p))
        comps = []
        
        for pref in order:
            base, _ = parse_prefix(pref)
            fields = groups[pref]

            if base == "linear":
                comps.append(Linear(
                    a=Parameter(value=fields.get("a", 0.0), fixed=True),
                    b=Parameter(value=fields.get("b", 0.0), fixed=True)
                ))
            elif base == "quadratic":
                comps.append(Quadratic(
                    q=Parameter(value=fields.get("q", 0.0), fixed=True)
                ))
            elif base in ("keplerian", "kep", "lite", "LiTE"):
                t0_val = fields.get("T0", fields.get("T", 0.0))
                comps.append(Keplerian(
                    amp=Parameter(value=fields.get("amp", 0.0), fixed=True),
                    e=Parameter(value=fields.get("e", 0.0), fixed=True),
                    omega=Parameter(value=fields.get("omega", 0.0), fixed=True),
                    P=Parameter(value=fields.get("P", 1.0), fixed=True),
                    T0=Parameter(value=t0_val, fixed=True),
                    name=pref,
                ))
            elif base == "sinusoidal":
                comps.append(Sinusoidal(
                    amp=Parameter(value=fields.get("amp", 0.0), fixed=True),
                    P=Parameter(value=fields.get("P", 1.0), fixed=True)
                ))

        x = np.asarray(data.data[x_col].to_numpy(), dtype=float)
        xmin, xmax = (float(np.min(x)), float(np.max(x))) if x.size else (0.0, 1.0)
        margin = (xmax - xmin) * extension_factor
        xline = np.linspace(xmin - margin, xmax + margin, n_points)

        band = None
        if plot_band:
            subset = az.extract(idata, num_samples=200)
            y_samples = []
            n_draws = subset.sample.size
            
            for s in range(n_draws):
                y_total = np.zeros_like(xline)
                for i, pref in enumerate(order):
                    comp = comps[i]
                    kwargs = {}
                    for pname in groups[pref].keys():
                        vn = f"{pref}_{pname}"
                        if vn in subset:
                            val = subset[vn].values[s]
                            kwargs[pname] = float(val)
                    y_total += comp.model_func(xline, **kwargs)
                y_samples.append(y_total)
            
            y_samples = np.array(y_samples)
            low = np.percentile(y_samples, 16, axis=0)
            high = np.percentile(y_samples, 84, axis=0)
            band = (xline, low, high)

        return cls.plot_model_components(
            comps,
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
        ax=None,
        x_col: str = "cycle",
        n_points: int = 500,
        plot_kwargs: Optional[dict] = None,
        extension_factor: float = 0.05
    ):
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
        ax=None,
        sum_kwargs: Optional[dict] = None,
        comp_kwargs: Optional[dict] = None,
        uncertainty_band: Optional[tuple] = None
    ):

        def _comp_name(comp):
            return getattr(comp, "name", comp.__class__.__name__.lower())

        def _sig_param_names(comp):
            sig = inspect.signature(comp.model_func)
            return [p.name for p in list(sig.parameters.values())[1:]]

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

        sum_kwargs = dict(lw=2.6, alpha=0.95, label="Sum of selected components", color="red", zorder=3) | (sum_kwargs or {})
        comp_kwargs = dict(lw=1.5, alpha=0.9, linestyle="--") | (comp_kwargs or {})

        comp_curves = []
        for comp in model_components:
            y_comp = _eval_component(comp, xline)
            comp_curves.append((comp, y_comp))
        y_sum = np.sum([yc for _, yc in comp_curves], axis=0) if comp_curves else np.zeros_like(xline)

        if uncertainty_band is not None:
            bx, blow, bhigh = uncertainty_band
            ax.fill_between(bx, blow, bhigh, color="red", alpha=0.3, linewidth=0, label=r"Uncertainty (1$\sigma$)", zorder=2)

        ax.plot(xline, y_sum, **sum_kwargs)
        for comp, y_comp in comp_curves:
            ax.plot(xline, y_comp, label=_comp_name(comp), **comp_kwargs)
        
        return ax

    @classmethod
    def plot(
        cls,
        data: "OC",
        model: Union["InferenceData", "ModelResult", List["ModelComponent"]] = None,
        *,
        ax=None,
        ax_res=None,
        residuals: bool = True,
        title: Optional[str] = None,
        x_col: str = "cycle",
        y_col: str = "oc",
        fig_size: tuple = (10, 7),
        plot_kwargs: Optional[dict] = None,
        extension_factor: float = 0.05
    ):
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

        main_ax = ax
        resid_ax = ax_res
        
        if main_ax is None:
            if model is not None and residuals:
                fig, (main_ax, resid_ax) = plt.subplots(2, 1, figsize=fig_size, sharex=True, 
                                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.04})
            else:
                fig, main_ax = plt.subplots(figsize=(fig_size[0], fig_size[1]*0.75))
                resid_ax = None
        else:
            if residuals and resid_ax is None:
                residuals = False
        
        cls.plot_data(data, ax=main_ax, x_col=x_col, y_col=y_col, plot_kwargs=plot_kwargs)

        def _plot_resid(ax_r, x_r, resid_r, yerr_r, labels_r):
             scatter_kwargs = dict(fmt='o', markersize=3, alpha=0.8, elinewidth=0.8, capsize=1)
             
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
             
             if is_pymc:
                 cls.plot_model_pymc(model, data, ax=main_ax, x_col=x_col, extension_factor=extension_factor)
                 if residuals and resid_ax is not None:
                     y_model_post = model.posterior["y_model"]
                     yfit = y_model_post.median(dim=("chain", "draw")).values
                     if yfit.shape == y.shape:
                        resid = y - yfit
                        _plot_resid(resid_ax, x, resid, yerr, labels)
                        resid_ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6)
             elif is_lmfit:
                 cls.plot_model_lmfit(model, data, ax=main_ax, x_col=x_col, extension_factor=extension_factor)
                 if residuals and resid_ax is not None:
                     y_fit_at_x = model.eval(x=x_clean)
                     resid = y_clean - y_fit_at_x
                     _plot_resid(resid_ax, x_clean, resid, yerr_clean, labels_clean)
                     resid_ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6)
             elif is_list:
                 xmin, xmax = (float(np.min(x)), float(np.max(x))) if x.size else (0.0, 1.0)
                 margin = (xmax - xmin) * extension_factor
                 xline = np.linspace(xmin - margin, xmax + margin, 800)
                 cls.plot_model_components(model, xline=xline, ax=main_ax)
                 
                 if residuals and resid_ax is not None:
                     y_model_at_obs = np.zeros_like(x)
                     def _sig_param_names(comp):
                        sig = inspect.signature(comp.model_func)
                        return [p.name for p in list(sig.parameters.values())[1:]]
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
                     _plot_resid(resid_ax, x, resid, yerr, labels)
                     resid_ax.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6)

        if resid_ax:
            resid_ax.set_ylabel("Resid")
            resid_ax.set_xlabel(x_col.capitalize())
            resid_ax.grid(True, alpha=0.25)
            main_ax.set_xlabel("")
        
        if title:
            main_ax.set_title(title)
            
        main_ax.legend(loc="best")
        if ax is None:
            if resid_ax is None:
                try:
                    fig.tight_layout()
                except Exception:
                    pass
        
        if resid_ax is not None:
            return (main_ax, resid_ax)
        return main_ax

    @staticmethod
    def _format_label(name: str, unit: Optional[str] = None) -> str:

        
        mapping = {
            "omega": r"$\omega$",
            "e": r"$e$",
            "P": r"$P$",
            "T0": r"$T_0$",
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
        idata, 
        var_names=None, 
        cornerstyle: Literal["corner", "arviz"] = "corner", 
        units: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        
        if var_names is None:
            candidates = [v for v in idata.posterior.data_vars
                            if getattr(idata.posterior[v], "ndim", 0) == 2
                            and v not in {"y_model", "y_model_dense", "y_obs"}]
        else:
            candidates = var_names

        final_vars = []
        for v in candidates:
            vals = idata.posterior[v].values
            if vals.std() > 1e-25:
                final_vars.append(v)
        
        if not final_vars:
                raise ValueError("No suitable (non-fixed) parameters found for corner plot.")
        


        if cornerstyle == "corner":
            if corner is None:
                raise ImportError("Corner plot requires 'corner' library. Please install it with `pip install corner`.")


            subset = az.extract(idata, var_names=final_vars)
            samples = np.vstack([subset[v].values for v in final_vars]).T
            
            plot_labels = [Plot._format_label(v, (units or {}).get(v)) for v in final_vars]

            if "quantiles" not in kwargs:
                kwargs["quantiles"] = [0.16, 0.5, 0.84]
            if "show_titles" not in kwargs:
                kwargs["show_titles"] = True
            if "title_fmt" not in kwargs:
                kwargs["title_fmt"] = ".4f"
            
            fig = corner.corner(samples, labels=plot_labels, **kwargs)
            return fig
            
        elif cornerstyle == "arviz":
            if "marginals" not in kwargs:
                kwargs["marginals"] = True
            if "kind" not in kwargs:
                kwargs["kind"] = "kde"
            
            with az.rc_context({"plot.max_subplots": 200}):
                return az.plot_pair(idata, var_names=final_vars, **kwargs)
        else:
            raise ValueError(f"Unknown cornerstyle: {cornerstyle}. Use 'corner' or 'arviz'.")

    @staticmethod
    def plot_trace(idata, var_names=None, **kwargs):
        
        if var_names is None:
             candidates = [v for v in idata.posterior.data_vars if v not in {"y_model", "y_model_dense", "y_obs"}]
        else:
             candidates = var_names

        final_vars = []
        for v in candidates:
            vals = idata.posterior[v].values
            if vals.std() > 1e-10:
                final_vars.append(v)
        
        if not final_vars:
             final_vars = candidates

        axes = az.plot_trace(idata, var_names=final_vars, **kwargs)
        
        try:
            fig = axes.flatten()[0].figure
            fig.tight_layout()
        except Exception:
            pass
            
        return axes
