import inspect
from typing import List, Optional, Dict
import numpy as np
import lmfit
from lmfit.model import ModelResult

from .oc import OC, Parameter, Linear, Quadratic, Keplerian
from ocpy.model_oc import ModelComponentModel


class OCLMFit(OC):
    def fit(
        self,
        model_components: list[ModelComponentModel],
        *,
        nan_policy: str = "raise",
        method: str = "leastsq",
        **kwargs,
    ) -> ModelResult:
        import lmfit
        import numpy as np
        from collections import Counter, defaultdict

        x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
        y = np.asarray(self.data["oc"].to_numpy(), dtype=float)

        comps = model_components

        def base_name(c):
            return getattr(c, "name", c.__class__.__name__.lower())

        totals = Counter(base_name(c) for c in comps)
        seen = defaultdict(int)
        prefixes = []
        for c in comps:
            b = base_name(c)
            seen[b] += 1
            prefixes.append(f"{b}_" if totals[b] == 1 else f"{b}{seen[b]}_")

        def make_model(comp, prefix) -> lmfit.Model:
            return lmfit.Model(comp.model_func, independent_vars=["x"], prefix=prefix)

        model = make_model(comps[0], prefixes[0])
        for c, pref in zip(comps[1:], prefixes[1:]):
            model = model + make_model(c, pref)

        params = model.make_params()
        for comp, pref in zip(comps, prefixes):
            cparams = getattr(comp, "params", {}) or {}
            for short_key, cfg in cparams.items():
                full_key = f"{pref}{short_key}"
                if full_key not in params:
                    continue
                p = params[full_key]
                if cfg.value is not None: p.set(value=cfg.value)
                if cfg.min   is not None: p.set(min=cfg.min)
                if cfg.max   is not None: p.set(max=cfg.max)
                p.set(vary=not bool(cfg.fixed))

        weights = self.data["weights"].to_numpy(dtype=float) if "weights" in self.data.columns else None

        return model.fit(
            y, params, x=x,
            nan_policy=nan_policy,
            method=method,
            weights=weights,
            **kwargs,
        )

    def residue(self, coefficients: ModelResult, *, x_col: str = "cycle", y_col: str = "oc") -> "OCLMFit":
        x = np.asarray(self.data[x_col].to_numpy(), dtype=float)
        yfit = coefficients.eval(x=x)
        new = OCLMFit(
            minimum_time=self.data["minimum_time"].to_list() if "minimum_time" in self.data else None,
            minimum_time_error=self.data["minimum_time_error"].to_list() if "minimum_time_error" in self.data else None,
            weights=self.data["weights"].to_list() if "weights" in self.data else None,
            minimum_type=self.data["minimum_type"].to_list() if "minimum_type" in self.data else None,
            labels=self.data["labels"].to_list() if "labels" in self.data else None,
            cycle=self.data["cycle"].to_list() if "cycle" in self.data else None,
            oc=(self.data[y_col].to_numpy() - yfit).tolist() if y_col in self.data else None,
        )
        return new

    def fit_linear(self, *, params: Optional[Dict[str, Parameter]] = None, **kwargs) -> ModelResult:
        if params is None:
            params = {
                "a": Parameter(value=0.0),
                "b": Parameter(value=0.0),
            }
        comp = Linear(params)
        return self.fit([comp], **kwargs)

    def fit_quadratic(self, *, params: Optional[Dict[str, Parameter]] = None, **kwargs) -> ModelResult:
        if params is None:
            params = {
                "q": Parameter(value=0.0),
            }
        comp = Quadratic(params)
        return self.fit([comp], **kwargs)

    def fit_lite(self, *, params: Optional[Dict[str, Parameter]] = None, **kwargs) -> ModelResult:
        if params is None:
            params = {
                "amp":   Parameter(value=1e-3,  min=0.0),
                "e":     Parameter(value=0.0,   min=0.0, max=0.95),
                "omega": Parameter(value=90.0),
                "P":     Parameter(value=3000.0, min=1.0),
                "T0":    Parameter(value=0.0),
            }
        comp = Keplerian(params)
        return self.fit([comp], **kwargs)

    def fit_keplerian(self, *, params: Optional[Dict[str, Parameter]] = None, **kwargs) -> ModelResult:
        return self.fit_lite(params=params, **kwargs)

    def fit_sinusoidal(self, *, params: Optional[Dict[str, Parameter]] = None, **kwargs) -> ModelResult:
        if params is None:
            params = {
                "amp":   Parameter(value=1e-3),
                "e":     Parameter(value=0.0, min=0.0, max=0.0, fixed=True),
                "omega": Parameter(value=180.0, min=180.0, max=180.0, fixed=True),
                "P":     Parameter(value=3000.0, min=1.0),
                "T0":    Parameter(value=0.0),
            }
        comp = Keplerian(params)
        return self.fit([comp], **kwargs)

    # TODO Testing için Bitene kadar dursun sonra kaldırılacak
    def fit_and_report(
        self,
        result: ModelResult,
        *,
        x_col: str = "cycle",
        y_col: str = "oc",
        title: Optional[str] = None,
    ) -> None:
        from matplotlib import pyplot as plt
        x = np.asarray(self.data[x_col].to_numpy(), dtype=float)
        y = np.asarray(self.data[y_col].to_numpy(), dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        x_dense = np.linspace(x.min(), x.max(), 500)
        y_fit_dense = result.eval(x=x_dense)
        y_fit_at_x = result.eval(x=x)
        resid = y - y_fit_at_x

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.scatter(x, y, s=20, label="Data")
        ax.plot(x_dense, y_fit_dense, label="Fit")
        ax.set_ylabel(y_col)
        ax.set_title(title or "Fit")
        ax.legend()

        axr = axes[1]
        axr.axhline(0.0, linestyle="--", linewidth=1)
        axr.scatter(x, resid, s=15)
        axr.set_xlabel(x_col)
        axr.set_ylabel("resid")

        lines = []
        method = getattr(result, "method", "")
        redchi = getattr(result, "redchi", np.nan)
        aic = getattr(result, "aic", np.nan)
        bic = getattr(result, "bic", np.nan)
        lines.append(f"method: {method}, redchi: {redchi:.3g}")
        lines.append(f"AIC: {aic:.3g}, BIC: {bic:.3g}")
        for name, par in result.params.items():
            val = par.value
            err = par.stderr if par.stderr is not None else float('nan')
            lines.append(f"{name} = {val:.6g} ± {err:.2g}")
        report_text = "\n".join(lines)

        ax.text(
            0.02, 0.98, report_text,
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.12, pad=0.5)
        )

        plt.show()

    # TODO Testing için Bitene kadar dursun sonra kaldırılacak
    def plot_components_on_data(
        self,
        model_components: list,
        *,
        ax=None,
        n_points: int = 800,
        scatter_kwargs: dict | None = None,
        sum_kwargs: dict | None = None,
        comp_kwargs: dict | None = None,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        import inspect

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
                    raise KeyError(
                        f"Component '{_comp_name(comp)}' missing parameter '{pname}' in comp.params"
                    )
                kwargs[pname] = float(_param_value(params_dict[pname]))
            return comp.model_func(xvals, **kwargs)

        x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
        y = np.asarray(self.data["oc"].to_numpy(), dtype=float)

        scatter_kwargs = dict(s=16, alpha=0.75) | (scatter_kwargs or {})
        sum_kwargs = dict(lw=2.6, alpha=0.95, label="Sum of selected components") | (sum_kwargs or {})
        comp_kwargs = dict(lw=1.5, alpha=0.9, linestyle="--") | (comp_kwargs or {})

        xmin, xmax = (float(np.min(x)), float(np.max(x))) if x.size else (0.0, 1.0)
        if n_points < 2:
            n_points = 2
        xline = np.linspace(xmin, xmax, n_points)

        comp_curves = []
        for comp in model_components:
            y_comp = _eval_component(comp, xline)
            comp_curves.append((comp, y_comp))
        y_sum = np.sum([yc for _, yc in comp_curves], axis=0) if comp_curves else np.zeros_like(xline)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10.0, 5.4))
        else:
            fig = ax.figure

        ax.scatter(x, y, **scatter_kwargs)
        ax.plot(xline, y_sum, **sum_kwargs)

        for comp, y_comp in comp_curves:
            ax.plot(xline, y_comp, label=_comp_name(comp), **comp_kwargs)

        ax.set_xlabel("Cycle")
        ax.set_ylabel("O−C")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax
