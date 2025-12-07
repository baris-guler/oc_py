from typing import Optional, Union
import numpy as np
from lmfit.model import ModelResult

from .oc import OC, Parameter, Linear, Quadratic, Keplerian, Sinusoidal
from ocpy.model_oc import ModelComponentModel


def _ensure_param(x, *, default: Parameter) -> Parameter:
    if isinstance(x, Parameter):
        return x
    if x is None:
        return default
    return Parameter(value=x)


class OCLMFit(OC):
    math = np

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

        for c in comps:
            if hasattr(c, "set_math"):
                c.set_math(self.math)

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
                if cfg.value is not None:
                    p.set(value=cfg.value)
                if cfg.min is not None:
                    p.set(min=cfg.min)
                if cfg.max is not None:
                    p.set(max=cfg.max)
                p.set(vary=not bool(cfg.fixed))

        weights = self.data["weights"].to_numpy(dtype=float)
        if np.isnan(weights).any():
            raise ValueError("OCLMFit.fit(...) found NaN values in 'weights'. Please fill or drop them.")

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

    def fit_linear(self, *, a: Union[Parameter, float, None] = None, b: Union[Parameter, float, None] = None, **kwargs) -> ModelResult:
        a = _ensure_param(a, default=Parameter(value=0.0))
        b = _ensure_param(b, default=Parameter(value=0.0))
        comp = Linear(a=a, b=b)
        return self.fit([comp], **kwargs)

    def fit_quadratic(self, *, q: Union[Parameter, float, None] = None, **kwargs) -> ModelResult:
        q = _ensure_param(q, default=Parameter(value=0.0))
        comp = Quadratic(q=q)
        return self.fit([comp], **kwargs)
    
    def fit_parabola(
        self,
        *,
        q: Union[Parameter, float, None] = None,
        a: Union[Parameter, float, None] = None,
        b: Union[Parameter, float, None] = None,
        **kwargs,
    ) -> ModelResult:
        q = _ensure_param(q, default=Parameter(value=0.0))
        a = _ensure_param(a, default=Parameter(value=0.0))
        b = _ensure_param(b, default=Parameter(value=0.0))
        comp_q = Quadratic(q=q)
        comp_l = Linear(a=a, b=b)
        return self.fit([comp_q, comp_l], **kwargs)

    def fit_lite(
        self,
        *,
        amp:   Union[Parameter, float, None] = None,
        e:     Union[Parameter, float, None] = None,
        omega: Union[Parameter, float, None] = None,
        P:     Union[Parameter, float, None] = None,
        T0:    Union[Parameter, float, None] = None,
        **kwargs,
    ) -> ModelResult:
        amp   = _ensure_param(amp,   default=Parameter(value=1e-3, min=0.0))
        e     = _ensure_param(e,     default=Parameter(value=0.0,   min=0.0, max=0.95))
        omega = _ensure_param(omega, default=Parameter(value=90.0))
        P     = _ensure_param(P,     default=Parameter(value=3000.0, min=1.0))
        T0    = _ensure_param(T0,    default=Parameter(value=0.0))

        comp = Keplerian(amp=amp, e=e, omega=omega, P=P, T0=T0)
        return self.fit([comp], **kwargs)

    def fit_keplerian(
        self,
        *,
        amp:   Union[Parameter, float, None] = None,
        e:     Union[Parameter, float, None] = None,
        omega: Union[Parameter, float, None] = None,
        P:     Union[Parameter, float, None] = None,
        T0:    Union[Parameter, float, None] = None,
        **kwargs,
    ) -> ModelResult:
        return self.fit_lite(amp=amp, e=e, omega=omega, P=P, T0=T0, **kwargs)

    def fit_sinusoidal(
        self,
        *,
        amp: Union[Parameter, float, None] = None,
        P:   Union[Parameter, float, None] = None,
        **kwargs,
    ) -> ModelResult:
        amp = _ensure_param(amp, default=Parameter(value=1e-3, min=0))
        P   = _ensure_param(P,   default=Parameter(value=3000.0, min=0))

        comp = Sinusoidal(
            amp=amp,
            P=P,
        )
        return self.fit([comp], **kwargs)

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
        ax.plot(x_dense, y_fit_dense, "r-", label="Fit")
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
                    raise KeyError(f"Component '{_comp_name(comp)}' missing parameter '{pname}' in comp.params")
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
