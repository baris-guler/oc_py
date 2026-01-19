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


