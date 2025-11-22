from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from collections import Counter, defaultdict

from .oc import OC, Linear, Quadratic, Keplerian, Sinusoidal, Parameter


class OCPyMC(OC):
    math = pm.math

    def _to_param(
        self,
        x,
        *,
        default: float = 0.0,
        min_: float | None = None,
        max_: float | None = None,
        fixed: bool = False,
    ) -> Parameter:
        if isinstance(x, Parameter):
            return x
        return Parameter(value=default if x is None else x, min=min_, max=max_, fixed=fixed)

    def fit(
        self,
        model_components: List[Any],
        *,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        progressbar: bool = False,
    ) -> az.InferenceData:
        x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
        y = np.asarray(self.data["oc"].to_numpy(), dtype=float)

        sigma_i = np.asarray(self.data["minimum_time_error"].to_numpy(), dtype=float)
        if np.isnan(sigma_i).any():
            raise ValueError("Found NaN in 'minimum_time_error'.")

        for c in model_components:
            if hasattr(c, "set_math"):
                c.set_math(self.math)

        def _rv(name: str, par):
            v = getattr(par, "value", 0.0)
            sd = getattr(par, "std", None)
            lo = getattr(par, "min", None)
            hi = getattr(par, "max", None)
            fix = bool(getattr(par, "fixed", False))
            if fix:
                return float(0.0 if v is None else v)
            sigma = float(sd) if (sd is not None and np.isfinite(sd) and sd > 0) else 1.0
            if (lo is not None and np.isfinite(lo)) or (hi is not None and np.isfinite(hi)):
                lower = None if lo is None else float(lo)
                upper = None if hi is None else float(hi)
                return pm.TruncatedNormal(
                    name,
                    mu=float(0.0 if v is None else v),
                    sigma=sigma,
                    lower=lower,
                    upper=upper,
                )
            return pm.Normal(name, mu=float(0.0 if v is None else v), sigma=sigma)

        with pm.Model() as model:
            prefixes = [
                f"{getattr(c,'name', c.__class__.__name__.lower())}{i+1}_"
                for i, c in enumerate(model_components)
            ]

            comp_rvs: Dict[str, Dict[str, Any]] = {}
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

            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                cores=min(chains, 4),
                progressbar=progressbar,
            )

        return idata

    def residue(
        self,
        idata: az.InferenceData,
        *,
        x_col: str = "cycle",
        y_col: str = "oc",
    ) -> "OCPyMC":
        y_model = idata.posterior["y_model"]
        yfit = y_model.median(dim=("chain", "draw")).values
        return OCPyMC(
            minimum_time=self.data["minimum_time"].to_list() if "minimum_time" in self.data else None,
            minimum_time_error=self.data["minimum_time_error"].to_list()
            if "minimum_time_error" in self.data
            else None,
            weights=self.data["weights"].to_list() if "weights" in self.data else None,
            minimum_type=self.data["minimum_type"].to_list() if "minimum_type" in self.data else None,
            labels=self.data["labels"].to_list() if "labels" in self.data else None,
            cycle=self.data["cycle"].to_list() if "cycle" in self.data else None,
            oc=(
                self.data[y_col].to_numpy(dtype=float) - yfit
            ).tolist()
            if y_col in self.data
            else None,
        )

    def fit_linear(
        self,
        *,
        a: float | Parameter | None = None,
        b: float | Parameter | None = None,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        progressbar: bool = False,
    ):
        lin = Linear(
            a=self._to_param(a, default=0.0),
            b=self._to_param(b, default=0.0),
        )
        return self.fit(
            [lin],
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    def fit_quadratic(
        self,
        *,
        q: float | Parameter | None = None,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        progressbar: bool = False,
    ) -> az.InferenceData:
        comp = Quadratic(q=self._to_param(q, default=0.0))
        return self.fit(
            [comp],
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    def fit_sinusoidal(
        self,
        *,
        amp: float | Parameter | None = None,
        P: float | Parameter | None = None,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        progressbar: bool = False,
    ) -> az.InferenceData:
        comp = Sinusoidal(
            amp=self._to_param(amp, default=1e-3, min_=0.0),
            P=self._to_param(P, default=200.0, min_=10.0, max_=800.0),
        )
        return self.fit(
            [comp],
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    def fit_keplerian(
        self,
        *,
        amp: float | Parameter | None = None,
        e: float | Parameter | None = None,
        omega: float | Parameter | None = None,
        P: float | Parameter | None = None,
        T0: float | Parameter | None = None,
        name: Optional[str] = None,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        progressbar: bool = False,
    ) -> az.InferenceData:
        comp = Keplerian(
            amp=self._to_param(amp, default=0.0, min_=0.0),
            e=self._to_param(e, default=0.0, min_=0.0, max_=0.9),
            omega=self._to_param(omega, default=90.0),
            P=self._to_param(P, default=100.0, min_=1.0),
            T0=self._to_param(T0, default=0.0),
            name=name or "keplerian1",
        )
        return self.fit(
            [comp],
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    def fit_lite(self, **kwargs) -> az.InferenceData:
        return self.fit_keplerian(**kwargs)
    
    def fit_parabola(
        self,
        *,
        q: float | Parameter | None = None,
        a: float | Parameter | None = None,
        b: float | Parameter | None = None,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        progressbar: bool = False,
    ) -> az.InferenceData:
        quad = Quadratic(q=self._to_param(q, default=0.0))
        lin  = Linear(
            a=self._to_param(a, default=0.0),
            b=self._to_param(b, default=0.0),
        )
        return self.fit(
            [quad, lin],
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )
    
    #Teste gerek yok çıkarılacak
    def fit_and_report(
        self,
        idata: az.InferenceData,
        *,
        x_col: str = "cycle",
        y_col: str = "oc",
        title: str | None = "Components (posterior median)",
    ) -> None:
        import re

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

        scalars = [
            vn
            for vn, da in idata.posterior.data_vars.items()
            if getattr(da, "ndim", 0) == 2 and vn not in {"y_model", "y_model_dense", "y_obs"}
        ]
        if not scalars:
            raise ValueError("Posterior’da skaler parametre yok.")

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
                comps.append(
                    Linear(
                        a=Parameter(value=fields.get("a", 0.0), fixed=True),
                        b=Parameter(value=fields.get("b", 0.0), fixed=True),
                    )
                )

            elif base == "quadratic":
                comps.append(
                    Quadratic(
                        q=Parameter(value=fields.get("q", 0.0), fixed=True),
                    )
                )

            elif base in ("keplerian", "kep", "lite", "LiTE"):
                T_val = fields.get("T0", fields.get("T", 0.0))
                comps.append(
                    Keplerian(
                        amp=Parameter(value=fields.get("amp", 0.0), fixed=True),
                        e=Parameter(value=fields.get("e", 0.0), fixed=True),
                        omega=Parameter(value=fields.get("omega", 0.0), fixed=True),
                        P=Parameter(value=fields.get("P", 1.0), fixed=True),
                        T0=Parameter(value=T_val, fixed=True),
                        name=pref,
                    )
                )

        fig, ax = self.plot_components_on_data(
            comps,
            sum_kwargs=dict(lw=2.6, alpha=0.95, label="Sum of median components"),
            comp_kwargs=dict(lw=1.5, alpha=0.85, linestyle="--"),
        )
        ax.set_title(title)
        fig.tight_layout()

    #Teste gerek yok çıkarılacak
    def create_corner_plot(
        self,
        idata,
        *,
        var_names=None,
        method="arviz",
        kind="kde",
        figsize=(6, 6),
    ):
        import matplotlib.pyplot as plt
        import arviz as az

        all_vars = list(idata.posterior.data_vars.keys())
        auto_vars = [
            v
            for v in all_vars
            if v not in ("y_model", "y_model_dense", "y_obs")
            and getattr(idata.posterior[v], "ndim", 0) == 2
        ]

        if var_names is None:
            use_vars = auto_vars
        else:
            missing = [v for v in var_names if v not in all_vars]
            if missing:
                use_vars = auto_vars
            else:
                use_vars = var_names

        if not use_vars:
            raise ValueError("corner çizecek uygun parametre bulunamadı.")

        if len(use_vars) == 1:
            fig = az.plot_posterior(
                idata,
                var_names=use_vars,
                figsize=figsize,
            )
            return fig

        fig = az.plot_pair(
            idata,
            var_names=use_vars,
            kind=kind,
            marginals=True,
            figsize=figsize,
        )
        return fig

    #Teste gerek yok çıkarılacak
    def create_trace_plot(
        self,
        idata: az.InferenceData,
        *,
        var_names: Optional[List[str]] = None,
        compact: bool = True,
        divergences: bool = True,
        figsize: tuple = (12, 8),
        show: bool = True,
    ):
        import arviz as az
        import matplotlib.pyplot as plt

        if var_names is None:
            var_names = [
                vn
                for vn, da in idata.posterior.data_vars.items()
                if getattr(da, "ndim", 0) == 2 and vn not in {"y_model", "y_obs"}
            ]
            if not var_names:
                raise ValueError("No scalar parameters found to trace-plot.")

        fig = plt.figure(figsize=figsize)
        az.plot_trace(idata, var_names=var_names, compact=compact, divergences=divergences)
        if show:
            plt.show()
        return fig

    #Teste gerek yok çıkarılacak
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
                    raise KeyError(f"Component '{_comp_name(comp)}' missing parameter '{pname}'")
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