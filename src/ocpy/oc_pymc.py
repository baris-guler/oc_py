from __future__ import annotations
from typing import List, Optional

import numpy as np
import pymc as pm
import arviz as az
import re

from .oc import OC, Linear, Quadratic, Keplerian, Sinusoidal, Parameter, ModelComponent


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
        target_accept: float = 0.9, 
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

            # Sabit parametreleri de Deterministic olarak kaydediyoruz.
            if fix:
                import pytensor.tensor as pt
                return pm.Deterministic(name, pt.as_tensor_variable(val))

            if sd is None or sd <= 0:
                sd = 1.0 

            if (lo is not None and np.isfinite(lo)) or (hi is not None and np.isfinite(hi)):
                lower = float(lo) if lo is not None else None
                upper = float(hi) if hi is not None else None
                return pm.TruncatedNormal(name, mu=val, sigma=float(sd), lower=lower, upper=upper, initval=val)
            
            return pm.Normal(name, mu=val, sigma=float(sd), initval=val)

        with pm.Model() as model:
            # Generate prefixes: append index only if names collide
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

            if return_model:
                return model

            if "cores" not in kwargs:
                kwargs["cores"] = min(chains, 4)
            
            if "init" not in kwargs:
                kwargs["init"] = "adapt_diag"

            idata = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                target_accept=target_accept, 
                random_seed=random_seed, 
                return_inferencedata=True, 
                progressbar=progressbar,
                **kwargs
            )

        return idata

    def residue(self, idata: az.InferenceData, *, x_col: str = "cycle", y_col: str = "oc") -> "OCPyMC":
        y_model = idata.posterior["y_model"]
        yfit = y_model.median(dim=("chain", "draw")).values
        
        return OCPyMC(
            minimum_time=self.data["minimum_time"].to_list() if "minimum_time" in self.data else None,
            minimum_time_error=self.data["minimum_time_error"].to_list() if "minimum_time_error" in self.data else None,
            weights=self.data["weights"].to_list() if "weights" in self.data else None,
            minimum_type=self.data["minimum_type"].to_list() if "minimum_type" in self.data else None,
            labels=self.data["labels"].to_list() if "labels" in self.data else None,
            cycle=self.data["cycle"].to_list() if "cycle" in self.data else None,
            oc=(self.data[y_col].to_numpy(dtype=float) - yfit).tolist() if y_col in self.data else None,
        )

    def fit_linear(self, *, a: float | Parameter | None = None, b: float | Parameter | None = None, **kwargs):
        lin = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([lin], **kwargs)

    def fit_quadratic(self, *, q: float | Parameter | None = None, **kwargs) -> az.InferenceData:
        comp = Quadratic(q=self._to_param(q, default=0.0))
        return self.fit([comp], **kwargs)

    def fit_sinusoidal(self, *, amp: float | Parameter | None = None, P: float | Parameter | None = None, **kwargs) -> az.InferenceData:
        comp = Sinusoidal(amp=self._to_param(amp, default=1e-3), P=self._to_param(P, default=1000.0))
        return self.fit([comp], **kwargs)

    def fit_keplerian(self, *, amp: float | Parameter | None = None, e: float | Parameter | None = None, omega: float | Parameter | None = None, P: float | Parameter | None = None, T0: float | Parameter | None = None, name: Optional[str] = None, **kwargs) -> az.InferenceData:
        comp = Keplerian(
            amp=self._to_param(amp, default=0.001),
            e=self._to_param(e, default=0.1),
            omega=self._to_param(omega, default=90.0),
            P=self._to_param(P, default=1000.0),
            T0=self._to_param(T0, default=0.0),
            name=name or "keplerian1",
        )
        return self.fit([comp], **kwargs)

    def fit_lite(self, **kwargs) -> az.InferenceData:
        return self.fit_keplerian(**kwargs)
    
    def fit_parabola(self, *, q: float | Parameter | None = None, a: float | Parameter | None = None, b: float | Parameter | None = None, **kwargs) -> az.InferenceData:
        quad = Quadratic(q=self._to_param(q, default=0.0))
        lin  = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([quad, lin], **kwargs)

    def fit_and_report(self, idata: az.InferenceData, *, title: str | None = "Components (posterior median)", residuals: bool = True) -> None:
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
            return

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

        if hasattr(self, "plot_components_on_data"):
            # --- Uncertainty Band Calculation ---
            # 1. Generate x-values for plotting (consistent with plot_components_on_data default)
            x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
            xmin, xmax = (float(np.min(x)), float(np.max(x))) if x.size else (0.0, 1.0)
            xline = np.linspace(xmin, xmax, 800)

            # 2. Sample from posterior
            # Using arviz extract to get a flat chain of samples
            subset = az.extract(idata, num_samples=200)
            
            # 3. Evaluate models
            # We need to reconstruct the total model for each sample.
            # We iterate over the `comps` we just built (which correspond to `order` in prefixes)
            # and evaluate them with the parameters from the sample.
            
            y_samples = []
            n_draws = subset.sample.size
            
            # Pre-fetch variable data to avoid repeated lookups
            # groups structure: groups[prefix][pname] -> median
            # we need: subset[f"{prefix}_{pname}"]
            
            for s in range(n_draws):
                y_total = np.zeros_like(xline)
                
                for i, pref in enumerate(order):
                    comp = comps[i]
                    # Get parameters for this component
                    # Note: We can't use comp.params directly because those are fixed to median from the factory above.
                    # We use the 'groups' keys to know which params to fetch.
                    
                    kwargs = {}
                    for pname in groups[pref].keys():
                        vn = f"{pref}_{pname}"
                        # split_name logic used rfind('_'), so we reconstruct it exactly.
                        # Wait, groups keys came from split_name.
                        # If prefix is "linear" and pname is "a", var is "linear_a".
                        
                        if vn in subset:
                            val = subset[vn].values[s]
                            kwargs[pname] = float(val)
                        else:
                            # Fallback for fixed parameters not in posterior (unlikely if they are in groups)
                            pass

                    # comp is an object, we can call model_func
                    y_total += comp.model_func(xline, **kwargs)
                
                y_samples.append(y_total)
            
            y_samples = np.array(y_samples)
            # Calculate 1-sigma interval (16% - 84%) or 95%
            # Let's use 1-sigma for "uncertainty bar" typically or 2-sigma. 
            # PyMC default is usually HDI, let's use percentiles for simplicity and speed.
            low = np.percentile(y_samples, 16, axis=0)
            high = np.percentile(y_samples, 84, axis=0)
            
            band = (xline, low, high)

            fig, ax = self.plot_components_on_data(
                comps,
                sum_kwargs=dict(lw=2.6, alpha=0.95, label="Median Model"),
                comp_kwargs=dict(lw=1.5, alpha=0.85, linestyle="--"),
                uncertainty_band=band,
                residuals=residuals
            )
            
            # If subplots were created (2 axes), set title on the main one
            if isinstance(ax, (tuple, list, np.ndarray)):
                 ax[0].set_title(title)
            else:
                 ax.set_title(title)
                 
            fig.tight_layout()

    # Teste gerek yok çıkarılacak
    def create_corner_plot(self, idata, var_names=None, textsize=10, kind="kde", **kwargs):
        import numpy as np
        import arviz as az
        import matplotlib.pyplot as plt

        # 1) Otomatik Değişken Seçimi
        if var_names is None:
            candidates = [v for v in idata.posterior.data_vars
                            if getattr(idata.posterior[v], "ndim", 0) == 2
                            and v not in {"y_model", "y_model_dense", "y_obs"}]
        else:
            candidates = var_names

        final_vars = []
        for v in candidates:
            final_vars.append(v)

        # 2) Sabit değişken kontrolü
        # Kullanıcı "fixed parametreleri çizdirmesin" dediği için
        # varyansı 0 (veya çok çok düşük) olanları listeden çıkarıyoruz.
        
        filtered_vars = []
        for v in final_vars:
            # Orijinal veriden kontrol edelim
            vals = idata.posterior[v].values
            # Gerçekten sabit (varyansı 0) olanları eleyelim. 
            # PyMC fixed parametreleri deterministic olarak kaydederse dümdüz sabit olur.
            # Yine de minik float hataları için çok küçük bir epsilon koyalım.
            if vals.std() > 1e-25:
                filtered_vars.append(v)
        
        final_vars = filtered_vars
        
        if not final_vars:
             raise ValueError("Corner plot için uygun (sabit olmayan) parametre bulunamadı.")

        # Orijinal veriyi bozmamak için kopya üzerinde çalışıyoruz.
        plot_data = idata.posterior.copy(deep=True)

        # 3) Çizim
        kde_kwargs = kwargs.pop("kde_kwargs", {"contour": True})
        
        # Divergences verisi yoksa veya varsayılan olarak False yapalım uyarılardan kaçınmak için
        # Eğer kullanıcı özel olarak istemediyse False yapıyoruz.
        divergences = kwargs.pop("divergences", False)
        
        # Çok fazla değişken varsa plot limiti uyarısı gelir, bunu arttıralım
        if "plot.max_subplots" in az.rcParams:
            az.rcParams['plot.max_subplots'] = max(az.rcParams['plot.max_subplots'], (len(final_vars)**2) + 10)
        
        ax = az.plot_pair(
            plot_data, # Kopya veriyi kullanıyoruz
            var_names=final_vars,
            kind=kind,
            marginals=True,
            textsize=textsize,
            point_estimate="median",
            divergences=divergences,
            show=False,
            kde_kwargs=kde_kwargs if kind == "kde" else None,
            **kwargs
        )

        # 4) Figür referansı döndür
        if isinstance(ax, np.ndarray):
            fig = ax.ravel()[0].figure
        else:
            fig = ax.figure

        fig.tight_layout()
        return fig

    # Teste gerek yok çıkarılacak
    def plot_components_on_data(
        self,
        model_components: list,
        *,
        ax=None,
        n_points: int = 800,
        scatter_kwargs: dict | None = None,
        sum_kwargs: dict | None = None,
        comp_kwargs: dict | None = None,
        uncertainty_band: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        residuals: bool = True,
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

        # Residualler için modelin gözlem noktalarındaki değerini hesaplayalım
        y_model_at_obs = np.zeros_like(x)
        for comp in model_components:
            y_model_at_obs += _eval_component(comp, x)
        res_data = y - y_model_at_obs

        ax_res = None
        if ax is None:
            if residuals:
                fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True, 
                                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.04})
            else:
                fig, ax = plt.subplots(figsize=(10.0, 5.4))
        else:
            fig = ax.figure
            # Dışarıdan ax verildiyse residuals çizmek zor, şimdilik atlıyoruz veya
            # kullanıcı residuals=True dediyse bile tek ax'e çizemeyiz.
            pass

        if uncertainty_band is not None:
            bx, blow, bhigh = uncertainty_band
            ax.fill_between(bx, blow, bhigh, color="k", alpha=0.2, label="Uncertainty (1σ)")

        ax.scatter(x, y, **scatter_kwargs)
        ax.plot(xline, y_sum, **sum_kwargs)
        for comp, y_comp in comp_curves:
            ax.plot(xline, y_comp, label=_comp_name(comp), **comp_kwargs)

        ax.set_ylabel("O−C")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        
        # Residuals çizimi
        if ax_res is not None:
            ax_res.scatter(x, res_data, s=16, alpha=0.75, color="k")
            ax_res.axhline(0, color="gray", lw=1.5, ls="--", alpha=0.6)
            ax_res.set_ylabel("Resid")
            ax_res.set_xlabel("Cycle")
            ax_res.grid(True, alpha=0.25)
            # Üst eksendeki x label'ı kaldıralım ki çakışmasın
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Cycle")

        # tight_layout çağrısını dışarı bıraktık (dönen fig üzerinden yapılabilir) veya burada yapabiliriz.
        # fit_and_report çağırıyor zaten.
        return fig, ax if ax_res is None else (ax, ax_res)