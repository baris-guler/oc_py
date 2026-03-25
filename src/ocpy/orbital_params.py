from __future__ import annotations

import re
from typing import Optional, Union

import numpy as np
from astropy import constants as const
from astropy import units as u

_C_AU_PER_DAY: float = const.c.to(u.au / u.day).value
_MSUN_TO_MJUP: float = const.M_sun.value / const.M_jup.value
_MSUN_TO_MEARTH: float = const.M_sun.value / const.M_earth.value
_AU_TO_RSUN: float = (1 * u.au).to(u.R_sun).value
_DAYS_PER_YEAR: float = 365.242199
_SECONDS_PER_DAY: float = 86400.0

_PARAM_META: dict[str, tuple[str, str, str]] = {
    "amp_day":         ("amp",            "day",      r"A"),
    "amp_s":           ("amp",            "s",        r"A"),
    "e":               ("e",              "",         r"e"),
    "omega_deg":       ("omega",          "deg",      r"\omega"),
    "omega_rad":       ("omega",          "rad",      r"\omega"),
    "T0":              ("T_0",            "BJD",      r"T_0"),
    "P_day":           ("P",              "day",      r"P"),
    "P_yr":            ("P",              "yr",       r"P"),
    "a12_sini_au":     ("a_{12} sin i",   "AU",       r"a_{12}\,\sin i"),
    "a12_sini_rsun":   ("a_{12} sin i",   "R_sun",    r"a_{12}\,\sin i"),
    "f_mass_msun":     ("f(m)",           "M_sun",    r"f(m_3)"),
    "m3_msun":         ("m_3",            "M_sun",    r"m_3"),
    "m3_mjup":         ("m_3",            "M_Jup",    r"m_3"),
    "m3_mearth":       ("m_3",            "M_Earth",  r"m_3"),
    "m3_sini_msun":    ("m_3 sin i",      "M_sun",    r"m_3\,\sin i"),
    "m3_sini_mjup":    ("m_3 sin i",      "M_Jup",    r"m_3\,\sin i"),
    "m3_sini_mearth":  ("m_3 sin i",      "M_Earth",  r"m_3\,\sin i"),
    "a3_au":           ("a_3",            "AU",       r"a_3"),
    "a3_sini_au":      ("a_3 sin i",      "AU",       r"a_3\,\sin i"),
}

_LATEX_UNITS: dict[str, str] = {
    "day":     r"\mathrm{d}",
    "s":       r"\mathrm{s}",
    "deg":     r"^{\circ}",
    "rad":     r"\mathrm{rad}",
    "yr":      r"\mathrm{yr}",
    "BJD":     r"\mathrm{BJD}",
    "AU":      r"\mathrm{AU}",
    "R_sun":   r"R_{\odot}",
    "M_sun":   r"M_{\odot}",
    "M_Jup":   r"M_{\mathrm{Jup}}",
    "M_Earth": r"M_{\oplus}",
}

ArrayLike = Union[float, np.ndarray]


class LatexString(str):

    def _repr_html_(self) -> str:
        from html import escape
        return (
            "<div style='background:#f8f8f8; border:1px solid #ddd; "
            "padding:8px 12px; border-radius:4px; font-family:monospace; "
            "font-size:0.9em; white-space:pre; overflow-x:auto'>"
            f"{escape(str(self))}</div>"
        )


class OrbitalParamsResult:

    _all_keys = list(_PARAM_META.keys())

    def __init__(self, data: dict[str, ArrayLike]) -> None:
        self._data = data
        self._keys = [k for k in self._all_keys if k in data]

    def __getitem__(self, key: str) -> ArrayLike:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_dict(self) -> dict[str, ArrayLike]:
        return dict(self._data)

    @property
    def median(self) -> dict[str, float]:
        if not self.is_posterior:
            return {k: v for k, v in self._data.items()}
        return {k: float(np.median(v)) for k, v in self._data.items()}

    @property
    def is_posterior(self) -> bool:
        return isinstance(self._data["P_yr"], np.ndarray)

    @staticmethod
    def _fmt_scalar(val: float) -> str:
        if abs(val) < 0.01 or abs(val) >= 1e6:
            return f"{val:.6e}"
        return f"{val:.6f}"

    @staticmethod
    def _fmt_posterior(arr: np.ndarray) -> tuple[str, str]:
        med = np.median(arr)
        lo = med - np.percentile(arr, 16)
        hi = np.percentile(arr, 84) - med
        if abs(med) < 0.01 or abs(med) >= 1e6:
            return f"{med:.4e}", f"+{hi:.4e} / -{lo:.4e}"
        return f"{med:.6f}", f"+{hi:.6f} / -{lo:.6f}"

    def __repr__(self) -> str:
        lines = ["OrbitalParamsResult:"]
        for key in self._keys:
            label, unit, _ = _PARAM_META[key]
            val = self._data[key]
            unit_str = f"  [{unit}]" if unit else ""
            if isinstance(val, np.ndarray):
                val_str, unc_str = self._fmt_posterior(val)
                lines.append(
                    f"  {label:16s} = {val_str}  ({unc_str}){unit_str}  (N={len(val)})"
                )
            else:
                lines.append(f"  {label:16s} = {self._fmt_scalar(val)}{unit_str}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows: list[str] = []
        for key in self._keys:
            label, unit, _ = _PARAM_META[key]
            val = self._data[key]
            if isinstance(val, np.ndarray):
                val_str, unc_str = self._fmt_posterior(val)
                unc_str = unc_str.replace("-", "&minus;")
                rows.append(
                    f"<tr><td><b>{label}</b></td>"
                    f"<td style='text-align:right'>{val_str}</td>"
                    f"<td style='color:#888; font-size:0.9em'>{unc_str}</td>"
                    f"<td>{unit}</td></tr>"
                )
            else:
                rows.append(
                    f"<tr><td><b>{label}</b></td>"
                    f"<td style='text-align:right'>{self._fmt_scalar(val)}</td>"
                    f"<td></td>"
                    f"<td>{unit}</td></tr>"
                )

        header = "<tr><th>Parameter</th><th>Value</th><th>Uncertainty</th><th>Unit</th></tr>"
        n_info = ""
        if self.is_posterior:
            n = len(self._data["P_yr"])
            n_info = f"<p style='color:#888; font-size:0.85em; margin:4px 0 0 0'>Posterior samples: N = {n}</p>"

        return (
            "<div style='font-family: monospace'>"
            "<table style='border-collapse:collapse; margin:4px 0'>"
            f"{header}{''.join(rows)}"
            "</table>"
            f"{n_info}"
            "</div>"
        )

    @staticmethod
    def _fmt_latex_val(val: float) -> str:
        if abs(val) < 0.01 or abs(val) >= 1e6:
            mantissa, exp = f"{val:.4e}".split("e")
            return rf"{mantissa} \times 10^{{{int(exp)}}}"
        return f"{val:.4f}"

    @property
    def latex(self) -> LatexString:
        rows: list[str] = []
        for key in self._keys:
            _, unit, ltx = _PARAM_META[key]
            ltx_unit = _LATEX_UNITS.get(unit, "")
            unit_col = rf"\mathrm{{{unit}}}" if not ltx_unit else ltx_unit
            val = self._data[key]

            if isinstance(val, np.ndarray):
                med = np.median(val)
                lo = med - np.percentile(val, 16)
                hi = np.percentile(val, 84) - med
                val_str = self._fmt_latex_val(med)
                row = rf"        ${ltx}$ & ${val_str}^{{+{self._fmt_latex_val(hi)}}}_{{-{self._fmt_latex_val(lo)}}}$"
            else:
                row = rf"        ${ltx}$ & ${self._fmt_latex_val(val)}$"

            if unit:
                row += rf" & ${unit_col}$ \\"
            else:
                row += r" & \\"
            rows.append(row)

        return LatexString(
            r"\begin{table}" "\n"
            r"    \centering" "\n"
            r"    \begin{tabular}{lcc}" "\n"
            r"        \hline" "\n"
            r"        Parameter & Value & Unit \\" "\n"
            r"        \hline" "\n"
            + "\n".join(rows) + "\n"
            r"        \hline" "\n"
            r"    \end{tabular}" "\n"
            r"\end{table}"
        )


class OrbitalParamsCollection:

    def __init__(self, results: dict[str, OrbitalParamsResult]) -> None:
        self._results = results

    def __getitem__(self, key: str) -> OrbitalParamsResult:
        return self._results[key]

    def __iter__(self):
        return iter(self._results)

    def __contains__(self, key: str) -> bool:
        return key in self._results

    def __len__(self) -> int:
        return len(self._results)

    def keys(self):
        return self._results.keys()

    def values(self):
        return self._results.values()

    def items(self):
        return self._results.items()

    def __repr__(self) -> str:
        parts = []
        for name, res in self._results.items():
            lines = repr(res).split("\n")
            lines[0] = f"{name}:"
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _repr_html_(self) -> str:
        parts = []
        for name, res in self._results.items():
            title = f"<h4 style='margin:12px 0 4px 0; font-family:monospace'>{name}</h4>"
            parts.append(title + res._repr_html_())
        return "<div>" + "".join(parts) + "</div>"

    @property
    def median(self) -> dict[str, dict[str, float]]:
        return {name: res.median for name, res in self._results.items()}

    @property
    def latex(self) -> LatexString:
        parts = []
        for name, res in self._results.items():
            parts.append(f"% {name}\n{str(res.latex)}")
        return LatexString("\n\n".join(parts))


def period_to_years(P_cycles: ArrayLike, ref_period: float) -> ArrayLike:
    return (P_cycles * ref_period) / _DAYS_PER_YEAR


def a12_sini(amp: ArrayLike, e: ArrayLike, omega_deg: ArrayLike) -> ArrayLike:
    omega_rad = np.deg2rad(omega_deg)
    return amp * _C_AU_PER_DAY / np.sqrt(1.0 - e**2 * np.cos(omega_rad)**2)


def mass_function(period_yr: ArrayLike, a12sini_au: ArrayLike) -> ArrayLike:
    return a12sini_au**3 / period_yr**2


def m3_sini(
    f_mass: ArrayLike,
    m_total: float,
    tol: float = 1e-10,
    maxiter: int = 100,
) -> ArrayLike:
    x = (f_mass * m_total**2) ** (1.0 / 3.0)
    for _ in range(maxiter):
        g = x**3 - f_mass * (m_total + x)**2
        gp = 3.0 * x**2 - 2.0 * f_mass * (m_total + x)
        dx = -g / gp
        x = x + dx
        if np.all(np.abs(dx) < tol):
            break
    return x


def a3_sini(
    a12sini_au: ArrayLike, m_total: float, m3sini_val: ArrayLike
) -> ArrayLike:
    return a12sini_au * m_total / m3sini_val


def msun_to_mjup(m_msun: ArrayLike) -> ArrayLike:
    return m_msun * _MSUN_TO_MJUP


def compute_orbital_params(
    amp: ArrayLike,
    e: ArrayLike,
    omega_deg: ArrayLike,
    P_cycles: ArrayLike,
    ref_period: float,
    m1: float,
    m2: float,
    T0: Optional[ArrayLike] = None,
) -> OrbitalParamsResult:
    m_total = m1 + m2
    P_yr = period_to_years(P_cycles, ref_period)
    P_day = P_cycles * ref_period
    a12 = a12_sini(amp, e, omega_deg)
    f_m = mass_function(P_yr, a12)
    m3s = m3_sini(f_m, m_total)
    a3s = a3_sini(a12, m_total, m3s)

    data = {
        "amp_day": amp,
        "amp_s": amp * _SECONDS_PER_DAY,
        "e": e,
        "omega_deg": omega_deg,
        "omega_rad": np.deg2rad(omega_deg),
    }
    if T0 is not None:
        data["T0"] = T0
    data.update({
        "P_day": P_day,
        "P_yr": P_yr,
        "a12_sini_au": a12,
        "a12_sini_rsun": a12 * _AU_TO_RSUN,
        "f_mass_msun": f_m,
        "m3_sini_msun": m3s,
        "m3_sini_mjup": m3s * _MSUN_TO_MJUP,
        "m3_sini_mearth": m3s * _MSUN_TO_MEARTH,
        "a3_sini_au": a3s,
    })
    return OrbitalParamsResult(data)


def _kepler3_a_to_P(a_au: ArrayLike, m_total_msun: ArrayLike) -> ArrayLike:
    return np.sqrt(a_au**3 / m_total_msun)


def _kepler3_P_to_a(P_yr: ArrayLike, m_total_msun: ArrayLike) -> ArrayLike:
    return (m_total_msun * P_yr**2) ** (1.0 / 3.0)


def compute_orbital_params_newtonian(
    m3: ArrayLike,
    e: ArrayLike,
    omega_deg: ArrayLike,
    m_central: ArrayLike,
    inc_deg: ArrayLike = 90.0,
    a3_au: Optional[ArrayLike] = None,
    P_day: Optional[ArrayLike] = None,
    T0: Optional[ArrayLike] = None,
) -> OrbitalParamsResult:
    if (a3_au is None) == (P_day is None):
        raise ValueError("Exactly one of a3_au or P_day must be provided")

    m_total = m_central + m3
    inc_rad = np.deg2rad(inc_deg)
    sin_i = np.sin(inc_rad)

    if a3_au is not None:
        P_yr = _kepler3_a_to_P(a3_au, m_total)
        P_d = P_yr * _DAYS_PER_YEAR
    else:
        P_d = P_day
        P_yr = P_d / _DAYS_PER_YEAR
        a3_au = _kepler3_P_to_a(P_yr, m_total)

    m3_sini_val = m3 * sin_i
    a3_sini_val = a3_au * sin_i
    a12_sini_val = a3_sini_val * m3 / m_total
    f_m = m3_sini_val**3 / (m_central + m3)**2
    amp = a12_sini_val / _C_AU_PER_DAY * np.sqrt(
        1.0 - e**2 * np.cos(np.deg2rad(omega_deg))**2
    )

    data = {
        "amp_day": amp,
        "amp_s": amp * _SECONDS_PER_DAY,
        "e": e,
        "omega_deg": omega_deg,
        "omega_rad": np.deg2rad(omega_deg),
    }
    if T0 is not None:
        data["T0"] = T0
    data.update({
        "P_day": P_d,
        "P_yr": P_yr,
        "a12_sini_au": a12_sini_val,
        "a12_sini_rsun": a12_sini_val * _AU_TO_RSUN,
        "f_mass_msun": f_m,
        "m3_msun": m3,
        "m3_mjup": m3 * _MSUN_TO_MJUP,
        "m3_mearth": m3 * _MSUN_TO_MEARTH,
        "m3_sini_msun": m3_sini_val,
        "m3_sini_mjup": m3_sini_val * _MSUN_TO_MJUP,
        "m3_sini_mearth": m3_sini_val * _MSUN_TO_MEARTH,
        "a3_au": a3_au,
        "a3_sini_au": a3_sini_val,
    })
    return OrbitalParamsResult(data)


def _find_prefixes(names: list[str], pattern: str) -> list[str]:
    prefixes: set[str] = set()
    for name in names:
        m = re.match(pattern, name)
        if m:
            prefixes.add(m.group(1))
    return sorted(prefixes)


def _find_body_indices(names: list[str], model_prefix: str) -> list[int]:
    indices: set[int] = set()
    for name in names:
        m = re.match(rf"^{re.escape(model_prefix)}b(\d+)_", name)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _get_param(result, key: str, is_lmfit: bool) -> ArrayLike:
    if is_lmfit:
        return result.params[key].value
    return result.posterior[key].values.flatten()


def _try_get_param(
    result, key: str, is_lmfit: bool, default: ArrayLike = None
) -> Optional[ArrayLike]:
    if is_lmfit:
        if key in result.params.keys():
            return result.params[key].value
    else:
        if key in result.posterior.data_vars:
            return result.posterior[key].values.flatten()
    return default


def _classify_prefixes(
    param_names: list[str], prefix: Optional[str],
) -> tuple[list[str], list[str]]:
    names_set = set(param_names)

    if prefix is not None:
        pfx = prefix if prefix.endswith("_") else prefix + "_"
        if f"{pfx}central_mass" in names_set:
            return [], [pfx]
        return [pfx], []

    kep_prefixes = _find_prefixes(param_names, r"^(keplerian\d*_)")
    newt_prefixes = _find_prefixes(param_names, r"^(newtonian\d*_)")

    if not kep_prefixes and not newt_prefixes:
        all_prefixes = _find_prefixes(param_names, r"^([a-zA-Z][a-zA-Z0-9]*_)")
        for pfx in all_prefixes:
            if f"{pfx}central_mass" in names_set:
                newt_prefixes.append(pfx)
            elif f"{pfx}amp" in names_set:
                kep_prefixes.append(pfx)

    return kep_prefixes, newt_prefixes


def compute_orbital_params_from_fit(
    result,
    ref_period: float = None,
    m1: float = None,
    m2: float = None,
    prefix: Optional[str] = None,
) -> Union[OrbitalParamsResult, OrbitalParamsCollection]:
    is_lmfit = hasattr(result, "params") and hasattr(result.params, "valuesdict")
    is_pymc = hasattr(result, "posterior")

    if not is_lmfit and not is_pymc:
        raise TypeError(
            "result must be an lmfit ModelResult or arviz InferenceData"
        )

    if is_lmfit:
        param_names = list(result.params.keys())
    else:
        param_names = list(result.posterior.data_vars)

    kep_prefixes, newt_prefixes = _classify_prefixes(param_names, prefix)

    if not kep_prefixes and not newt_prefixes:
        raise ValueError("No keplerian or newtonian parameters found in result")

    results: dict[str, OrbitalParamsResult] = {}

    for pfx in kep_prefixes:
        if ref_period is None or m1 is None or m2 is None:
            raise ValueError(
                "ref_period, m1, and m2 are required for keplerian results"
            )
        amp = _get_param(result, pfx + "amp", is_lmfit)
        e_val = _get_param(result, pfx + "e", is_lmfit)
        omega = _get_param(result, pfx + "omega", is_lmfit)
        P_cycles = _get_param(result, pfx + "P", is_lmfit)
        T0_val = _try_get_param(result, pfx + "T0", is_lmfit)

        results[pfx.rstrip("_")] = compute_orbital_params(
            amp=amp, e=e_val, omega_deg=omega,
            P_cycles=P_cycles, ref_period=ref_period,
            m1=m1, m2=m2, T0=T0_val,
        )

    for pfx in newt_prefixes:
        m_central = _get_param(result, pfx + "central_mass", is_lmfit)
        body_indices = _find_body_indices(param_names, pfx)

        for bi in body_indices:
            bp = f"{pfx}b{bi}_"
            m3_val = _get_param(result, bp + "m", is_lmfit)
            e_val = _try_get_param(result, bp + "e", is_lmfit, 0.0)
            omega = _try_get_param(result, bp + "omega", is_lmfit, 0.0)
            inc = _try_get_param(result, bp + "inc", is_lmfit, 90.0)
            a_val = _try_get_param(result, bp + "a", is_lmfit)
            P_val = _try_get_param(result, bp + "P", is_lmfit)
            T_val = _try_get_param(result, bp + "T", is_lmfit)

            r = compute_orbital_params_newtonian(
                m3=m3_val, e=e_val, omega_deg=omega,
                m_central=m_central, inc_deg=inc,
                a3_au=a_val, P_day=P_val, T0=T_val,
            )
            label = f"{pfx.rstrip('_')}_b{bi}"
            results[label] = r

    if len(results) == 1:
        return next(iter(results.values()))
    return OrbitalParamsCollection(results)


def _kepler_solve_numpy(M: np.ndarray, e, n_iter: int = 10) -> np.ndarray:
    E = M.copy() if isinstance(M, np.ndarray) else np.array(M, dtype=float)
    E -= 2.0 * np.pi * np.round(E / (2.0 * np.pi))
    M_red = E.copy()
    for _ in range(n_iter):
        sin_E = np.sin(E)
        cos_E = np.cos(E)
        E = E - (E - e * sin_E - M_red) / (1.0 - e * cos_E)
    return E


def forward_model(
    cycles: ArrayLike,
    amp: ArrayLike,
    e: ArrayLike,
    omega_deg: ArrayLike,
    P_cycles: ArrayLike,
    T0: ArrayLike,
) -> np.ndarray:
    cycles = np.atleast_1d(np.asarray(cycles, dtype=float))
    is_posterior = np.ndim(amp) > 0

    if is_posterior:
        amp       = np.asarray(amp,       dtype=float)[:, np.newaxis]
        e         = np.asarray(e,         dtype=float)[:, np.newaxis]
        omega_deg = np.asarray(omega_deg, dtype=float)[:, np.newaxis]
        P_cycles  = np.asarray(P_cycles,  dtype=float)[:, np.newaxis]
        T0        = np.asarray(T0,        dtype=float)[:, np.newaxis]
        cycles    = cycles[np.newaxis, :]

    w_rad = np.deg2rad(omega_deg)
    M = 2.0 * np.pi * (cycles - T0) / P_cycles
    E = _kepler_solve_numpy(M, e)

    sin_E = np.sin(E)
    cos_E = np.cos(E)
    sin_w = np.sin(w_rad)
    cos_w = np.cos(w_rad)
    e2 = e ** 2

    bracket = np.sqrt(1.0 - e2) * sin_E * cos_w + cos_E * sin_w
    K = amp / np.sqrt(1.0 - e2 * cos_w ** 2)

    return K * bracket


def forward_model_from_fit(
    result,
    cycles: ArrayLike,
    prefix: Optional[str] = None,
) -> np.ndarray:
    is_lmfit = hasattr(result, "params") and hasattr(result.params, "valuesdict")
    is_pymc = hasattr(result, "posterior")

    if not is_lmfit and not is_pymc:
        raise TypeError(
            "result must be an lmfit ModelResult or arviz InferenceData"
        )

    if is_lmfit:
        param_names = list(result.params.keys())
    else:
        param_names = list(result.posterior.data_vars)

    kep_prefixes, newt_prefixes = _classify_prefixes(param_names, prefix)

    if newt_prefixes and not kep_prefixes:
        raise ValueError(
            "Forward modelling from Newtonian fit results requires the "
            "original NewtonianModel component. Use "
            "model.model_func(cycles, **params) directly."
        )

    if not kep_prefixes:
        raise ValueError("No keplerian parameters found in result")

    cycles = np.atleast_1d(np.asarray(cycles, dtype=float))
    total = None

    for pfx in kep_prefixes:
        amp_val = _get_param(result, pfx + "amp", is_lmfit)
        e_val   = _get_param(result, pfx + "e", is_lmfit)
        omega   = _get_param(result, pfx + "omega", is_lmfit)
        P_val   = _get_param(result, pfx + "P", is_lmfit)
        T0_val  = _get_param(result, pfx + "T0", is_lmfit)

        contribution = forward_model(cycles, amp_val, e_val, omega, P_val, T0_val)
        total = contribution if total is None else total + contribution

    return total
