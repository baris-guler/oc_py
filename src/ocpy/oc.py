from typing import Union, Optional, Dict, Self, Callable, List
from lmfit.model import ModelResult
from pathlib import Path
import numpy as np
import pandas as pd

from ocpy.custom_types import ArrayReducer, BinarySeq, NumberOrParam
from ocpy.utils import Fixer
from ocpy.model_oc import OCModel, ModelComponentModel, ParameterModel
from dataclasses import dataclass

@dataclass
class Parameter(ParameterModel):
    value: Optional[float] = None
    min:   Optional[float] = None
    max:   Optional[float] = None
    std:   Optional[float] = None
    fixed: Optional[bool]  = False


class ModelComponent(ModelComponentModel):
    params: Dict[str, Parameter]

    def model_function(self):
        return self.model_func

    @staticmethod
    def _param(v: NumberOrParam) -> Parameter:
        if isinstance(v, Parameter):
            return v
        return Parameter(value=None if v is None else float(v))


class Linear(ModelComponent):
    name = "linear"

    def __init__(self, a: NumberOrParam = 1.0, b: NumberOrParam = 0.0, *, name: Optional[str] = None) -> None:
        if name is not None:
            self.name = name
        self.params = {
            "a": self._param(a),
            "b": self._param(b),
        }

    @staticmethod
    def model_func(x, a, b):
        x = np.asarray(x, dtype=float)
        return a * x + b


class Quadratic(ModelComponent):
    name = "quadratic"

    def __init__(self, q: NumberOrParam = 0.0, *, name: Optional[str] = None) -> None:
        if name is not None:
            self.name = name
        self.params = {
            "q": self._param(q),
        }

    @staticmethod
    def model_func(x, q):
        x = np.asarray(x, dtype=float)
        return q * x**2


class Keplerian(ModelComponent):
    name = "keplerian"

    def __init__(
        self,
        *,
        amp:   NumberOrParam = None,
        e:     NumberOrParam = 0.0,
        omega: NumberOrParam = 0.0,
        P:     NumberOrParam = None,
        T0:    NumberOrParam = None,
        name:  Optional[str] = None,
    ) -> None:
        if name is not None:
            self.name = name
        self.params = {
            "amp":   self._param(amp),
            "e":     self._param(e),
            "omega": self._param(omega),
            "P":     self._param(P),
            "T0":    self._param(T0),
        }

    @staticmethod
    def _kepler_solve(M, e, tol=1e-12, max_iter=50):
        M = (M + np.pi) % (2*np.pi) - np.pi
        e = np.clip(e, 0.0, 1.0 - 1e-12)
        E = M + e*np.sin(M)
        for _ in range(max_iter):
            f  = E - e*np.sin(E) - M
            fp = 1.0 - e*np.cos(E)
            dE = -f / fp
            E  = E + dE
            if np.all(np.abs(dE) < tol):
                break
        return E

    @staticmethod
    def model_func(x, amp, e, omega, P, T0):
        x = np.asarray(x, dtype=float)
        wr = np.deg2rad(omega)
        E = Keplerian._kepler_solve(2*np.pi * (x - T0) / P, e)
        return (amp / np.sqrt(np.maximum(0.0, 1 - (e*np.cos(wr))**2))) * (
            np.sqrt(np.maximum(0.0, 1 - e*e)) * np.sin(E) * np.cos(wr) + np.cos(E) * np.sin(wr)
        )

# Bir şeyleri test edeceğim onun için duruyor
# TODO Silinecek 
class Keplerian_Old(ModelComponent):
    name = "keplerian"

    def __init__(self, params: Optional[Dict[str, Parameter]] = None) -> None:
        super().__init__(params)

    @staticmethod
    def _kepler_solve(M, e, tol=1e-12, max_iter=50):
        M = (M + np.pi) % (2.0 * np.pi) - np.pi
        e = np.clip(e, 0.0, 1.0 - 1e-12)

        if np.ndim(e) == 0:
            if e < 1e-12:
                return M
        else:
            if np.max(e) < 1e-12:
                return M

        E = M + e * np.sin(M)

        for _ in range(max_iter):
            f  = E - e * np.sin(E) - M
            fp = 1.0 - e * np.cos(E)
            dE = -f / fp
            E  = E + dE
            if np.max(np.abs(dE)) < tol:
                break
        return E

    @staticmethod
    def model_func(x, K, e, omega, P, T0):
        x = np.asarray(x, dtype=float)
        omega = np.deg2rad(omega)

        M = 2.0 * np.pi * (x - T0) / P
        E = Keplerian._kepler_solve(M, e)

        cosE = np.cos(E)
        sinE = np.sin(E)
        sqrt1me2 = np.sqrt(np.maximum(0.0, 1.0 - e * e))

        return K * ((cosE - e) * np.sin(omega) + sqrt1me2 * sinE * np.cos(omega))


class OC(OCModel):
    @classmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        file_path = Path(file)
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Use `csv`, `xls`, or `xlsx` instead")
        expected = ["minimum_time", "minimum_time_error", "weights", "minimum_type", "labels", "cycle", "oc"]
        if columns:
            if any(k in expected for k in columns.keys()):
                rename_map = {v: k for k, v in columns.items()}
            else:
                rename_map = columns
            df = df.rename(columns=rename_map)
        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}
        return cls(**kwargs)

    def __init__(
        self,
        minimum_time: List,
        minimum_time_error: Optional[List] = None,
        weights: Optional[List] = None,
        minimum_type: Optional[BinarySeq] = None,
        labels: Optional[List] = None,
        cycle: Optional[List] = None,
        oc: Optional[List] = None,
    ):
        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, minimum_time)
        fixed_weights = Fixer.length_fixer(weights, minimum_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, minimum_time)
        fixed_labels_to = Fixer.length_fixer(labels, minimum_time)
        fixed_cycle = Fixer.length_fixer(cycle, minimum_time)
        fixed_oc = Fixer.length_fixer(oc, minimum_time)
        self.data = pd.DataFrame(
            {
                "minimum_time": minimum_time,
                "minimum_time_error": fixed_minimum_time_error,
                "weights": fixed_weights,
                "minimum_type": fixed_minimum_type,
                "labels": fixed_labels_to,
                "cycle": fixed_cycle,
                "oc": fixed_oc,
            }
        )

    def __str__(self) -> str:
        return self.data.__str__()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data[item]

        cls = self.__class__ 

        if isinstance(item, int):
            row = self.data.iloc[item]
            return cls(
                minimum_time=[row.get("minimum_time")],
                minimum_time_error=[row.get("minimum_time_error")],
                weights=[row.get("weights")],
                minimum_type=[row.get("minimum_type")],
                labels=[row.get("labels")],
                cycle=[row.get("cycle")] if "cycle" in self.data.columns else None,
                oc=[row.get("oc")] if "oc" in self.data.columns else None,
            )

        filtered = self.data[item]
        return cls(
            minimum_time=filtered["minimum_time"].tolist(),
            minimum_time_error=filtered["minimum_time_error"].tolist() if "minimum_time_error" in filtered.columns else None,
            weights=filtered["weights"].tolist() if "weights" in filtered.columns else None,
            minimum_type=filtered["minimum_type"].tolist() if "minimum_type" in filtered.columns else None,
            labels=filtered["labels"].tolist() if "labels" in filtered.columns else None,
            cycle=filtered["cycle"].tolist() if "cycle" in filtered.columns else None,
            oc=filtered["oc"].tolist() if "oc" in filtered.columns else None,
        )

    def __setitem__(self, key, value) -> None:
        self.data.loc[:, key] = value

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _equal_bins(df: pd.DataFrame, xcol: str, bin_count: int) -> np.ndarray:
        xvals = df[xcol].to_numpy(dtype=float)
        xmin = float(np.min(xvals))
        xmax = float(np.max(xvals))

        bins = np.empty((0, 2), dtype=float)
        total_span = xmax - xmin
        bin_length = total_span / max(1, int(bin_count))

        for k in range(int(bin_count)):
            start = xmin + k * bin_length
            end = xmin + (k + 1) * bin_length if k < bin_count - 1 else xmax
            bins = np.vstack([bins, np.array([[start, end]], dtype=float)])

        return bins

    @staticmethod
    def _smart_bins(
        df: pd.DataFrame,
        xcol: str,
        bin_count: int,
        smart_bin_period: float = 50.0
    ) -> np.ndarray:
        if smart_bin_period is None or smart_bin_period <= 0:
            raise ValueError("smart_bin_period must be a positive number for _smart_bins")

        df_sorted = df.sort_values(by=xcol)
        xvals = df_sorted[xcol].to_numpy(dtype=float)
        xmin = float(np.min(xvals))
        xmax = float(np.max(xvals))

        bins = np.empty((0, 2), dtype=float)
        bin_start = xmin

        gaps = np.diff(xvals)
        big_gaps = gaps > smart_bin_period
        gap_indexes = np.where(big_gaps)[0]

        for i in gap_indexes:
            bins = np.vstack([bins, np.array([[bin_start, float(xvals[i])]], dtype=float)])
            bin_start = float(xvals[i + 1])

        bins = np.vstack([bins, np.array([[bin_start, xmax]], dtype=float)])

        target_bin_count = int(max(1, bin_count))

        if len(bins) > target_bin_count:
            while len(bins) > target_bin_count:
                inter_gaps = bins[1:, 0] - bins[:-1, 1]
                merge_pos = int(np.argmin(inter_gaps))
                merged_segment = np.array([[bins[merge_pos, 0], bins[merge_pos + 1, 1]]], dtype=float)
                bins = np.vstack([bins[:merge_pos], merged_segment, bins[merge_pos + 2:]])

        if int(bin_count) > len(bins):
            lacking_bins = int(bin_count - len(bins))
            lens = (bins[:, 1] - bins[:, 0]).astype(float)
            weights = lens / np.sum(lens) * lacking_bins
            add_counts = weights.astype(int)
            remainder = lacking_bins - int(np.sum(add_counts))

            if remainder > 0:
                rema = weights % 1.0
                top = np.argsort(-rema)[:remainder]
                add_counts[top] += 1

            new_bins = np.empty((0, 2), dtype=float)
            for i, (start, end) in enumerate(bins):
                k = int(add_counts[i])
                if k <= 0:
                    new_bins = np.vstack([new_bins, np.array([[start, end]], dtype=float)])
                else:
                    edges = np.linspace(start, end, k + 2)
                    segs = np.column_stack([edges[:-1], edges[1:]])
                    new_bins = np.vstack([new_bins, segs])

            bins = new_bins

        return bins

    def bin(
        self,
        bin_count: int = 1,
        bin_method: Optional[ArrayReducer] = None,
        bin_error_method: Optional[ArrayReducer] = None,
        bin_style: Optional[Callable[[pd.DataFrame, int], np.ndarray]] = None,
    ) -> Self:
        if "cycle" in self.data.columns:
            xcol = "cycle"
        else:
            raise ValueError("`OC.bin` needs or 'cycle' column as x-axis.")

        if "oc" not in self.data.columns:
            raise ValueError("`oc` column is required")

        if "weights" not in self.data.columns:
            raise ValueError("`weights` column is required")

        if self.data["weights"].hasnans:
            raise ValueError("`weights` contain NaN values")

        if self.data[xcol].hasnans:
            raise ValueError(f"`{xcol}` contain NaN values")

        def mean_binner(array: np.ndarray, weights: np.ndarray) -> float:
            return float(np.average(array, weights=weights))

        def error_binner(weights: np.ndarray) -> float:
            return float(1.0 / np.sqrt(np.sum(weights)))

        if bin_method is None:
            bin_method = mean_binner
        if bin_error_method is None:
            bin_error_method = error_binner

        if bin_style is None:
            bins = self._equal_bins(self.data, xcol, int(bin_count))
        else:
            bins = bin_style(self.data, int(bin_count))

        binned_x: List[float] = []
        binned_ocs: List[float] = []
        binned_errors: List[float] = []

        n_bins = len(bins)
        for i, (start, end) in enumerate(bins):
            if i < n_bins - 1:
                mask = (self.data[xcol] >= start) & (self.data[xcol] < end)
            else:
                mask = (self.data[xcol] >= start) & (self.data[xcol] <= end)

            if not np.any(mask):
                continue

            w = self.data["weights"][mask].to_numpy(dtype=float)
            xarray = self.data[xcol][mask].to_numpy(dtype=float)
            ocarray = self.data["oc"][mask].to_numpy(dtype=float)

            binned_x.append(bin_method(xarray, w))
            binned_ocs.append(bin_method(ocarray, w))
            binned_errors.append(bin_error_method(w))

        new_df = pd.DataFrame()
        new_df["minimum_time"] = np.nan
        new_df["minimum_time_error"] = binned_errors
        new_df["weights"] = np.nan
        new_df["minimum_type"] = None
        new_df["labels"] = "Binned"
        new_df["oc"] = binned_ocs

        new_df["cycle"] = binned_x

        cls = self.__class__
        return cls(
            minimum_time=new_df["minimum_time"].tolist(),
            minimum_time_error=new_df["minimum_time_error"].tolist(),
            weights=new_df["weights"].tolist(),
            minimum_type=new_df["minimum_type"].tolist(),
            labels=new_df["labels"].tolist(),
            cycle=new_df["cycle"].tolist() if "cycle" in new_df.columns else None,
            oc=new_df["oc"].tolist(),
        )

    def merge(self, oc: Self) -> Self:
        from copy import deepcopy
        new_oc = deepcopy(self)
        new_oc.data = pd.concat([self.data, oc.data], ignore_index=True, sort=False)
        return new_oc
    
    def calculate_oc(self, reference_minimum: float, reference_period: float, model_type: str = "lmfit_model") -> Self:
        import numpy as np
        import pandas as pd

        df = self.data.copy()
        t = np.asarray(df["minimum_time"].to_numpy(), dtype=float)
        phase = (t - reference_minimum) / reference_period
        cycle = np.rint(phase)

        if "minimum_type" in df.columns:
            vals = df["minimum_type"].to_numpy()
            sec = np.zeros_like(t, dtype=bool)
            for i, v in enumerate(vals):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                s = str(v).strip().lower()
                if s in {"1", "ii", "sec", "secondary", "s"} or "ii" in s:
                    sec[i] = True
                elif s in {"0", "i", "pri", "primary", "p"}:
                    sec[i] = False
                else:
                    try:
                        n = int(s)
                        sec[i] = (n == 2)
                    except Exception:
                        pass
            if np.any(sec):
                cycle_sec = np.rint(phase - 0.5) + 0.5
                cycle = np.where(sec, cycle_sec, cycle)

        calculated = reference_minimum + cycle * reference_period
        oc = (t - calculated).astype(float).tolist()

        new_data = {
            "minimum_time": df["minimum_time"].tolist(),
            "minimum_time_error": df["minimum_time_error"].tolist() if "minimum_time_error" in df else None,
            "weights": df["weights"].tolist() if "weights" in df else None,
            "minimum_type": df["minimum_type"].tolist() if "minimum_type" in df else None,
            "labels": df["labels"].tolist() if "labels" in df else None,
        }

        if model_type == "lmfit_model":
            try:
                from .oc_lmfit import OCLMFit
                Target = OCLMFit
            except Exception:
                Target = OC
        else:
            Target = OC

        return Target(
            minimum_time=new_data["minimum_time"],
            minimum_time_error=new_data["minimum_time_error"],
            weights=new_data["weights"],
            minimum_type=new_data["minimum_type"],
            labels=new_data["labels"],
            cycle=cycle,
            oc=oc,
        )

    def residue(self, coefficients: "ModelResult") -> Self:
        pass

    def fit(self, functions: Union[List["ModelComponentModel"], "ModelComponentModel"]) -> "ModelResult":
        pass

    def fit_keplerian(self, parameters: List["ParameterModel"]) -> "ModelComponentModel":
        pass

    def fit_lite(self, parameters: List["ParameterModel"]) -> "ModelComponentModel":
        pass

    def fit_linear(self, parameters: List["ParameterModel"]) -> "ModelComponentModel":
        pass

    def fit_quadratic(self, parameters: List["ParameterModel"]) -> "ModelComponentModel":
        pass

    def fit_sinusoidal(self, parameters: List["ParameterModel"]) -> "ModelComponentModel":
        pass
