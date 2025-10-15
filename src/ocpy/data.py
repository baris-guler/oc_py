from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Callable, Literal

from numpy._typing import NDArray
from typing_extensions import Self

import pandas as pd
import numpy as np
from copy import deepcopy

from ocpy.model_data import DataModel
from ocpy.custom_types import ArrayReducer, BinarySeq
from ocpy.utils import Fixer

from .errors import LengthCheckError
import warnings
from .oc import OC
from .oc_lmfit import OCLMFit


class Data(DataModel):
    def __init__(
            self, minimum_time: List,
            minimum_time_error: Optional[List] = None,
            weights: Optional[List] = None,
            minimum_type: Optional[BinarySeq] = None,
            labels: Optional[List] = None,
    ) -> None:
        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, minimum_time)
        fixed_weights = Fixer.length_fixer(weights, minimum_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, minimum_time)
        fixed_labels_to = Fixer.length_fixer(labels, minimum_time)

        self.data = pd.DataFrame(
            {
                "minimum_time": minimum_time,
                "minimum_time_error": fixed_minimum_time_error,
                "weights": fixed_weights,
                "minimum_type": fixed_minimum_type,
                "labels": fixed_labels_to,
            }
        )

    def __str__(self) -> str:
        return self.data.__str__()

    def __getitem__(self, item) -> Union[Self, pd.Series]:
        if isinstance(item, str):
            # str ise pd.series döndürüyor ne yapsam bilemedim. Bunu konuşalım...
            return self.data[item]
        elif isinstance(item, int):
            row = self.data.iloc[item]
            return Data(
                minimum_time=[row["minimum_time"]],
                minimum_time_error=[row["minimum_time_error"]],
                weights=[row["weights"]],
                minimum_type=[row["minimum_type"]],
                labels=[row["labels"]],
            )
        else:
            filtered_table = self.data[item]

            return Data(
                minimum_time=filtered_table["minimum_time"],
                minimum_time_error=filtered_table["minimum_time_error"],
                weights=filtered_table["weights"],
                minimum_type=filtered_table["minimum_type"],
                labels=filtered_table["labels"],
            )

    def __setitem__(self, key, value) -> None:
        self.data.loc[:, key] = value

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        file_path = Path(file)

        # TODO change file reading (do without extension) 
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Use `csv`, `xls`, or `xlsx` instead")

        expected = ["minimum_time", "minimum_time_error", "weights", "minimum_type", "labels"]
        if columns:
            if any(k in expected for k in columns.keys()):
                rename_map = {v: k for k, v in columns.items()}
            else:
                rename_map = columns
            df = df.rename(columns=rename_map)

        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}

        return cls(**kwargs)

    def _assign_or_fill(self, df: pd.DataFrame, col: str, values, override: bool) -> None:
        """
        If override=True or column doesn't exist, assign values directly.
        Otherwise, fill only where existing entries are NaN.
        """
        if override or col not in df.columns:
            df[col] = values
        else:
            base = df[col]
            df[col] = base.where(~pd.isna(base), values)

    def fill_errors(self, errors: Union[List, Tuple, np.ndarray, float], override: bool = False) -> Self:
        new_data = deepcopy(self)
        if isinstance(errors, (list, tuple, np.ndarray)) and len(errors) != len(new_data.data):
            raise LengthCheckError("Length of `errors` must be equal to the length of the data")
        self._assign_or_fill(new_data.data, "minimum_time_error", errors, override)
        return new_data

    def fill_weights(self, weights: Union[List, Tuple, np.ndarray, float], override: bool = False) -> Self:
        new_data = deepcopy(self)
        if isinstance(weights, (list, tuple, np.ndarray)) and len(weights) != len(new_data.data):
            raise LengthCheckError("Length of `weights` must be equal to the length of the data")
        self._assign_or_fill(new_data.data, "weights", weights, override)
        return new_data

    def calculate_weights(self, method: Callable[[pd.Series], pd.Series] = None, override: bool = True) -> Self:
        def inverse_variance_weights(err_days: pd.Series) -> pd.Series:
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / np.square(err_days)

        new_data = deepcopy(self)
        minimum_time_error = new_data.data["minimum_time_error"]

        if minimum_time_error.hasnans:
            raise ValueError("minimum_time_error contains NaN value(s)")
        if (minimum_time_error == 0).any():
            raise ValueError("minimum_time_error contains `0`")

        if method is not None and not callable(method):
            raise TypeError("`method` must be callable or None for inverse variance weights")

        if method is None:
            method = inverse_variance_weights

        weights = method(minimum_time_error)
        self._assign_or_fill(new_data.data, "weights", weights, override)
        return new_data
    
    """
    @staticmethod
    def _equal_bins(data_dataframe: pd.DataFrame, bin_count: int) -> np.ndarray:
        ecorr_vals = data_dataframe["ecorr"].to_numpy(dtype=float)
        ecorr_min = float(np.min(ecorr_vals))
        ecorr_max = float(np.max(ecorr_vals))

        bins = np.empty((0, 2), dtype=float)
        total_span = ecorr_max - ecorr_min
        bin_length = total_span / max(1, int(bin_count))

        for k in range(int(bin_count)):
            start = ecorr_min + k * bin_length
            end = ecorr_min + (k + 1) * bin_length if k < bin_count - 1 else ecorr_max
            bins = np.vstack([bins, np.array([[start, end]], dtype=float)])

        return bins

    @staticmethod
    def _smart_bins(data_dataframe: pd.DataFrame, bin_count: int, smart_bin_period: float = 50) -> np.ndarray:
        if smart_bin_period is None or smart_bin_period <= 0:
            raise ValueError("smart_bin_period must be a positive number for _smart_bins")

        df_sorted = data_dataframe.sort_values(by="ecorr")
        ecorr_vals = df_sorted["ecorr"].to_numpy(dtype=float)
        ecorr_min = float(np.min(ecorr_vals))
        ecorr_max = float(np.max(ecorr_vals))

        bins = np.empty((0, 2), dtype=float)
        bin_start = ecorr_min

        gaps = np.diff(ecorr_vals)
        big_gaps = gaps > smart_bin_period
        gap_indexes = np.where(big_gaps)[0]

        for i in gap_indexes:
            bins = np.vstack([bins, np.array([[bin_start, float(ecorr_vals[i])]], dtype=float)])
            bin_start = float(ecorr_vals[i + 1])

        bins = np.vstack([bins, np.array([[bin_start, ecorr_max]], dtype=float)])

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

    def bin(self,
            bin_count: int = 1,
            bin_method: Optional[ArrayReducer] = None,
            bin_error_method: Optional[ArrayReducer] = None,
            bin_style: Optional[Callable[[pd.DataFrame, int], np.ndarray]] = None) -> Self:

        def mean_binner(array: NDArray, weights: NDArray) -> float:
            return float(np.average(array, weights=weights))

        def error_binner(weights: NDArray) -> float:
            return float(1.0 / np.sqrt(np.sum(weights)))

        if self.data["weights"].hasnans:
            raise ValueError("`weights` contain NaN values")

        if self.data["ecorr"].hasnans:
            raise ValueError("`ecorr` contain Nan values")

        new_data = deepcopy(self)

        if bin_method is None:
            bin_method = mean_binner

        if bin_error_method is None:
            bin_error_method = error_binner

        if (bin_style is None):
            bins = self._equal_bins(new_data.data, int(bin_count))
        else:
            bins = bin_style(new_data.data, int(bin_count))

        binned_ecorrs: list[float] = []
        binned_ocs: list[float] = []
        binned_errors: list[float] = []

        n_bins = len(bins)

        for i, (start, end) in enumerate(bins):
            if i < n_bins - 1:
                mask = (new_data.data["ecorr"] >= start) & (new_data.data["ecorr"] < end)
            else:
                mask = (new_data.data["ecorr"] >= start) & (new_data.data["ecorr"] <= end)

            if not np.any(mask):
                continue

            weights = new_data.data["weights"][mask]
            binned_ecorrs.append(bin_method(new_data.data["ecorr"][mask], weights))
            binned_ocs.append(bin_method(new_data.data["oc"][mask], weights))
            binned_errors.append(bin_error_method(weights))

        new_data_df = pd.DataFrame()
        new_data_df["minimum_time"] = np.nan
        new_data_df["minimum_time_error"] = binned_errors
        new_data_df["weights"] = np.nan
        new_data_df["minimum_type"] = None
        new_data_df["labels"] = "Binned"
        new_data_df["ecorr"] = binned_ecorrs
        new_data_df["oc"] = binned_ocs

        new_data.data = new_data_df

        return new_data
    """

    def calculate_oc(self, reference_minimum: float, reference_period: float, model_type: str = "lmfit_model") -> OC:
        df = self.data.copy()

        if "minimum_time" not in df.columns:
            raise ValueError("`minimum_time` column is required to compute O–C.")

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

        new_data: Dict[str, Optional[list]] = {
            "minimum_time": df["minimum_time"].tolist(),
            "minimum_time_error": df["minimum_time_error"].tolist() if "minimum_time_error" in df else None,
            "weights": df["weights"].tolist() if "weights" in df else None,
            "minimum_type": df["minimum_type"].tolist() if "minimum_type" in df else None,
            "labels": df["labels"].tolist() if "labels" in df else None,
        }

        cycle = cycle

        common_kwargs = dict(
            minimum_time=new_data["minimum_time"],
            minimum_time_error=new_data["minimum_time_error"],
            weights=new_data["weights"],
            minimum_type=new_data["minimum_type"],
            labels=new_data["labels"],
            cycle=cycle,
            oc=oc,
        )

        cls_map = {
            "lmfit_model": OCLMFit,
            # "mcmc_model": OC_PyMC,
        }
        Target = cls_map.get(model_type, OC)

        return Target(**common_kwargs)

    def merge(self, data: Self) -> Self:
        new_data = deepcopy(self)
        new_data.data = pd.concat([self.data, data.data], ignore_index=True, sort=False)
        return new_data

    def group_by(self, column: str) -> List["Data"]:
        if column not in self.data.columns:
            return [deepcopy(self)]

        s = self.data[column]

        if s.isna().all():
            return [deepcopy(self)]

        groups: List["Data"] = []

        for _, df_group in self.data.groupby(s, dropna=False):
            new_obj = deepcopy(self)
            new_obj.data = df_group.copy()
            groups.append(new_obj)

        return groups