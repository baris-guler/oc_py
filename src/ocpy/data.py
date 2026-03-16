from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Callable

from typing_extensions import Self

import pandas as pd
import numpy as np
from copy import deepcopy

from ocpy.model_data import DataModel
from ocpy.custom_types import BinarySeq
from ocpy.utils import Fixer

from .errors import LengthCheckError
from .oc import OC
from .oc_lmfit import OCLMFit
from .oc_pymc import OCPyMC

class Data(DataModel):
    def __init__(
            self,
            minimum_time: List,
            minimum_time_error: Optional[List] = None,
            weights: Optional[List] = None,
            minimum_type: Optional[BinarySeq] = None,
            labels: Optional[List] = None,
    ) -> None:
        if minimum_time is None:
             raise ValueError("`minimum_time` is required and cannot be None.")

        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, minimum_time)
        fixed_weights = Fixer.length_fixer(weights, minimum_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, minimum_time)
        fixed_labels_to = Fixer.length_fixer(labels, minimum_time)

        minimum_time_sequence = minimum_time if hasattr(minimum_time, "__len__") else [minimum_time]

        self.data = pd.DataFrame(
            {
                "minimum_time": minimum_time_sequence,
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

        if "minimum_time" not in df.columns:
            available = list(df.columns)
            raise ValueError(f"Could not find 'minimum_time' in file columns. Available columns: {available}. "
                             f"Please check your 'columns' mapping.")

        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}

        return cls(**kwargs)

    def _assign_or_fill(self, df: pd.DataFrame, col: str, values, override: bool) -> None:
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

    def calculate_oc(self, reference_minimum: float, reference_period: float, model_type: str = "lmfit") -> OC:
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

        common_kwargs = dict(
            minimum_time=new_data["minimum_time"],
            minimum_time_error=new_data["minimum_time_error"],
            weights=new_data["weights"],
            minimum_type=new_data["minimum_type"],
            labels=new_data["labels"],
            cycle=cycle,
            oc=oc,
        )

        targets = str(model_type).strip().lower()
        if targets in {"lmfit", "lmfit_model"}:
            Target = OCLMFit
        elif targets in {"pymc", "pymc_model"}:
            Target = OCPyMC
        else:
            Target = OC

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
    
