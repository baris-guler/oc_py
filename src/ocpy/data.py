from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Literal, Callable

from astropy.table import QTable
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


class Data(DataModel):
    def __init__(
        self, minimum_time: List,
        minimum_time_error: Optional[List] = None,
        weights: Optional[List] = None,
        minimum_type: Optional[BinarySeq] = None,
        labels: Optional[List] = None,
        ecorr: Optional[List] = None,
        oc: Optional[List] = None,
    ) -> None:
        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, minimum_time)
        fixed_weights = Fixer.length_fixer(weights, minimum_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, minimum_time)
        fixed_labels_to = Fixer.length_fixer(labels, minimum_time)
        fixed_ecorr = Fixer.length_fixer(ecorr, minimum_time)
        fixed_oc = Fixer.length_fixer(oc, minimum_time)


        self.data = pd.DataFrame(
            {
                "minimum_time": minimum_time,
                "minimum_time_error": fixed_minimum_time_error,
                "weights": fixed_weights,
                "minimum_type": fixed_minimum_type,
                "labels": fixed_labels_to,
                "ecorr": fixed_ecorr,
                "oc": fixed_oc
            }
        )

    def __str__(self) -> str:
        return self.data.__str__()

    def __getitem__(self, item) -> Self:
        if isinstance(item, str):
            # str ise pd.series döndürüyor ne yapsam bilemedim
            return self.data[item]
        elif isinstance(item, int):
            row = self.data.iloc[item] 
            return Data(
                minimum_time=[row["minimum_time"]],
                minimum_time_error=[row["minimum_time_error"]],
                weights=[row["weights"]],
                minimum_type=[row["minimum_type"]],
                labels=[row["labels"]],
                ecorr=[row["ecorr"]],
                oc=[row["oc"]],
            )
        else:
            filtered_table = self.data[item]

            return Data(
                minimum_time=filtered_table["minimum_time"],
                minimum_time_error=filtered_table["minimum_time_error"],
                weights=filtered_table["weights"],
                minimum_type=filtered_table["minimum_type"],
                labels=filtered_table["labels"],
                ecorr=filtered_table["ecorr"],
                oc=filtered_table["oc"]
            )
        
    def __setitem__(self, key, value) -> None:
        self.data.loc[:, key] = value

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

        expected = ["minimum_time", "minimum_time_error", "weights", "minimum_type", "labels", "ecorr", "oc"]
        if columns:
            if any(k in expected for k in columns.keys()):
                rename_map = {v: k for k, v in columns.items()}
            else:
                rename_map = columns
            df = df.rename(columns=rename_map)

        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}

        return cls(**kwargs)

    def fill_errors(self, errors: Union[List, Tuple, NDArray, float], override: bool = False) -> Self:
        # Eğer Liste verildiyse ve override değilse sadece listedeki o element için işlem yapıyor böyle mi olmalı emin değilim
        new_data = deepcopy(self)
        td_error_series = new_data.data["minimum_time_error"] 

        if isinstance(errors, (list, tuple, np.ndarray)):
            if len(errors) != len(new_data.data):
                raise LengthCheckError("Length of `errors` must be equal to the length of the data")

        if override:
           new_data.data["minimum_time_error"] = errors
        else:
            mask = pd.isna(new_data.data["minimum_time_error"])
            new_data.data["minimum_time_error"] = td_error_series.where(~mask, errors)
        return new_data
    
    def fill_weights(self, weights: Union[List, Tuple, NDArray, float], override: bool = False) -> Self:
        new_data = deepcopy(self)
        td_error_series = new_data.data["weights"] 

        if isinstance(weights, (list, tuple, np.ndarray)):
            if len(weights) != len(new_data.data):
                raise LengthCheckError("Length of `weights` must be equal to the length of the data")

        if override:
           new_data.data["weights"] = weights
        else:
            mask = pd.isna(new_data.data["weights"])
            new_data.data["weights"] = td_error_series.where(~mask, weights)
        return new_data
    
    def calculate_weights(self, method: Callable[[pd.Series], pd.Series] = None, override: bool = True) -> Self:
        def inverse_variance_weights(err_days: pd.Series) -> pd.Series:
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / np.square(err_days)
            
        new_data = deepcopy(self)

        minimum_time_error = new_data.data["minimum_time_error"]

        if minimum_time_error.hasnans: 
            warnings.warn(f"minimum_time_error contains NaN value(s)")
        if (minimum_time_error == 0).any():
            warnings.warn(f"minimum_time_error contains `0`")

        if method is not None and not callable(method):
            raise TypeError("`method` must be callable or None for inverse variance weights")

        if method is None:
            method = inverse_variance_weights

        w = method(minimum_time_error)

        if override or "weights" not in new_data.data:
            new_data.data["weights"] = w
        else:
            new_data.data["weights"] = new_data.data["weights"].where(
                ~new_data.data["weights"].isna(), w
            )

        return new_data

    def bin(self, bin_count: int = 1, smart_bin_period: Optional[float] = None,
            bin_method: Optional[ArrayReducer] = None, bin_error_method: Optional[ArrayReducer] = None) -> Self:

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

        bins = np.empty((0, 2), dtype=float)
        ec_min = float(np.min(new_data.data["ecorr"]))
        ec_max = float(np.max(new_data.data["ecorr"]))
        bin_start = ec_min

        if smart_bin_period is None:
            total_span = ec_max - ec_min
            bin_length = total_span / max(1, int(bin_count))
            for k in range(int(bin_count)):
                start = ec_min + k * bin_length
                end = ec_min + (k + 1) * bin_length if k < bin_count - 1 else ec_max
                bins = np.vstack([bins, np.array([[start, end]], dtype=float)])
        else:
            # Smart bin aralarında boşluk olan verileri ayrı ayrı binlemeye yarıyor
            # Bu boşluğun periyotu smart bin period ile belirleniyor
            # Bu bilimsel bir yöntem değil ama gerekli, başka bir fonksiyona ayrılmalı mı bilemedim
            new_data.data = new_data.data.sort_values(by="ecorr")
            gaps = np.diff(new_data.data["ecorr"])
            big_gaps = gaps > smart_bin_period
            gap_indexes = np.where(big_gaps)[0]

            for i in gap_indexes:
                bins = np.vstack([bins, np.array([[bin_start, float(new_data.data["ecorr"][i])]], dtype=float)])
                bin_start = float(new_data.data["ecorr"][i + 1])

            bins = np.vstack([bins, np.array([[bin_start, ec_max]], dtype=float)])

            if bin_count > len(bins):
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
                    w = int(add_counts[i])
                    if w <= 0:
                        new_bins = np.vstack([new_bins, np.array([[start, end]], dtype=float)])
                    else:
                        edges = np.linspace(start, end, w + 2)
                        segs = np.column_stack([edges[:-1], edges[1:]])
                        new_bins = np.vstack([new_bins, segs])
                bins = new_bins

        binned_ecorrs = []
        binned_ocs = []
        binned_errors = []

        n_bins = len(bins)
        for i, (start, end) in enumerate(bins):
            mask = (new_data.data["ecorr"] >= start) & (new_data.data["ecorr"] < end) if i < n_bins - 1 else (new_data.data["ecorr"] >= start) & (new_data.data["ecorr"] <= end)
            if not np.any(mask):
                continue
            w = new_data.data["weights"][mask]
            binned_ecorrs.append(bin_method(new_data.data["ecorr"][mask], w))
            binned_ocs.append(bin_method(new_data.data["oc"][mask], w))
            binned_errors.append(bin_error_method(w))

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
    
    def calculate_oc(self, p0: float, t0: float) -> Self:
        new_data = deepcopy(self)

        mt  = np.asarray(new_data.data["minimum_time"], dtype=float)
        mty = np.asarray(new_data.data["minimum_type"], dtype=int)

        epoch = (mt - float(t0)) / float(p0)

        ecorr = epoch.copy()
        m0 = (mty == 0)
        m1 = (mty == 1)
        ecorr[m0] = np.rint(ecorr[m0])
        ecorr[m1] = np.floor(ecorr[m1]) + 0.5

        oc = mt - (ecorr * float(p0) + float(t0))

        new_data.data.loc[:, "ecorr"] = ecorr
        new_data.data.loc[:, "oc"]    = oc
        return new_data

    def merge(self, data: Self) -> Self:
        # Bunu bir class method yapıp çoklu data birleştirmeyi sağlayabiliriz
        # merge(cls, *datas) -> Self
        if not isinstance(data, Data):
            raise TypeError("merge expects a Data instance")

        new_data = deepcopy(self)
        new_data.data = pd.concat([self.data, data.data], ignore_index=True, sort=False)
        return new_data


    def group_by(self, column: Union[str, int]) -> List[Self]:
        if isinstance(column, int):
            column = self.data.columns[column]

        grouped = self.data.groupby(column)
        result: List[Data] = []

        for _, group in grouped:
            result.append(
                Data(
                    minimum_time=group["minimum_time"],
                    minimum_time_error=group["minimum_time_error"],
                    weights=group["weights"],
                    minimum_type=group["minimum_type"],
                    labels=group["labels"],
                    ecorr=group["ecorr"],
                    oc=group["oc"]
                )
            )

        return result

