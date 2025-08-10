from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple

from astropy.table import QTable
from astropy.time import Time, TimeDelta
from numpy._typing import NDArray
from typing_extensions import Self

from ocpy.model_data import DataModel
from ocpy.types import ArrayReducer, OC, BinarySeq
from ocpy.utils import Fixer


class DataExample(DataModel):
    def __init__(self, minimum_time: Time, minimum_time_error: Optional[TimeDelta] = None,
                 weights: Optional[List] = None, minimum_type: BinarySeq = None,
                 labels: Optional[List] = None) -> None:
        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, minimum_time)
        fixed_weights = Fixer.length_fixer(weights, minimum_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, minimum_time)
        fixed_labels_to = Fixer.length_fixer(labels, minimum_time)

        self.data = QTable(
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

    def __getitem__(self, item) -> Self:
        if isinstance(item, str):
            return self.data[item]
        elif isinstance(item, int):
            row = self.data[item]
            return DataExample(
                minimum_time=Time([row["minimum_time"].jd], format="jd"),
                minimum_time_error=TimeDelta([row["minimum_time_error"].jd], format="jd"),
                weights=[row["weights"]],
                minimum_type=[row["minimum_type"]],
                labels=[row["labels"]]
            )
        else:
            filtered_table = self.data[item]

            return DataExample(
                minimum_time=filtered_table["minimum_time"],
                minimum_time_error=filtered_table["minimum_time_error"],
                weights=filtered_table["weights"],
                minimum_type=filtered_table["minimum_type"],
                labels=filtered_table["labels"]
            )

    @classmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict] = None) -> Self:
        pass

    def fill_errors(self, errors: Union[List, Tuple, NDArray], override: bool = False) -> None:
        pass

    def fill_weights(self, weights: Union[List, Tuple, NDArray], override: bool = False) -> None:
        pass

    def bin(self, bin_count: int = 1, smart_bin_period: Optional[float] = None,
            method: Optional[ArrayReducer] = None) -> Self:
        pass

    def oc(self, period: Time, minimum: Optional[Time] = None) -> OC:
        pass

    def merge(self, data: Self) -> None:
        pass

    def group_by(self, column: Union[str, int]) -> List[Self]:
        pass