from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Callable
from typing_extensions import Self
from ocpy.oc import OC

import pandas as pd
from numpy._typing import NDArray

from ocpy.custom_types import BinarySeq


class DataModel(ABC):
    @abstractmethod
    def __init__(
        self,
        minimum_time: List,
        minimum_time_error: Optional[List] = None,
        weights: Optional[List] = None,
        minimum_type: Optional[BinarySeq] = None,
        labels: Optional[List] = None,
        ecorr: Optional[List] = None,
        oc: Optional[List] = None,
    ) -> None:
        ...

    @abstractmethod
    def __getitem__(self, item) -> Union[Self, pd.Series]:
        ...

    @classmethod
    @abstractmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        ...

    @abstractmethod
    def _assign_or_fill(self, df: pd.DataFrame, col: str, values, override: bool) -> None:
        ...

    @abstractmethod
    def fill_errors(self, errors: Union[List, Tuple, NDArray, float], override: bool = False) -> Self:
        ...

    @abstractmethod
    def fill_weights(self, weights: Union[List, Tuple, NDArray, float], override: bool = False) -> Self:
        ...

    @abstractmethod
    def calculate_weights(
        self,
        method: Optional[Callable[[pd.Series], pd.Series]] = None,
        override: bool = True
    ) -> Self:
        ...

    @abstractmethod
    def calculate_oc(self, reference_minimum: float, reference_period: float, model_type: str = "lmfit") -> OC:
        ...

    @abstractmethod
    def merge(self, data: Self) -> Self:
        ...

    @abstractmethod
    def group_by(self, column: Union[str, int]) -> List[Self]:
        ...
