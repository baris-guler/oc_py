from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Callable
from typing_extensions import Self

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from ocpy.custom_types import BinarySeq, ArrayReducer


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
        """Constructor method of data class"""

    @abstractmethod
    def __getitem__(self, item) -> Union[Self, pd.Series]:
        """Get item works"""

    @classmethod
    @abstractmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        """Read data from file"""

    @abstractmethod
    def fill_errors(self, errors: Union[List, Tuple, NDArray, float], override: bool = False) -> Self:
        """Fills th errors"""

    @abstractmethod
    def fill_weights(self, weights: Union[List, Tuple, NDArray, float], override: bool = False) -> Self:
        """Fills th weights"""

    @abstractmethod
    def calculate_weights(
        self,
        method: Optional[Callable[[pd.Series], pd.Series]] = None,
        override: bool = True
    ) -> Self:
        """Calculates weights using errors"""

    @staticmethod
    @abstractmethod
    def _equal_bins(data_dataframe: pd.DataFrame, bin_count: int) -> np.ndarray:
        """Produce equal-length bins on ecorr"""

    @abstractmethod
    def bin(
        self,
        bin_count: int = 1,
        bin_method: Optional[ArrayReducer] = None,
        bin_error_method: Optional[ArrayReducer] = None,
        bin_style: Optional[Callable[[pd.DataFrame, int], np.ndarray]] = None
    ) -> Self:
        """Bins the data and returns each a new Self"""

    @abstractmethod
    def calculate_oc(self, p0: float, t0: float) -> Self:
        """Calculates the O-C for this Data"""

    @abstractmethod
    def merge(self, data: Self) -> Self:
        """Appends data to this DataModel"""

    @abstractmethod
    def group_by(self, column: Union[str, int]) -> List[Self]:
        """Group data by column's data"""