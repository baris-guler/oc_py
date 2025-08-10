from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from typing_extensions import Self

from astropy.time import Time, TimeDelta
from numpy._typing import NDArray

from ocpy.types import BinarySeq, ArrayReducer, OC


class DataModel(ABC):
    @abstractmethod
    def __init__(self, minimum_time: Time, minimum_time_error: Optional[TimeDelta] = None,
                 weights: Optional[List] = None, minimum_type: BinarySeq = None,
                 labels: Optional[List] = None) -> None:
        """Constructor method of data class"""

    @abstractmethod
    def __getitem__(self, item) -> Self:
        """Get item works"""

    @classmethod
    @abstractmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict] = None) -> Self:
        """Read data from file"""

    @abstractmethod
    def fill_errors(self, errors: Union[List, Tuple, NDArray], override: bool = False) -> None:
        """Fills th errors"""

    @abstractmethod
    def fill_weights(self, weights: Union[List, Tuple, NDArray], override: bool = False) -> None:
        """Fills th weights"""

    @abstractmethod
    def bin(self, bin_count: int = 1, smart_bin_period: Optional[float] = None,
            method: Optional[ArrayReducer] = None) -> Self:
        """Bins the data and returns each a new Self"""

    @abstractmethod
    def oc(self, period: Time, minimum: Optional[Time] = None) -> OC:
        """Calculates the O-C for this Data"""

    @abstractmethod
    def merge(self, data: Self) -> None:
        """Appends data to this DataModel"""

    def group_by(self, column: Union[str, int]) -> List[Self]:
        """Group data by column's data"""
