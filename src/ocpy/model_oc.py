from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Optional, Dict, Self, Callable, Literal

import numpy as np
import pandas as pd

from ocpy.custom_types import ArrayReducer


class OCModel(ABC):
    @classmethod
    @abstractmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        """Read OC from file"""

    @abstractmethod
    def bin(
            self,
            bin_count: int = 1,
            bin_method: Optional[ArrayReducer] = None,
            bin_error_method: Optional[ArrayReducer] = None,
            bin_style: Optional[Callable[[pd.DataFrame, int], np.ndarray]] = None
    ) -> Self:
        """Bins the OC and returns each a new Self"""

    @abstractmethod
    def linear_fit(self, weight: Optional[pd.Series] = None) -> Self:
        """Fits a linear to OC"""

    @abstractmethod
    def sinusoidal_fit(self) -> np.array:
        """Fits a sinusoidal to OC"""

    @abstractmethod
    def quadratic_fit(self, weight: Optional[pd.Series] = None) -> np.array:
        """Fits a parabola to OC"""

    @abstractmethod
    def fit(self, kinds: Literal["linear", "sinusoidal", "quadratic"]) -> np.array:
        """Fits either a linear, quadratic or sinusoidal to OC"""

    @abstractmethod
    def residue(self, coefficients: np.array) -> Self:
        """Calculates the residual of OC from coefficients"""

    @abstractmethod
    def merge(self, oc: Self) -> Self:
        """Merges the given OC by itself"""
