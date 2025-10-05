from abc import abstractmethod, ABC

from typing import Union, Optional, Dict, Self, Callable, List, Any
from lmfit.model import ModelResult

from pathlib import Path
import numpy as np
import pandas as pd

from ocpy.custom_types import ArrayReducer


class ParameterModel(ABC):
    value: float
    min: float
    max: float
    std: float
    fixed: bool = False


class ModelComponentModel(ABC):
    @abstractmethod
    def __init__(self, args: Optional[List[ParameterModel]] = None) -> None:
        """Constructor method"""

    @abstractmethod
    def model_function(self) -> Any:
        """Definition of the function to fit"""


class OCModel(ABC):
    @classmethod
    @abstractmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        """Read data from file"""

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
    def merge(self, oc: Self) -> Self:
        """Appends oc to this oc"""

    @abstractmethod
    def residue(self, coefficients: ModelResult) -> Self:
        """Removes the fit from current data"""

    @abstractmethod
    def fit(self, functions: Union[List[ModelComponentModel], ModelComponentModel]) -> ModelResult:
        """Fits the given ModelComponents to the O-C"""

    @abstractmethod
    def fit_keplerian(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a keplerian fit (also known as lite)"""

    @abstractmethod
    def fit_lite(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a lite fit (also known as keplerian fit)"""

    @abstractmethod
    def fit_linear(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a linear fit"""

    @abstractmethod
    def fit_quadratic(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a quadratic fit"""

    @abstractmethod
    def fit_sinusoidal(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a sinusoidal fit"""
