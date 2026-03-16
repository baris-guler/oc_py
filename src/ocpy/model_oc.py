from abc import abstractmethod, ABC

from typing import Union, Optional, Dict, Self, Callable, List, Any
from lmfit.model import ModelResult

from pathlib import Path
import numpy as np
import pandas as pd

from ocpy.custom_types import ArrayReducer
from ocpy.custom_types import ArrayReducer, BinarySeq


class ParameterModel(ABC):
    value: float
    min: float
    max: float
    std: float
    fixed: bool = False


class ModelComponentModel(ABC):
    @abstractmethod
    def __init__(self, args: Optional[List[ParameterModel]] = None) -> None:
        ...

    @abstractmethod
    def model_function(self) -> Any:
        ...


class OCModel(ABC):
    @classmethod
    @abstractmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        ...

    @abstractmethod
    def bin(
            self,
            bin_count: int = 1,
            bin_method: Optional[ArrayReducer] = None,
            bin_error_method: Optional[ArrayReducer] = None,
            bin_style: Optional[Callable[[pd.DataFrame, int], np.ndarray]] = None
    ) -> Self:
        ...

    def __init__(
        self,
        minimum_time: List,
        minimum_time_error: Optional[List] = None,
        weights: Optional[List] = None,
        minimum_type: Optional[BinarySeq] = None,
        labels: Optional[List] = None,
        ecorr: Optional[List] = None,
        oc: Optional[List] = None,):
        ...

    @abstractmethod
    def merge(self, oc: Self) -> Self:
        ...

    @abstractmethod
    def residue(self, coefficients: ModelResult) -> Self:
        ...

    @abstractmethod
    def fit(self, model_components: Union[List[ModelComponentModel], ModelComponentModel]) -> ModelResult:
        ...

    @abstractmethod
    def fit_keplerian(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        ...

    @abstractmethod
    def fit_lite(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        ...

    @abstractmethod
    def fit_linear(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        ...

    @abstractmethod
    def fit_quadratic(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        ...

    @abstractmethod
    def fit_sinusoidal(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        ...
