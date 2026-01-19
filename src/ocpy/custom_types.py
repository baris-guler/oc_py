from typing import Union, List, Tuple, Literal, Callable

import numpy as np
from numpy._typing import NDArray

# A container (List, Tuple, ndarray) that contains only 1s and 0s
BinarySeq = Union[
    List[Literal[0, 1]],
    Tuple[Literal[0, 1], ...],
    NDArray[np.bool_],
]

# A callable that takes a 1D ndarray and returns a numeric value
ArrayReducer = Callable[[NDArray[np.floating]], Union[int, float, np.number]]
NumberOrParam = Union[float, int, None, "Parameter"]
