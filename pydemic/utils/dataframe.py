from typing import Union, Sequence

import numpy as np
import pandas as pd

Data = Union[Sequence, pd.Series, np.ndarray]


def trim_zeros(data: Union[pd.Series, Sequence]) -> Data:
    """
    Remove null values from the start and end of series or array.
    """

    i = j = 0
    for i, x in enumerate(data):
        if x != 0:
            break
    for j, x in enumerate(reversed(data)):
        if x != 0:
            break
    j = len(data) - j

    if isinstance(data, pd.Series):
        return data.iloc[i:j]
    return data[i:j]
