import numpy as np

from . import LinearEquation, Term
from ._types import LinearEquationLike
from typing import Optional

def to_linear_equation(equation: LinearEquationLike) -> LinearEquation:
    if isinstance(equation, Term):
        equation = equation.to_equation()
    elif isinstance(equation, float) or isinstance(equation, int):
        equation = LinearEquation({}, equation)

    return equation



def pad_right(array: Optional[np.ndarray], max_length: int) -> np.ndarray:
    if max_length < 0:
        raise ValueError("Max length cannot be less than zero.")

    if array is None:
        return np.zeros(max_length)

    if array.shape[0] > max_length:
        raise ValueError("Length of given array exceeds max length.")

    if array.shape[0] == max_length:
        return array

    return np.r_[array, np.zeros(max_length - array.shape[0])]
