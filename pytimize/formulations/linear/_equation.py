import numpy as np

from ._constraint import LinearConstraint
from ._utilities import pad_right
from typing import Dict, Union, Optional

class LinearEquation:
    def __init__(self, terms: Dict[int, float], constant: float=0):
        # TODO Verify terms is valid
        self._terms = terms
        self._constant = constant



    def __neg__(self) -> "LinearEquation":
        self._terms = {label: -coefficient for label, coefficient in self._terms.items()}
        
        return self



    def __pos__(self) -> "LinearEquation":
        return self



    def __add__(self, other) -> "LinearEquation":
        if isinstance(other, LinearEquation):
            for label, coefficient in other.terms.items():
                self._terms.setdefault(label, 0)

                self._terms[label] += coefficient
        elif isinstance(other, float) or isinstance(other, int):
            self._constant += other
        elif hasattr(other, "key"):
            self._terms.setdefault(other.key, 0)
            
            self._terms[other.key] += 1

        return self



    def __radd__(self, other) -> "LinearEquation":
        return self + other



    def __iadd__(self, other) -> "LinearEquation":
        return self + other


    def __sub__(self, other) -> "LinearEquation":
        return self + -other



    def __rsub__(self, other) -> "LinearEquation":
        return -self + other



    def __isub__(self, other) -> "LinearEquation":
        return self + -other



    def __mul__(self, other: float) -> "LinearEquation": 
        if isinstance(other, float) or isinstance(other, int): 
            self._terms = {label: coefficient * other for label, coefficient in self._terms.items()}

            return self



    def __rmul__(self, other) -> "LinearEquation":
        return self * other



    def __imult__(self, other) -> "LinearEquation":
        return self * other



    def __le__(self, other: Union["LinearEquation", float]) -> LinearConstraint:
        return self._generate_constraint("<=", other)



    def __ge__(self, other: Union["LinearEquation", float]) -> LinearConstraint:
        return self._generate_constraint(">=", other)



    def __eq__(self, other: Union["LinearEquation", float]) -> LinearConstraint:
        return self._generate_constraint("=", other)



    def _generate_constraint(self, inequality: str, other: Union["LinearEquation", float]) -> LinearConstraint:
        if isinstance(other, float) or isinstance(other, int):
            other = LinearEquation({}, other)

        constant = other.constant - self._constant
        coefficients = self._broadcast_subtract(self.coefficients, other.coefficients)

        if coefficients is None:
            raise ValueError("Constraint cannot be constructed with only constants.")

        return LinearConstraint(coefficients, inequality, constant)



    def _broadcast_subtract(self, array_one: Optional[np.ndarray], array_two: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array_one is None:
            return array_two
        
        if array_two is None:
            return array_one

        array_two = -array_two

        if array_one.shape[0] == array_two.shape[0]:
            return array_one + array_two

        if array_one.shape[0] > array_two.shape[0]:
            array_one, array_two = array_two, array_one
        
        return array_two + pad_right(array_one, array_two.shape[0])



    @property
    def terms(self) -> Dict[int, float]:
        return self._terms



    @property
    def coefficients(self) -> Optional[np.ndarray]:
        if len(self._terms) == 0:
            return None

        coefficients = np.zeros(max(self._terms.keys()) + 1)

        for label, coefficient in self._terms.items():
            coefficients[label] = coefficient

        return coefficients



    @property
    def constant(self) -> float:
        return self._constant
