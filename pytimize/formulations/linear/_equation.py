import numpy as np

from ._constraint import LinearConstraint
from ._utilities import pad_right
from typing import Dict, Union, Optional

class LinearEquation:
    def __init__(self, terms: Dict[int, float], constant: float=0):
        # TODO Verify terms is valid
        self._terms = terms
        self._constant = constant



    def __repr__(self):
        pass # TODO



    def __neg__(self):
        self._coefficient = -self._coefficient
        
        return self



    def __pos__(self):
        return self



    def __add__(self, other):
        # if isinstance(other, Term):
        #     if other.label == self._label:
        #         self._coefficient += other.coefficient
        #         return self
            
        #     return LinearEquation({ 
        #         self._label: self._coefficient,
        #         other.label: other.coefficient
        #     })
        # elif isinstance(other, LinearEquation):
        #     return other.add_variable(self._label, self._coefficient)
        # elif isinstance(other, int) or isinstance(other, float): 
        #     return LinearEquation({ self._label: self._coefficient }, other)
        # raise Exception()
        pass



    def __radd__(self, other):
        pass



    def __iadd__(self, other):
        pass



    def __mul__(self, other: float): 
        if isinstance(other, float) or isinstance(other, int): 
            self._coefficient *= other
            return self



    def __rmul__(self, other):
        return self * other



    def __imult__(self, other):
        return self * other



    def __le__(self, other: Union["LinearEquation", float]) -> LinearConstraint:
        return self._generate_constraint("<=", other)



    def __ge__(self, other: Union["LinearEquation", float]) -> LinearConstraint:
        return self._generate_constraint(">=", other)



    def __eq__(self, other: Union["LinearEquation", float]) -> LinearConstraint:
        return self._generate_constraint("=", other)



    def add_variable(self, label: int, coefficient: float) -> None:
        previous = self._terms.setdefault(label, 1)
        previous[label] = previous + coefficient



    def compile(self):
        pass # TODO 



    def _generate_constraint(self, inequality: str, other: Union["LinearEquation", float]) -> LinearConstraint:
        if isinstance(other, float):
            other = LinearEquation({}, other)

        constant = other.constant + self._constant
        coefficients = self._broadcast_add(self.coefficients, other.coefficients)

        if coefficients is None:
            raise ValueError("Constraint cannot be constructed with only constants.")

        return LinearConstraint(coefficients, inequality, constant)



    def _broadcast_add(self, array_one: Optional[np.ndarray], array_two: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array_one is None or array_two is None:
            return array_one or array_two
        
        if array_one.shape[0] == array_two.shape[0]:
            return array_one + array_two

        if array_one.shape[0] > array_two.shape[0]:
            array_one, array_two = array_two, array_one
        
        return array_two + pad_right(array_one, array_two.shape[0])



    @property
    def coefficients(self) -> Optional[np.ndarray]:
        if len(self._terms) == 0:
            return None

        coefficients = np.zeros(max(self._terms.keys()))

        for label, coefficient in self._terms.items():
            coefficients[label - 1] = coefficient

        return coefficients



    @property
    def constant(self) -> float:
        return self._constant
