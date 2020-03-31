import numpy as np

from ._utilities import pad_right
from ._constraint import LinearConstraint
from ._equation import LinearEquation
from ._abstract import AbstractLinearProgram
from typing import Union

class ObjectiveFunction:
    def __init__(self, equation: LinearEquation, objective: str):
        self._c = equation.coefficients
        self._z = equation.constant
        self._objective = objective



    def __repr__(self):
        pass # TODO


    # TODO for unconstrained programs, user can directly call .where or .compile
    def subject_to(self, *constraints: LinearConstraint) -> AbstractLinearProgram:
        if constraints is None: 
            raise ValueError("At least one constraint must be provided.")

        inequalities = []
        length = max(constraint.coefficients.shape[0] for constraint in constraints)
        A = np.empty(length)
        b = np.empty(1)

        for constraint in constraints:
            inequalities.append(constraint.inequality)
            
            A = np.vstack((A, pad_right(constraint.coefficients, length)))
            b = np.r_[b, constraint.constant]

        A = A[1:]
        b = b[1:]

        return AbstractLinearProgram(A, b, pad_right(self._c, length), self._z, self._objective, inequalities)



    def fill(self, length):
        pass # TODO



def minimize(equation: Union[LinearEquation, float]) -> ObjectiveFunction:
    if isinstance(equation, float) or isinstance(equation, int):
        equation = LinearEquation({}, equation)

    return ObjectiveFunction(equation, "min")



def maximize(equation: Union[LinearEquation, float]) -> ObjectiveFunction:
    if isinstance(equation, float) or isinstance(equation, int):
        equation = LinearEquation({}, equation)

    return ObjectiveFunction(equation, "max")
