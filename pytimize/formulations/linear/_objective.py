import numpy as np

from . import LinearConstraint, LinearEquation, Term, AbstractLinearProgram
from ._types import LinearEquationLike
from ._utilities import to_linear_equation, pad_right

class ObjectiveFunction:
    def __init__(self, equation: LinearEquation, objective: str):
        self._c = equation.coefficients
        self._z = equation.constant
        self._objective = objective



    def __repr__(self):
        pass # TODO



    def subject_to(self, *constraints: LinearConstraint) -> AbstractLinearProgram:
        inequalities = []
        length = max(map(lambda constraint: constraint.coefficient.shape[0], constraints))
        A = np.zeros(length)
        b = np.zeros(1)

        for constraint in constraints:
            inequalities.append(constraint.inequality)
            
            A = np.vstack(A, pad_right(constraint.coefficients, length))
            b = np.r_[b, constraint.constant]

        return AbstractLinearProgram(A, b, self._c, self._z, self._objective, inequalities)



    def fill(self, length):
        pass # TODO



def minimize(equation: LinearEquationLike) -> ObjectiveFunction:
    return ObjectiveFunction(to_linear_equation(equation), "min")



def maximize(equation: LinearEquationLike) -> ObjectiveFunction:
    return ObjectiveFunction(to_linear_equation(equation), "max")
