import numpy as np

from . import LinearConstraint, LinearEquation, Term, AbstractLinearProgram
from typing import Union

class ObjectiveFunction:
    def __init__(self, equation: LinearEquation, objective: str):
        self._c = equation.coefficients
        self._z = equation.constant
        self._objective = objective



    def subject_to(self, *constraints: LinearConstraint) -> AbstractLinearProgram:    
        A = []
        b = []
        inequalities = list(map(lambda constraint: constraint.inequality, constraints))
        return AbstractLinearProgram(A, b, self._c, self._z, self._objective, inequalities)



    def fill(self, length):
        pass



def minimize(equation: Union[LinearConstraint, Term]) -> ObjectiveFunction:
    if isinstance(equation, Term):
        equation = equation.to_equation()
    return ObjectiveFunction(equation, "min")



def maximize(equation: Union[LinearEquation, Term]) -> ObjectiveFunction:
    if isinstance(equation, Term):
        equation = equation.to_equation()
    return ObjectiveFunction(equation, "max")
