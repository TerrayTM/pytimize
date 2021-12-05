import numpy as np

from ...programs import LinearProgram
from ._constraint import VariableConstraint
from typing import List, Any

class AbstractLinearProgram:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, z: float, objective: str, inequalities: List[str]):
        self._A = A
        self._b = b
        self._c = c
        self._z = z
        self._objective = objective
        self._inequalities = inequalities
        self._negative_variables = []
        self._positive_variables = []
        self._program = None
        self._recompile = False



    def __repr__(self) -> str:
        return str(self.program)



    def __getattr__(self, key: str) -> Any:
        return getattr(self.program, key)



    def where(self, *constraints: VariableConstraint) -> "AbstractLinearProgram":
        for constraint in constraints:
            self._add_variable_constraint(constraint.positive_variables, self._positive_variables)
            self._add_variable_constraint(constraint.negative_variables, self._negative_variables)

        positive_set = set(self._positive_variables)
        negative_set = set(self._negative_variables)

        if not len(positive_set) == len(self._positive_variables) or not len(negative_set) == len(self._negative_variables):
            raise ValueError("Duplicated variable constraints detected.")

        if len(positive_set.intersection(negative_set)) > 0:
            raise ValueError("Variable cannot be both positive and negative.")
        
        self._recompile = True

        return self

    
    def whereAllPositive(self) -> "AbstractLinearProgram":
        if len(self._negative_variables) + len(self._positive_variables) > 0:
            raise ValueError("Variable constraints must be unset.")

        self._add_variable_constraint([], self._positive_variables)
        self._recompile = True
        
        return self



    def whereAllNegative(self) -> "AbstractLinearProgram":
        if len(self._negative_variables) + len(self._positive_variables) > 0:
            raise ValueError("Variable constraints must be unset.")

        self._add_variable_constraint([], self._negative_variables)
        self._recompile = True

        return self



    def extend(self, length) -> "AbstractLinearProgram":
        self._recompile = True
        pass



    def _add_variable_constraint(self, source, target):
        if source is not None:
            if len(source) == 0:
                target.extend(list(range(1, self._c.shape[0] + 1)))
            else:
                target.extend(source)



    @property
    def program(self) -> LinearProgram: 
        if not self._program or self._recompile:
            union = set(self._negative_variables + self._positive_variables)
            free_variables = list(filter(lambda x: x not in union, range(1, self._c.shape[0] + 1)))

            self._program = LinearProgram(
                self._A, 
                self._b, 
                self._c, 
                self._z, 
                self._objective, 
                self._inequalities,
                free_variables,
                self._negative_variables
            )
            self._recompile = False

        return self._program
