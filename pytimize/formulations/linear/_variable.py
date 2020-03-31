from ._equation import LinearEquation
from ._constraint import VariableConstraint
from typing import Union

class MetaVariable(type):
    def __getitem__(cls, key: Union[int, slice]) -> LinearEquation:
        if isinstance(key, slice):
            pass
        elif isinstance(key, int):
            if key <= 0:
                raise ValueError("Variable indexing must starts at 1.")
            
            key -= 1

        return LinearEquation({ key: 1 })



    def __le__(self, other: int) -> VariableConstraint:
        pass



    def __ge__(self, other: int) -> VariableConstraint:
        pass



class x(object, metaclass=MetaVariable):
    def __init__(self):
        raise Exception("Incorrect usage of `x`. Please refer to the Pytimize formulation docs.")
