from . import LinearProgram
from typing import List

class IntegerProgram(LinearProgram):
    def __init__(self, A, b, c, z, objective: str="max", inequalities: List[str]=None, free_variables: List[int]=None, integral_variables: List[int]=None):
        super().__init__(A, b, c, z, objective, inequalities, free_variables)
        
        self._integral_variables = integral_variables

    def __str__(self): #TODO add integral constraint
        return super().__str__()

    #TODO implement constructor
    #TODO branch and bound method
    #TODO LP relaxation method (returns LP without integer constraint)
    #TODO cutting plane method
    #TODO override evaluate of base to take consideration of integers
    #TODO add integer constraint to __str__