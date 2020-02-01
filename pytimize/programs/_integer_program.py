from . import LinearProgram
from typing import List

class IntegerProgram(LinearProgram):
    def __init__(self, A, b, c, z, objective: str="max", inequalities: List[str]=None, free_variables: List[int]=None, integral_variables: List[int]=None):
        super().__init__(A, b, c, z, objective, inequalities, free_variables)
        
        self._integral_variables = integral_variables

    def __str__(self): #TODO add integral constraint
        return super().__str__()

    def branch_and_bound(self): #TODO branch and bound method
        pass

    def cutting_plane(self):
        pass

    def linear_program_relaxation(self):
        pass

    #TODO LP relaxation method (returns LP without integer constraint)
    #TODO override evaluate of base to take consideration of integers
