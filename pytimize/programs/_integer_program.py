from . import LinearProgram
from typing import List

class IntegerProgram(LinearProgram):
    def __init__(self, A, b, c, z, objective: str="max", inequalities: List[str]=None, free_variables: List[int]=None, integral_variables: List[int]=None):
        super().__init__(A, b, c, z, objective, inequalities, free_variables)
        
        self._integral_variables = integral_variables

    def __str__(self): #TODO add integral constraint
        return super().__str__()



    def branch_and_bound(self): #TODO branch and bound method
        """
        Applies the branch and bound solution method.

        Returns
        -------
        result : ndarray of int #TODO change later to accomodate infeasible result

        """

        """
        Remove comment once method is completed
        Methodology:
        Start by calling initial branch fn while tracking the most optimal sln outside the fn (min/max value where sln is integer)
        branch fn:
            use base class' simplex method to solve LP relaxation of current IP
            if solution is integer: done! return the solution and optimal value
            else if solution is infeasible: return "infeasible" (or something else to indicate infeasibility)
            else, branch on first non-int entry in solution: recurse on branch fn and call twice with an added constraint:
                once with >= ceiling of non-int, and other with <= floor of non-int
            once both have fully evaluated, take best solution of either and return (or return infeasible if both are infeasible)
        return the result of calling the branch fn (whether it's a solution or an infeasible result)
        """
        def branch(lp):
            """
            The recursive portion of the branch and bound solution method.

            Parameters
            -------
            lp : LinearProgram

            Returns
            -------
            result : ndarray of int #TODO change later to accomodate infeasible result

            """
            solution = lp.simplex_solution()  #TODO need basis of lp
            pass
        
        relaxation = self.linear_program_relaxation()
        return branch(relaxation)


    def cutting_plane(self):
        pass

    def linear_program_relaxation(self):
        """
        Creates the LP relaxation of the current IP.

        Returns
        -------
        result : LinearProgram

        """
        return super.__init__(self._A, self._b, self._c, self._z, self._objective, self.inequalities, self.free_variables)

    #TODO override evaluate of base to take consideration of integers
