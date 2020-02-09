from . import LinearProgram
from typing import List
import numpy as np

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
        result : ndarray of int OR bool

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
            result : tuple (ndarray of int OR bool, int)  #TODO change later to accomodate infeasible result

            """
            solution, basis, certificate = lp.two_phase_simplex()
            
            # solution is False if program is infeasible
            if not solution:
                return False

            # check if solution is entirely integer
            # if any aren't integer, branch on that entry in the x vector and return the best result
            position = 0
            for value in solution:
                if not value.is_integer():
                    copy_A = self._A.copy()
                    copy_b = self._b.copy()
                    copy_c = self._c.copy()
                    new_inequalities = self.inequalities.copy()

                    # lower branch:
                    # create the new row in A as the bounding constraint
                    new_row = np.zeros(len(self._A[0]))
                    new_row[position] = 1
                    new_A = copy_A.concatenate(new_row)
                    new_b = copy_b.concatenate(np.floor(value))
                    new_c = copy_c.concatenate(0)  # new entry should not affect objective value
                    new_inequalities.append("<=")
                    lp_lower = LinearProgram(new_A, new_b, new_c, self._z, self._objective, new_inequalities)

                    lower_sln, lower_opt_value = branch(lp_lower)

                    # higher branch:
                    new_b[len(self._b)] = np.ceil(value)
                    new_inequalities[len(new_inequalities) - 1] = ">="
                    lp_higher = LinearProgram(new_A, new_b, new_c, self._z, self._objective, new_inequalities)

                    higher_sln, higher_opt_value = branch(lp_higher)

                    if not lower_sln and not higher_sln:
                        return False
                    elif not lower_sln:
                        return higher_sln
                    elif not higher_sln:
                        return lower_sln
                    # both branches were feasible, return the solution with the best optimal value
                    elif self._objective == "max":
                        if higher_opt_value > lower_opt_value:
                            return higher_sln
                        return lower_sln
                    elif higher_opt_value > lower_opt_value:
                        return higher_sln
                    return lower_sln

                position += 1

            return solution  # solution is entirely integer
        
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
