from . import LinearProgram
from typing import List
import numpy as np
import copy

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
        #if not self._is_sef:
        #    raise ArithmeticError()

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

        TODO make sure branch and bound solves integer for only those variables mentioned in integral_variable
        any variables x_i with i not in that list means it can be anything (rational number)
        If integral_variables is None then by default all x_i is integer
        """
        def branch(lp):
            """
            The recursive portion of the branch and bound solution method.

            Parameters
            -------
            lp : LinearProgram

            Returns
            -------
            result : tuple (ndarray of int OR bool, int)

            """
            if not lp.is_sef:
                lp = lp.to_sef(show_steps=False)
            solution, basis, certificate = lp.two_phase_simplex()
            
            # solution is False if program is infeasible
            if isinstance(solution, bool):
                return False

            opt_value = lp.evaluate(solution)

            # check if solution is entirely integer
            # if any aren't integer, branch on that entry in the x vector and return the best result
            position = 0
            print(solution)
            for value in solution:
                if not value.is_integer():
                    copy_A = lp.A.copy()
                    copy_b = lp.b.copy()
                    new_inequalities = copy.deepcopy(lp.inequalities)

                    # lower branch:
                    # create the new row in A as the bounding constraint
                    new_row = np.zeros(lp.A.shape[1])
                    new_row[position] = 1
                    new_A = np.concatenate((copy_A, [new_row]))
                    new_b = np.append(copy_b, np.floor(value))
                    new_inequalities.append("<=")
                    lp_lower = LinearProgram(new_A, new_b, lp.c, lp.z, lp.objective, new_inequalities)

                    lower_sln, lower_opt_value = branch(lp_lower)

                    # higher branch:
                    new_b[len(lp.b)] = np.ceil(value)
                    new_inequalities[len(new_inequalities) - 1] = ">="
                    lp_higher = LinearProgram(new_A, new_b, lp.c, lp.z, lp.objective, new_inequalities)

                    higher_sln, higher_opt_value = branch(lp_higher)

                    if isinstance(lower_sln, bool) and isinstance(higher_sln, bool):
                        return False
                    elif isinstance(lower_sln, bool):
                        return higher_sln, higher_opt_value
                    elif isinstance(higher_sln, bool):
                        return lower_sln, lower_opt_value
                    # both branches were feasible, return the solution with the best optimal value
                    elif lp.objective == "max":
                        if higher_opt_value > lower_opt_value:
                            return higher_sln, higher_opt_value
                        return lower_sln, lower_opt_value
                    elif higher_opt_value > lower_opt_value:
                        return higher_sln, higher_opt_value
                    return lower_sln, lower_opt_value

                position += 1

            return solution, opt_value  # solution is entirely integer
        
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
        return LinearProgram(self._A, self._b, self._c, self._z, self._objective, self.inequalities, self.free_variables)

    #TODO override evaluate of base to take consideration of integers
