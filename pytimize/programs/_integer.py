import copy
import math
import numpy as np

from ._linear import LinearProgram
from ..utilities import Comparator
from typing import List, Tuple, Optional, Union

class IntegerProgram(LinearProgram):
    def __init__(self, A, b, c, z: float=0, objective: str="max", inequalities: Optional[List[str]]=None, free_variables: Optional[List[int]]=None, negative_variables: Optional[List[int]]=None, integral_variables: Optional[List[int]]=None):
        """
        Integral variable = None => all integers
        """
        super().__init__(A, b, c, z, objective, inequalities, free_variables, negative_variables)
        
        self._integral_variables = integral_variables

    def __repr__(self) -> str: #TODO add integral constraint
        output = super().__repr__().rstrip("\n")

        if self._integral_variables is None:
            output += "x ∈ Z"

        return output



    def branch_and_bound(self) -> Optional[np.ndarray]:
        """
        Applies the branch and bound solution method.

        Returns
        -------
        result : Optional[ndarray of int]
            The optimal solution of the program. None if IP is infeasible or unbounded.

        """

        relaxation = self.create_relaxation()

        result = self.__branch(relaxation)

        if result is None:
            return None
        return result[0]


    
    def __branch(self, lp: LinearProgram) -> Optional[np.ndarray]:
        """
        The recursive portion of the branch and bound solution method.

        Parameters
        -------
        lp : LinearProgram
            The relaxation of the IP with the necessary bounds as additional constraints.

        Returns
        -------
        result : Optional[List (ndarray of int, int)]
            The optimal solution of the program, and its corresponding optimal value.

        """

        """
        TODO Currently, returns None if IP is infeasible OR unbounded - handle unbounded case separately in the future
        """
        solution = lp.solve()

        if solution is None:
            return None

        opt_value = lp.evaluate(solution)

        # check if solution is entirely integer
        # if any aren't integer, branch on that entry in the x vector and return the best result
        position = 0
        for i in range(len(solution)):
            # if value is not integer, make sure it is has the integral constraint before branching - not working atm
            if self._integral_variables is not None and position not in self._integral_variables:
                position += 1
                continue

            value = solution[i]
            if not Comparator.is_integer(value):
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

                lower_result = self.__branch(lp_lower)

                # higher branch:
                new_b[len(lp.b)] = np.ceil(value)
                new_inequalities[len(new_inequalities) - 1] = ">="
                lp_higher = LinearProgram(new_A, new_b, lp.c, lp.z, lp.objective, new_inequalities)

                higher_result = self.__branch(lp_higher)

                # both branches were infeasible/unbounded
                if lower_result is None and higher_result is None:
                    return None

                # one branch was infeasible/unbounded while the other was feasible
                elif lower_result is None:
                    return higher_result
                elif higher_result is None:
                    return lower_result
                
                # both branches were feasible, return the solution with the best optimal value
                elif lp.objective == "max":
                    if higher_result[1] > lower_result[1]:
                        return higher_result
                    return lower_result
                elif higher_result[1] > lower_result[1]:
                    return higher_result
                return lower_result

            position += 1
        
        return [solution, opt_value]  # solution passes constraints
    


    def cutting_plane(self) -> Optional[np.ndarray]:
        """
        Applies the cutting plane solution method.

        Returns
        -------
        result : Optional[ndarray of int]
            The optimal solution of the program. None if IP is infeasible or unbounded.

        """
        pass


    # TODO make public function so user can call find cutting plane which returns the constraint
    def __find_cutting_plane(self, lp: LinearProgram, sln: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Finds a cutting plane (additional constraint) for the cutting plane solution method.

        Parameters
        -------
        lp : LinearProgram
            The relaxation of the IP with the already applied cutting planes as additional constraints.

        sln : ndarray
            The current optimal solution to lp. Must be non-integral for this method to be called.

        Returns
        -------
        result : Tuple[np.ndarray, str, float]
            The new cutting plane, to be added as a constraint on lp. Format of the tuple:
            The coefficient array, the inequality, and the bound.
            i.e. x_1 - 3x_2 <= 6 is the new cutting plane, and becomes ([1, -3], "<=", 6)

        """
        pass



    def create_relaxation(self) -> LinearProgram:
        """
        Creates the corresponding linear program relaxation.

        Returns
        -------
        result : LinearProgram

        """
        return LinearProgram(self._A, self._b, self._c, self._z, self._objective, self.inequalities, self.free_variables, self.negative_variables)



    def is_feasible(self, x, show_steps: bool=True) -> bool:
        result = super().is_feasible(x, show_steps)
        return result



    #TODO override evaluate of base to take consideration of integers
