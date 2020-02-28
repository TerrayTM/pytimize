import functools
import math
import copy
import numpy as np

from ..parsers._description_parser import render_descriptor
from ..utilities._typecheck import typecheck
from matplotlib import pyplot as plt
from collections import deque
from typing import List, Tuple, Optional, Union

# TODO: for make independent rows, check for sef at end
# TODO: add <= to variables
class LinearProgram:
    def __init__(self, A: Union[np.ndarray, List[List[float]]], b, c, z: float=0, objective: str="max", inequalities: List[str]=None, free_variables: List[int]=None):
        """
        Constructs a linear program of the form [objective]{cx + z : Ax [inequalities] b, variables >= 0},
        where objective denotes whether this is a maximization or minimization problem, inequalities is a list of 
        operators between Ax and b, and variables are entries of x that are not free.

        Parameters
        ----------
        A : 2D array-like of int, float
            The coefficient matrix of the constraints. 

        b : array-like of int, float
            The values of the constraints.

        c : array-like of int, float
            The coefficient vector of the objective function.

        z : float (default=0)
            The constant of the objective function.

        objective : str, optional (default="max")
            The objective of the program. Must be either `max` or `min`.

        inequalities : array-like of str ["=", ">=", "<="], optional (default=None)
            The operator type between Ax and b. Each index of the array-like corresponds to the same index of the 
            constraint row. If input is none, the constraint becomes Ax = b.

        free_variables : array-like of int, optional (default=None)
            The variables that are not bounded to be nonnegative. Use math indexing to represent the free ones.

        """
        A = self.__to_ndarray(A)
        b = self.__to_ndarray(b)
        c = self.__to_ndarray(c)

        if not A.ndim == 2:
            raise ValueError()
        
        if not self.__is_vector_of_size(b, A.shape[0]) or not self.__is_vector_of_size(c, A.shape[1]):
            raise ValueError()

        inequality_indices = {}
        sef_condition = True

        if inequalities is not None:
            inequalities = self.__array_like_to_list(inequalities)

            if not len(inequalities) == b.shape[0]:
                raise ValueError()

            for i in range(len(inequalities)):
                inequality = inequalities[i]

                if inequality == ">=" or inequality == "<=":
                    inequality_indices[i] = inequality

                    sef_condition = False
                elif not inequality == "=":
                    raise ValueError()
            
        if type(z) not in [int, float]:
            raise ValueError()

        if not objective == "max" and not objective == "min":
            raise ValueError()

        if free_variables is not None:
            free_variables = self.__array_like_to_list(free_variables)

            if len(free_variables) > c.shape[0]:
                raise ValueError()

        free_variables = self.__convert_indices(free_variables or [], 0, c.shape[0]) # Must update to use new function
        free_variables.sort()

        self._A = A
        self._b = b
        self._c = c
        self._z = z
        self._steps = []
        self._objective = objective
        self._inequality_indices = inequality_indices
        self._is_sef = sef_condition and len(free_variables) == 0 and objective == "max"
        self._free_variables = free_variables


    # TODO If in SEF output x >= 0 else output correct inequalities
    # EXAMPLE
    #====================================
    # Max [0. 0. 4. -11. -1.]x + 17
    # Subject To:

    # [1. 0. 2.  7.  -1.]     =   [2.]
    # [0. 1. -4. -5. 3. ]x    =   [1.]
    # x ≥ 0

    # BUG formatted incorrectly (b does not line up)
    # Max [0. 0. 0. 0. 0. 0. -1. -1. -1.]x
    # Subject To:

    # [2.  -1. -2. 1.  0.  0.  1. 0. 0.]     =   [4.]
    # [-2. 3.  1.  0. -1. 0. 0. 1. 0.]x    =   [5.]
    # [1.  -1. -1. 0. 0. -1. 0. 0. 1.]     =   [1.]
    # x ≥ 0
    def __str__(self):
        """
        Generates a nicely formatted string representation of the linear program.

        Returns
        -------
        result : str
            A string representation of the model.

        """
        output = ""
        shape = self._A.shape

        # set objective to be human readable
        if self._objective == "min":
            obj = "Min ["
        else:
            obj = "Max ["

        output += obj

        # add c vector to output
        for i in range(len(self._c)):
            if self._c[i].is_integer():
                output += str(int(self._c[i])) + "."
            else:
                output += str(self._c[i])

            if not i == len(self._c) - 1:
                # add a space between numbers only
                output += " "
        
        output += "]x"

        if abs(self._z) > 10e-10:
            sign = "+"
            number = self._z

            if self.z < 0:
                sign = "-"
                number = str(self._z)[1:]
            
            output += f" {sign} {number}"

        output += "\nSubject To:\n\n"

        # list of spaces required for each column
        col_spaces = []

        # find max length of each column for formatting
        for col in range(shape[1]):
            length = 0
            for row in range(shape[0]):
                entry_len = len(str(self._A[row, col]))
                if self._A[row, col].is_integer():
                    entry_len -= 1  # account for extra x.0 at end instead of x.
                length = max(entry_len, length)

            col_spaces.append(length)

        # list of spaces required for b vector
        b_spaces = 0

        for i in range(shape[0]):
            b_entry_len = len(str(self._b[i]))
            if self._b[i].is_integer():
                b_entry_len -= 1
            b_spaces = max(b_entry_len, b_spaces)

        # add each number to output string
        for row in range(shape[0]):
            output += "["
            for col in range(shape[1]):
                if self._A[row, col].is_integer():
                    spaces = col_spaces[col] - len(str(self._A[row, col])) + 1
                    output += str(int(self._A[row, col]))
                    output += "."
                else:
                    spaces = col_spaces[col] - len(str(self._A[row, col]))
                    output += str(self._A[row, col])
                output += " " * spaces

                if not col == shape[1] - 1:
                    # add a space between numbers only
                    output += " "
            
            output += "]"

            if row == shape[0] // 2:
                output += "x    "
            else:
                output += "     "

            if row in self._inequality_indices:
                if self._inequality_indices[row] == ">=":
                    output += "≥   "
                else:
                    output += "≤   "
            else:
                output += "=   "

            # add row-th entry from b
            output += "["
            
            if self._b[row].is_integer():
                output += f"{str(int(self._b[row]))}."
                output += " " * (b_spaces - len(str(self._b[row])) + 1)
            else:
                output += str(self._b[row])
                output += " " * (b_spaces - len(str(self._b[row])))

            output += "]\n"

        output += "x ≥ 0\n" # TODO add variable constraints

        return output


    # TODO
    def append_constraint(self, row, value, sign):
        pass


    # TODO
    def append_variable(self):
        pass



    def is_canonical_form_for(self, basis: List[int]):
        """
        Checks if the linear program is in canonical form for the specified basis.
        
        Parameters
        ----------
        basis : array-like of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        Returns
        -------
        result : bool
            Whether or not the model is in canonical form for the basis.

        """
        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])

        return all([
            self.is_basis(basis),
            np.allclose(self._A[basis], np.eye(len(basis))),
            np.allclose(self._c[basis], 0)
        ])


    
    def is_basic_solution(self, x, basis: List[int], show_steps: bool=True):
        """
        Checks if the given vector is a basic solution for the specified basis.

        Parameters
        ----------
        basis : array of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        show_steps : bool, optional (default=True)
            Whether steps should be stored or not for this operation.

        Returns
        -------
        result : bool
            Whether or not the vector is a basic solution for the basis.

        """
        if not self._is_sef:
            raise ArithmeticError() # raise error if not in SEF form ?

        x = self.__to_ndarray(x)

        if not self.__is_vector_of_size(x, self._c.shape[0]):
            raise ValueError()

        if not self.is_basis(basis):
            raise ValueError()

        show_steps and self.__append_to_steps(('4.01', basis))

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])
        result = True

        for i in range(self._c.shape[0]):
            if i not in basis and not self.__is_close_to_zero(x[i]):
                show_steps and self.__append_to_steps(("4.02", i + 1))

                result = False

        if not np.allclose(self._A @ x, self._b):
            show_steps and self.__append_to_steps("4.03")

            result = False

        show_steps and not result and self.__append_to_steps(("4.05", x, basis))
        show_steps and result and self.__append_to_steps([
            "4.06",
            "4.07",
            ("4.04", x, basis)
        ])

        return result



    def is_feasible_basic_solution(self, x, basis: List[int]):
        """
        Checks if the given vector is a feasible basic solution for the specified basis.

        Parameters
        ----------
        basis : array of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        Returns
        -------
        result : bool
            Whether or not the vector is a feasible basic solution for the basis.

        """
        x = self.__to_ndarray(x)

        if not self.__is_vector_of_size(x, self._c):
            raise ValueError()

        return (x >= 0).all() and self.is_basic_solution(x, basis) 



    def is_basis(self, basis: List[int]):
        """
        Checks if the given base indices form a valid basis for the current model.

        Parameters
        ----------
        basis : array of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        Returns
        -------
        result : bool
            Whether or not the base indices form a valid basis.

        """
        basis = self.__to_array_indexing(basis)
 
        basis.sort()

        if not self._A.shape[0] == len(basis):
            return False

        if max(basis) >= self._A.shape[1] or min(basis) < 0:
            return False
        
        return not self.__is_close_to_zero(np.linalg.det(self._A[:, basis]))


    @typecheck
    def is_basis_feasible(self, basis: List[int]):
        """
        Tests if the given basis is feasible.

        Parameters
        ----------
        basis : array of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        Returns
        -------
        result : bool
            Whether or not the basis is feasible.

        """
        return self.is_basis(basis) and (self.compute_basic_solution(basis) >= 0).all()



    def compute_basic_solution(self, basis: List[int]) -> np.ndarray:
        """
        Computes the basic solution corresponding to the specified basis.

        Parameters
        ----------
        basis : array of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        Returns
        -------
        result : ndarray of float
            The basic solution corresponding to the basis.

        """
        if not self.is_basis(basis):
            raise ArithmeticError()

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis)

        return self.__compute_basic_solution(basis)



    @typecheck
    def to_canonical_form(self, basis: List[int], show_steps: bool=True, in_place: bool=False) -> "LinearProgram":
        """
        Converts the linear program into canonical form for the given basis.

        Parameters
        ----------
        basis : array of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        show_steps : bool, optional (default=True)
            Whether steps should be stored or not for this operation.

        in_place : bool, optional (default=True)
            Whether the operation should return a copy or be performed in place.

        Returns
        -------
        result : LinearProgram
            The copy of the linear program or self in canonical form.

        """
        if not in_place:
            return self.copy().to_canonical_form(basis, show_steps, True)

        if not self._is_sef:
            raise ArithmeticError()

        show_steps and self.__append_to_steps(('1.01', basis))

        if not self.is_basis(basis):
            raise ValueError()

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])

        return self.__to_canonical_form(basis, show_steps)



    def __to_canonical_form(self, basis: List[int], show_steps: bool):
        """
        Helper function for converting the linear program into canonical form for the given basis.

        Parameters
        ----------
        basis : array-like of int
            The column indices of the coefficient matrix that forms a basis. Use zero indexing for format.

        show_steps : bool, optional (default=True)
            Whether steps should be stored or not for this operation.

        in_place : bool, optional (default=True)
            Whether the operation should return a copy or be performed in place.

        Returns
        -------
        result : LinearProgram
            The linear program in canonical form.

        """
        Ab = self._A[:, basis]
        cb = self._c[basis]
        Ab_inverse = np.linalg.inv(Ab)
        y_transpose = (Ab_inverse.T @ cb).T

        show_steps and self.__append_to_steps(('1.02', Ab))       
        show_steps and self.__append_to_steps(('1.03', cb))
        show_steps and self.__append_to_steps(('1.04', Ab_inverse))
        show_steps and self.__append_to_steps(('1.05', y_transpose))

        A = Ab_inverse @ self._A
        b = Ab_inverse @ self._b
        c = self._c - y_transpose @ self._A
        z = y_transpose @ self._b + self._z
        self._A = A
        self._b = b
        self._c = c
        self._z = z

        return self



    def create_dual(self, show_steps: bool=True) -> "LinearProgram":
        """
        Creates the corresponding dual program.

        Parameters
        ----------
        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : LinearProgram
            The corresponding dual program.

        """
        pass
    
        #          max          |   min
        #--------------------|-------------
        #              <=   >= 0
        #   constraint  =   free variable
        #              >=   <= 0
        #
        #    variable  >=0  >=
        #              free =  
        #              <=0  <=  constraint

        #LinearProgram(self._A.T, self._c, self._b, 0)

    
    def create_auxiliary(self, show_steps: bool=True) -> "LinearProgram":
        """
        Creates the corresponding auxiliary program. This operation requires 
        the program to be in SEF.

        Parameters
        ----------
        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : "LinearProgram"
            The corresponding auxiliary program.

        """
        if not self._is_sef:
            raise ArithmeticError("Linear program must be in SEF.")

        negative_indices = np.where(self._b < 0)
        rows, columns = self._A.shape
        A_aux = self._A.copy() 
        b_aux = self._b.copy()
        A_aux[negative_indices] *= -1
        b_aux[negative_indices] *= -1
        A_aux = np.c_[A_aux, np.eye(rows)]
        c_aux = np.zeros(columns + rows)
        c_aux[columns:] = 1

        return LinearProgram(A_aux, b_aux, c_aux, 0, "min")



    def solve(self, show_steps: bool=True) -> Optional[np.ndarray]:
        """
        Solves the linear program and returns an optimal solution if one exists.

        Parameters
        ----------
        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : Optional[np.ndarray]
            The optimal solution of the program.

        """
        if self._is_sef:
            return self.two_phase_simplex()[0]
        
        sef = self.to_sef(show_steps)
        solution = sef.two_phase_simplex(show_steps)[0]

        if solution is not None:
            solution = np.delete(solution, np.s_[self._c.shape[0]:self._c.shape[0] + sef._reverse_sef["drop"]], 0)

            for i, index in enumerate(sef._reverse_sef["concat"]):
                index -= i - 1
                solution[index] -= solution[index]
                solution = np.delete(solution, index)
            # TODO
            # If conversion of basis and certificate to original is possible add them to return
            # Distinguish between unbounded and infeasible 

        return solution



    def two_phase_simplex(self, show_steps: bool=True) -> Tuple[Optional[np.ndarray], Optional[List[int]], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Creates an auxiliary program and solves it to determine a starting basis for the given linear program.
        With the starting basis, computes simplex and returns a solution if it has one, the optimal basis 
        if it exists, and a certificate. The certificate can certify either unboundedness, optimality, or infeasibility
        depending on the outcome of the program. This operation requires the program to be in SEF.

        Parameters
        ----------
        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : Tuple[Optional[np.ndarray], Optional[List[int]], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
            The solution vector, the basis, and the certificate. If the optimal basis is none,
            then the linear program is either infeasible or unbounded. If unbounded, the certificate will be a tuple
            of a feasible solution followed by a certifying vector.
    
        """
        if not self._is_sef:
            raise ArithmeticError("Linear program must be in SEF.")

        p_aux = self.create_auxiliary(show_steps).to_sef(show_steps, True)
        rows, columns = self._A.shape
        basis = [i for i in range(columns + 1, rows + columns + 1)]
        negative_indices = np.where(self._b < 0)

        solution, basis, certificate = p_aux.simplex(basis)

        if self.__is_close_to_zero(p_aux.evaluate(solution)):
            p_basis = basis

            if not self.is_basis(p_basis):
                p_basis = []
                zero_set = []

                for i, value in enumerate(solution[:columns], 1):
                    if not self.__is_close_to_zero(value):
                        p_basis.append(i)
                    else:
                        zero_set.append(i)
                
                subset_size = self._A.shape[0] - len(p_basis)
                queue = deque()

                for i in range(0, len(zero_set) - subset_size + 1):
                    queue.append((i, [zero_set[i]]))

                while len(queue) > 0:
                    current = queue.pop()

                    if len(current[1]) == subset_size:
                        basis_candidate = p_basis + current[1]

                        basis_candidate.sort()

                        if self.is_basis(basis_candidate):
                            p_basis = basis_candidate

                            break
                    else:
                        for i in range(current[0] + 1, len(zero_set)):
                            copy = current[1] + [zero_set[i]]

                            queue.append((i, copy))

            return self.simplex(p_basis, show_steps)

        basis = self.__to_array_indexing(basis)
        certificate = np.linalg.inv(p_aux.A[:, basis].T) @ p_aux.c[basis]
        certificate[negative_indices] *= -1

        return None, None, certificate



    def simplex(self, basis: List[int], show_steps: bool=True) -> Tuple[Optional[np.ndarray], Optional[List[int]], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]: 
        """
        Computes simplex iterations until termination. Returns the optimal solution if it has one, 
        the optimal basis if it exists, and the certificate of unboundedness or optimality.
        This operation requires the program to be in SEF.

        Parameters
        ----------
        basis : List[int]
            The column indices of the coefficient matrix that forms an starting basis.
            Use math indexing for format.

        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : Tuple[Optional[np.ndarray], Optional[List[int]], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
            The solution vector, the basis, and the certificate. If the optimal basis is none,
            then the linear program is unbounded. In this case the certificate will be a tuple
            of a feasible solution followed by a certifying vector.
        
        """
        if not self._is_sef:
            raise ArithmeticError("Linear program must be in SEF.")
        
        solution = None
        certificate = None
        updated_lp = None
        counter = 0
        negative_indices = np.where(self._b < 0)

        self._b[negative_indices] *= -1
        self._A[negative_indices] *= -1
    
        show_steps and self.__append_to_steps(("5.02", counter))
        show_steps and self.__append_to_steps(("5.01", self))

        while solution is None and basis is not None:
            solution, basis, updated_lp = self.simplex_iteration(basis, show_steps)
            counter += 1

            show_steps and self.__append_to_steps(("5.02", counter))
            show_steps and self.__append_to_steps(("5.01", updated_lp))
        
        if basis is None:
            certificate = (updated_lp._feasible_solution, np.zeros(self._c.shape[0]))
            k = np.argmax(updated_lp.c > 0)
            certificate[1][updated_lp._feasible_basis] = -updated_lp.A[:, k]
            certificate[1][k] = 1

            show_steps and self.__append_to_steps("5.05")
        else:
            converted_basis = self.__to_array_indexing(basis)
            certificate = np.linalg.inv(self._A[:, converted_basis].T) @ self._c[converted_basis]
            certificate[negative_indices] *= -1

            show_steps and self.__append_to_steps(("5.03", solution))
            show_steps and self.__append_to_steps(("5.04", basis))
            show_steps and self.__append_to_steps(("5.06", certificate))

        self._b[negative_indices] *= -1
        self._A[negative_indices] *= -1
        
        return solution, basis, certificate



    def simplex_iteration(self, basis: List[int], show_steps: bool=True, in_place: bool=False) -> Tuple[Optional[np.ndarray], Optional[List[int]], "LinearProgram"]:
        """
        Computes a single iteration of the simplex algorithm with Bland's rule. Returns the optimal solution
        if it has been found, the next or optimal basis if it exists, and the updated linear program. 
        This operation requires the program to be in SEF.

        Parameters
        ----------
        basis : List[int]
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        in_place : bool (default=False)
            Whether or not the operation should be performed in place. 

        Returns
        -------
        result : Tuple[Optional[np.ndarray], Optional[List[int]], "LinearProgram"]
            The solution vector, the basis, and the updated program. If the basis is none,
            then the linear program is unbounded.

        """
        if not self._is_sef:
            raise ArithmeticError("Linear program must be in SEF.")
        
        if not in_place:
            return self.copy().simplex_iteration(basis, show_steps, True)

        if not self.is_basis(basis):
            raise ValueError("The given basis is invalid.")

        basis = self.__to_array_indexing(basis)

        basis.sort()

        negative_indices = np.where(self._b < 0)

        self._b[negative_indices] *= -1
        self._A[negative_indices] *= -1

        self.__to_canonical_form(basis, show_steps)

        x = self.__compute_basic_solution(basis)
        N = [i for i in range(self._A.shape[1]) if i not in basis]
        k = None

        if (self._c[N] <= 0).all():
            if (x < 0).any():
                raise ArithmeticError("The given linear program is infeasible.")

            return x, self.__to_math_indexing(basis), self

        for i in N:
            if self._c[i] > 0:
                k = i

                break

        Ak = self._A[:, k]
        leave = None
        t = math.inf

        if (Ak <= 0).all():
            self._feasible_solution = x
            self._feasible_basis = basis

            return None, None, self

        for i in range(len(Ak)):
            if Ak[i] > 0:
                ratio = self._b[i] / Ak[i]

                if ratio < t:
                    t = ratio
                    leave = i 
        
        basis.remove(basis[leave])
        basis.append(k)
        basis.sort()

        return None, self.__to_math_indexing(basis), self



    def verify_infeasibility(self, certificate: Union[np.ndarray, List[float]], show_steps: bool=True) -> bool:
        """
        Verifies the certificate of infeasibility. This operation requires 
        the program to be in SEF.

        Parameters
        ----------
        certificate : Union[np.ndarray, List[float]]
            The certificate to validated.

        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : bool
            Whether or not the certificate is valid.

        """
        if not self._is_sef:
            raise ArithmeticError("Linear program must be in SEF.")

        certificate = self.__to_ndarray(certificate)
        yA = certificate @ self._A
        yb = certificate @ self._b

        return self.__is_close_compare(yA, 0, ">") and not self.__is_close_compare(yb, 0, ">")



    def verify_unboundedness(self, x: Union[np.ndarray, List[float]], certificate: Union[np.ndarray, List[float]], show_steps: bool=True) -> bool:
        """
        Verifies the certificate of unboundedness. This operation requires 
        the program to be in SEF.

        Parameters
        ----------
        x : Union[np.ndarray, List[float]]
            A feasible solution for the linear program.

        certificate : Union[np.ndarray, List[float]]
            The certificate to validated.

        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : bool
            Whether or not the certificate is valid.

        """
        if not self._is_sef:
            raise ArithmeticError("Linear program must be in SEF.")

        certificate = self.__to_ndarray(certificate)
        x = self.__to_ndarray(x)

        if not self.is_feasible(x):
            return False
        
        if self.__is_close_compare(self._c @ certificate, 0, "<"):
            return False
        
        if not np.allclose(self._A @ certificate, 0):
            return False
        
        return self.__is_close_compare(certificate, 0, ">")



    def verify_optimality(self, certificate: Union[np.ndarray, List[float]], show_steps: bool=True) -> bool:
        """
        Verifies the certificate of optimality. This operation requires 
        the program to be in SEF.

        Parameters
        ----------
        certificate : Union[np.ndarray, List[float]]
            The certificate to validated.

        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        Returns
        -------
        result : bool
            Whether or not the certificate is valid.

        """
        if not self._is_sef:
            raise ArithmeticError("Linear program must be in SEF.")

        certificate = self.__to_ndarray(certificate)
        test = self._c - certificate @ self._A
        
        return self.__is_close_compare(test, 0, "<")



    def is_feasible(self, x, show_steps: bool=True):
        """
        Checks if the given vector is a feasible solution.

        Parameters
        ----------
        x : array-like of int, float

        Returns
        -------
        result : bool

        """
        x = self.__to_ndarray(x)

        if not self.__is_vector_of_size(x, self._c.shape[0]):
            raise ValueError()

        show_steps and self.__append_to_steps(("2.01", x))

        if self._is_sef:
            all_nonnegative = (x >= 0).all()
            satisfy_constraints = np.allclose(self._A @ x, self._b)
            is_feasible = all_nonnegative and satisfy_constraints

            show_steps and is_feasible and self.__append_to_steps([
                ("2.02", x),
                "2.03",
                ("2.04", x),
                "2.05"
            ])
            show_steps and not is_feasible and self.__append_to_steps([
                ("2.06", x),
                "2.03",
                ("2.07", x) if not all_nonnegative else None,
                "2.08" if not satisfy_constraints else None
            ])

            return is_feasible
        
        is_feasible = True

        for i in range(self._A.shape[0]):
            row = self._A[i, :]
            value = row @ x
            step = None

            if i not in self._free_variables and x[i] < 0:
                show_steps and self.__append_to_steps([
                    ("2.06", x) if is_feasible else None,
                    ("2.09", i + 1),
                    ("2.10", i + 1)
                ])

                is_feasible = False

                if not show_steps:
                    return False

            if i in self._inequality_indices:
                current = self._inequality_indices[i]

                if current == "<=" and value > self._b[i]:
                    step = "2.11"

                    if not show_steps:
                        return False
                elif current == ">=" and value < self._b[i]:
                    step = "2.12"

                    if not show_steps:
                        return False
            elif not math.isclose(value, self._b[i]):
                step = "2.13"

                if not show_steps:
                    return False

            if step:
                show_steps and self.__append_to_steps([
                    ("2.06", x) if is_feasible else None,
                    (step, row, x, value, value, self.b[i])
                ])

                is_feasible = False
        
        if is_feasible:
            show_steps and self.__append_to_steps([
                ("2.02", x),
                "2.14",
                "2.15"
            ])

        return is_feasible


    
    def graph_polyhedron(self, graph_limit: float=1000) -> None:
        """
        Graphs the feasible region of the linear program. Only supports 2 dimensional visualization.

        Constraints for the program must be in the form Ax <= b or Ax >= b.
        
        Graph is limited to the region between -1000 and 1000 in both x and y coordinates.

        """
        # run preliminary checks on data validity
        if not self._A.shape[1] == 2:
            raise ArithmeticError()

        if not len(self._inequality_indices) == self._b.shape[0]:
            raise ArithmeticError()

        if not all(i == "<=" for i in list(self._inequality_indices.values())):
            if not all(i == ">=" for i in list(self._inequality_indices.values())):
                raise ArithmeticError()

        graph_limit = abs(graph_limit)

        # add boundary inequalities at x, y = +/-graph_limit
        copy_A = self._A.copy()
        copy_b = self._b.copy()

        copy_A = np.append(copy_A, [[1, 0], [1, 0], [0, 1], [0, 1]], 0)
        copy_b = np.append(copy_b, [graph_limit, -graph_limit, graph_limit, -graph_limit])

        num_inequalities = len(self._inequality_indices)
        self._inequality_indices[num_inequalities] = "<="
        self._inequality_indices[num_inequalities + 1] = ">="
        self._inequality_indices[num_inequalities + 2] = "<="
        self._inequality_indices[num_inequalities + 3] = ">="

        A = np.array([[0, 0], [0, 0]])
        b = np.array([0, 0])

        shape = copy_A.shape
        points = []

        # get intersect points of inequalities/lines
        # points are sorted later, only if necessary
        for i in range(shape[0]):
            for j in range(shape[0]):
                if i < j:
                    A[0, :] = copy_A[i, :]
                    A[1, :] = copy_A[j, :]

                    b[0] = copy_b[i]
                    b[1] = copy_b[j]

                    # if both lines are horizontal/vertical, skip point
                    # (these lines will never intersect and thus solving will cause an error)
                    if A[0][0] == 0 and A[1][0] == 0:
                        continue
                    elif A[0][1] == 0 and A[1][1] == 0:
                        continue

                    point = np.linalg.solve(A, b)
                    points.append(point)


        # check if each point satisfies every inequality:
        # new_points is a list of all points that do so, and is copied to points afterward
        new_points = []
        for point in points:
            valid_point = True

            for i in range(len(copy_A)):
                value = point[0] * copy_A[i, 0] + point[1] * copy_A[i, 1]

                if self._inequality_indices[i] == "<=":
                    if value > copy_b[i] and not math.isclose(value, copy_b[i]):
                        valid_point = False
                elif self._inequality_indices[i] == ">=":
                    if value < copy_b[i] and not math.isclose(value, copy_b[i]):
                        valid_point = False

            if valid_point:
                new_points.append(point)

        points = copy.deepcopy(new_points)

        
        # if no points remain, then none of the points found satisfy all inequalities;
        # thus the system is inconsistent
        if len(points) == 0:
            print("Error: inconsistent system of equations")
            exit()

        
        # sort points so the polygon is drawn properly
        # method from https://stackoverflow.com/questions/10846431/ordering-shuffled-points-that-can-be-joined-to-form-a-polygon-in-python/10852917
        # compute centroid
        cent = (sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points))
        # sort by polar angle
        points.sort(key = lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))

        # add the first point again to create a closed loop
        points.append(points[0])

        # prepare the points for plotting
        xs, ys = zip(*points)

        # remove added boundary inequalities so data isn't mutated
        del self._inequality_indices[num_inequalities + 3]
        del self._inequality_indices[num_inequalities + 2]
        del self._inequality_indices[num_inequalities + 1]
        del self._inequality_indices[num_inequalities]

        # plot and display the feasible region
        plt.figure()
        plt.plot(xs, ys)
        plt.grid()
        plt.fill(xs, ys)
        plt.show()



    def evaluate(self, x: Union[np.ndarray, List[float]]) -> float:
        """
        Evaluates the objective function with a given vector.

        Parameters
        ----------
        x : Union[np.ndarray, List[float]]
            The input vector.

        Returns
        -------
        result : float
            The value of the program with respect to the vector.

        """
        x = self.__to_ndarray(x)
        
        if not self.__is_vector_of_size(x, self._c.shape[0]):
            raise ValueError()
        
        return self._c @ x + self._z



    def value_of(self, x: Union[np.ndarray, List[float]]) -> float:
        """
        Evaluates the objective function with a given vector. This operation requires 
        the vector to be feasible.

        Parameters
        ----------
        x : Union[np.ndarray, List[float]]
            The input vector.

        Returns
        -------
        result : float
            The value of the program with respect to the vector.
        
        """
        x = self.__to_ndarray(x)

        if not self.is_feasible(x):
            raise ValueError()
        
        return self.evaluate(x)



    def copy(self) -> "LinearProgram":
        """
        Creates a deep copy of the linear program.

        Returns
        -------
        result : "LinearProgram"
            The copy of the program.

        """
        p = LinearProgram(self._A.copy(), self._b.copy(), self._c.copy(), self._z, self._objective)

        p._inequality_indices = self._inequality_indices.copy()
        p._free_variables = self._free_variables.copy()
        p._steps = self._steps.copy()
        p._is_sef = self._is_sef

        return p

    

    @typecheck
    def to_sef(self, show_steps: bool=True, in_place: bool=False) -> "LinearProgram":
        """
        Converts the linear program to standard equality form.

        Parameters
        ----------
        show_steps : bool (default=True)
            Whether or not the steps should be displayed.

        in_place : bool (default=False)
            Whether or not the operation should be performed in place. 

        Returns
        -------
        result : "LinearProgram"
            The program in standard equality form.

        """
        if not in_place:
            return self.copy().to_sef(show_steps, True)

        if self._is_sef:
            return self

        show_steps and self.__append_to_steps("3.01")

        if self._objective == "min":
            self._c = -self._c
            self._objective = "max"

            show_steps and self.__append_to_steps([
                "3.02",
                ("3.03", -self._c, self._c)
            ])

        self._reverse_sef = { "drop": 0, "concat": [] }

        for i in range(len(self._free_variables)):
            index = self._free_variables[i] + i
            self._c = np.insert(self._c, index + 1, -self._c[index])
            self._A = np.insert(self._A, index + 1, -self._A[:, index], 1)

            self._reverse_sef["concat"].append(index)

        self._free_variables = []

        for i in range(self._b.shape[0]):
            if i in self._inequality_indices:
                operator = self._inequality_indices[i]
                self._A = np.c_[self._A, np.zeros(self._A.shape[0])]
                self._c = np.r_[self._c, 0]

                if operator == ">=":
                    self._A[i, -1] = -1
                elif operator == "<=":
                    self._A[i, -1] = 1
                
                self._reverse_sef["drop"] += 1

        self._inequality_indices = {}
        self._is_sef = True
        
        return self



    def is_solution_optimal(self, x):
        """
        Checks if the given vector is a optimal solution.

        Returns
        -------
        result : bool

        """
        self.is_feasible(x)



    def clear_steps(self, in_place=False):
        """
        Clears the steps that are stored.

        """
        if not in_place:
            return self.copy().clear_steps()

        self._steps = []

        return self



    def __to_ndarray(self, source):
        """
        Converts array-like to an ndarray.

        Returns
        -------
        result : ndarray of float

        """
        if isinstance(source, np.ndarray):
            if not np.issubdtype(source.dtype, np.number):
                raise ValueError()
            
            if not np.issubdtype(source.dtype, np.floating):
                source = source.astype(float)
                    
            return source

        if isinstance(source, list):
            length = None

            for row in source:
                if isinstance(row, list):
                    if length == None:
                        length = len(row)
                    elif not len(row) == length:
                        raise ValueError()

                    for column in row:
                        if isinstance(column, list):
                            raise ValueError()
                        elif type(column) not in [int, float]:
                            raise ValueError()
                elif type(row) not in [int, float]:
                    raise ValueError()

            return np.array(source, dtype=float)

        raise ValueError()



    def __compute_basic_solution(self, basis):
        """
        Helper function for computing the basic solution corresponding to the basis.

        basis : array-like of int
            The column indices of the coefficient matrix that forms a basis. First item index starts at 0.

        Returns
        -------
        result : ndarray of float
            The basic solution corresponding to the basis.

        """
        components = np.linalg.inv(self._A[:, basis]) @ self._b
        solution = np.zeros(self._c.shape[0])
        
        basis.sort() # TODO copy? To prevent reference

        for index, value in zip(basis, components):
            solution[index] = value
        
        return solution



    def __get_inequalities(self):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        inequalities = []

        for i in range(self._b.shape[0]):
            if i in self._inequality_indices:
                inequalities.append(self._inequality_indices[i])
            else:
                inequalities.append("=")

        return inequalities



    def __append_to_steps(self, entity):
        """
        Appends the given step to the procedure array.

        """
        if isinstance(entity, list):
            for step in entity:
                if step:
                    if isinstance(step, str):
                        key = step
                    elif isinstance(step, tuple):
                        key = step[0]
                    else: 
                        raise ValueError()

                    self._steps.append({
                        "key": key,
                        "text": render_descriptor(key, list(step[1:]))
                    })
        else:
            key = entity[0] if isinstance(entity, tuple) else None

            if not key:
                if isinstance(entity, str):
                    key = entity
                else:
                    raise ValueError()

            self._steps.append({
                "key": key,
                "text": render_descriptor(key, list(entity[1:]))
            })



    def __is_close_to_zero(self, value: float) -> bool:
        """
        Checks if the given value is close to zero. Use this function over `math.isclose` for 
        comparisions over 0.
        
        """
        return abs(value) < 1.0e-10


    # TODO replace <= and >= with the below function
    def __is_close_compare(self, value: Union[float, np.ndarray], test: float, comparison: str) -> bool:
        """
        Performs less than or greater than comparision with equality. If the two values are
        close enough then they are treated as equal.
        
        """
        comparator = lambda x, y: x < y

        if comparison == ">":
            comparator = lambda x, y: x > y
        #TODO replace vectorize and test with higher dimension arrays
        if test == 0:
            if isinstance(value, np.ndarray):
                return np.vectorize(lambda x: self.__is_close_to_zero(x) or comparator(x, 0))(value).all()
            
            return self.__is_close_to_zero(value) or comparator(value, 0)

        if isinstance(value, np.ndarray):
            return np.vectorize(lambda x: math.isclose(x, test) or comparator(x, test))(value).all()

        return math.isclose(value, test) or comparator(value, test)


    def __get_free_variables(self):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        return self.__to_math_indexing(self._free_variables)


    
    def __format_steps(self):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        return functools.reduce((lambda previous, current: f"{previous}\n{current['text']}"), self.steps, "").strip()



    def __to_math_indexing(self, array: List[int]) -> List[int]:
        """
        Converts array indexing to math indexing.

        """
        return [i + 1 for i in array]



    def __to_array_indexing(self, array: List[int]) -> List[int]:
        """
        Converts math indexing to array indexing.

        """
        return [i - 1 for i in array]



    def __convert_indices(self, indices, min_value=None, max_value=None):
        """ 
        OBSOLETE: USE ABOVE
        Converts from math indexing to array indexing.
        
        min_value is the closed lower bound allowed for minimum index value.
        max_value is the open upper bound allowed for minimum index value.

        """
        indices = list(map(lambda i: i - 1, indices))

        if len(indices) > 0:
            conditions = [
                not min_value == None and min(indices) < min_value,
                max_value and max(indices) >= max_value
            ]

            if any(conditions):
                raise IndexError()
                
        return indices



    def __is_vector_of_size(self, x, dimension):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        return isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == dimension



    def __array_like_to_list(self, array_like):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        if not isinstance(array_like, list) and not isinstance(array_like, np.ndarray):
                raise ValueError()

        if isinstance(array_like, np.ndarray):
            if not array_like.ndim == 1:
                raise ValueError()

            array_like = array_like.tolist()

        return array_like


    # TODO integration
    def __is_in_rref(self, arr):
        """
        Returns whether or not the given array is in RREF.

        :param arr: An m x n matrix.

        :return: A boolean value indicating if the matrix is in RREF.
        
        """
        shape = arr.shape
        row_has_leading_one = False
        has_zero_row = False

        # Check that all rows either have a leading one or are zero rows
        # For all leading ones, check that the rest of the column is empty, then ignore rest of row
        # Also check that all zero rows are at the bottom of the array
        for row in range(shape[0]):
            row_has_leading_one = False
            for col in range(shape[1]):
                if math.isclose(arr[row, col], 0):
                    # have not found a non-zero entry yet, search rest of row
                    continue

                elif has_zero_row:
                    # row has non-zero entries but there is a zero row above it
                    return False

                elif math.isclose(arr[row, col], 1):
                    # found a leading one, check rest of column for zeros
                    row_has_leading_one = True
                    for r in range(shape[0]):
                        if not r == row and not math.isclose(arr[r, col], 0):
                            return False
                    break

                else:
                    # row has non-zero entries but first number is not 1
                    return False

            if not row_has_leading_one:
                # row was empty
                has_zero_row = True
                
        return True


    # TODO need tests and validation
    def to_rref(self, arr):
        """
        Returns an array that has been row reduced into row reduced echelon 
        form (RREF) using the Gauss-Jordan algorithm. 

        :param arr: An m x n matrix.

        :return: An m x n matrix in RREF that is row equivalent to the given matrix.

        """
        arr = arr.astype(np.floating)
        shape = arr.shape
        col = 0
        for row in range(shape[0]):
            # Get a 1 in the row,col entry
            if not math.isclose(arr[row, col], 1):
                i = 0
                while math.isclose(arr[row, col], 0):
                    # If number in row,col is 0, find a lower row with a non-zero entry
                    # If all lower rows have a 0 in the column, move on to the next column
                    if math.isclose(i + row, shape[0]):
                        i = 0
                        col += 1
                        if math.isclose(col, shape[1]):
                            break
                        continue
                    if not math.isclose(arr[row + i, col], 0):
                        # Found a lower row with non-zero entry, swap rows
                        arr[[row, row + i]] = arr[[row + i, row]]
                        break
                    i += 1
                
                if math.isclose(col, shape[1]):
                    # Everything left is 0
                    break

                # Divide row by row,col entry to get a 1
                num = arr[row, col]
                arr[row, :] /= num

            # Subtract a multiple of row from all other rows to get 0s in rest of col
            for i in range(shape[0]):
                if not i == row:
                    multiple = arr[i, col] / arr[row, col]
                    arr[i, :] -= arr[row, :] * multiple
                        
            col += 1
            if math.isclose(col, shape[1]):
                break

        return arr


    # TODO need integration
    def __remove_dependent_rows(self):
        """
        Removes dependent rows from the constraints and returns the new values for A and b. The linear program must be in almost SEF.
        
        """
        arr = np.c_[self._A, self._b]

        shape = arr.shape

        # iterate through rows and check each one against every other row for multiple
        for i in range(shape[0]):
            for j in range(shape[0]):
                if j <= i:
                    # Don't check row against itself or previously checked rows
                    continue
                
                multiple = 0
                duplicate = True
                for col in range(shape[1]):
                    if math.isclose(arr[i, col], 0):
                        if math.isclose(arr[j, col], 0):
                            # both zero entries, move on to next column
                            continue
                        # one row has a zero while other doesn't, move on to next row
                        duplicate = False
                        break
                    elif math.isclose(arr[j, col], 0):
                        # one row has a zero while other doesn't
                        duplicate = False
                        break
                    
                    if col == 0:
                        multiple = arr[i, col] / arr[j, col]
                    elif not math.isclose(arr[i, col] / arr[j, col], multiple):
                        duplicate = False
                        break

                if duplicate:
                    # duplicate row found, turn it into a zero row for removal later
                    arr[j, :].fill(0)

        new_arr = np.copy(arr)
        for i in range(shape[0]):
            # iterate through array backwards and remove zero rows
            if np.count_nonzero(arr[shape[0] - 1 - i, :]) == 0:
                new_arr = np.delete(new_arr, shape[0] - 1 - i, 0)

        new_shape = new_arr.shape
        self._b = new_arr[:, new_shape[1] - 1]
        self._A = np.delete(new_arr, new_shape[1] - 1, 1)



    @property
    def A(self):
        """ 
        Gets the constraint coefficient matrix.

        Returns
        -------
        result : 2D ndarray of float

        """
        return self._A



    @property
    def b(self):
        """ 
        Gets the constraint vector. 

        Returns
        -------
        result : ndarray of float
        
        """
        return self._b



    @property
    def c(self):
        """ 
        Gets the coefficient vector of the objective function.

        Returns
        -------
        result : ndarray of float

        """
        return self._c



    @property
    def z(self):
        """ 
        Gets the constant of the objective function. 
        
        Returns
        -------
        result : int, float

        """
        return self._z



    @property
    def inequalities(self):
        """
        Gets the constraint inequalities.

        Returns
        -------
        result : list of str ["=", ">=", "<="]

        """
        return self.__get_inequalities()



    @property
    def objective(self):
        """ 
        Gets the objective of the optimization. 

        Returns
        -------
        result : bool

        """
        return self._objective



    @property
    def is_sef(self):
        """ 
        Gets whether or not the linear program is in standard equality form.
        
        Returns
        -------
        result : bool

        """
        return self._is_sef



    @property
    def steps(self):
        """ 
        Gets the steps of all operations since last reset. 
        
        Returns
        -------
        result : list of dict { "key" : int, "text" : str }

        """
        return self._steps



    @property
    def steps_string(self):
        """ 
        Gets the steps of all operations since last reset in string form. 
        
        Returns
        -------
        result : str

        """
        return self.__format_steps()
    


    @property
    def free_variables(self):
        """ 
        Gets the free variable indices in math indexing format.

        Returns
        -------
        result : list of int

        """
        return self.__get_free_variables()



    @property
    def is_in_rref(self):   #EXPERIMENTAL
        """ 
        Gets if the constraint is in RREF. Model must be in SEF. 
        
        Returns
        -------
        result : bool

        """
        return self.__is_in_rref(self._A)
