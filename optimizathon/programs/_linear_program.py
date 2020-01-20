import math
import functools
import numpy as np
import copy

from ..parsers._description_parser import render_descriptor
from matplotlib import pyplot as plt

# TODO: for make independent rows, check for sef at end
# TODO: add <= to variables
class LinearProgram:
    def __init__(self, A, b, c, z, objective="max", inequalities=None, free_variables=None):
        """
        Constructs a linear programming model of the form [objective]{cx + z : Ax [inequalities] b, variables >= 0},
        where objective denotes whether this is a maximization or minimization problem, inequalities is a list of 
        operators between Ax and b, and variables are entries of x that are not free.

        Parameters
        ----------
        A : 2D array-like of int, float
            The coefficient matrix of the model. 

        b : array-like of int, float
            The constraint values of the model.

        c : array-like of int, float
            The coefficient vector of the objective function.

        z : int, float
            The constant of the objective function.

        objective : str, optional (default="max")
            The objective of the linear programming model. Must be either "max" or "min".

        inequalities : array-like of str ["=", ">=", "<="], optional (default=None)
            The operator type between Ax and b. Each index of the array-like corresponds to the same index of the 
            constraint row. If input is "None", the constraint becomes Ax = b.

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

        if not inequalities is None:
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
            
        if not type(z) in [int, float]:
            raise ValueError()

        if not objective == "max" and not objective == "min":
            raise ValueError()

        if not free_variables is None:
            free_variables = self.__array_like_to_list(free_variables)

            if len(free_variables) > c.shape[0]:
                raise ValueError()

        free_variables = self.__convert_indices(free_variables or [], 0, c.shape[0])
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


    # TODO include free variables in string
    # TODO If in SEF output x >= 0 else output correct inequalities
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

        if not math.isclose(self._z, 0):
            output += " + " + str(self._z)

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

        return output



    def is_canonical_form_for(self, basis):
        """
        Checks if the linear program is in canonical form for the specified basis.

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


    
    def is_basic_solution(self, x, basis, show_steps=True):
        """
        Checks if the given vector is a basic solution for the specified basis.

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
            if not i in basis and not math.isclose(x[i], 0):
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



    def is_feasible_basic_solution(self, x, basis):
        """
        Checks if the given vector is a feasible basic solution for the specified basis.

        Returns
        -------
        result : bool
            Whether or not the vector is a feasible basic solution for the basis.

        """
        x = self.__to_ndarray(x)

        if not self.__is_vector_of_size(x, self._c):
            raise ValueError()

        return (x >= 0).all() and self.is_basic_solution(x, basis) 



    def is_basis(self, basis):
        """
        Checks if the given base indices form a valid basis for the current model.

        Returns
        -------
        result : bool
            Whether or not the base indices form a valid basis.

        """
        if not self._is_sef:
            raise ArithmeticError() # requires sef form ? TODO REMOVE

        basis = self.__array_like_to_list(basis) #optimize as it is called twice from parent
        basis = self.__convert_indices(basis)

        if not self._A.shape[0] == len(basis):
            raise ValueError()

        test = np.linalg.det(self._A[:, basis])
        tolerance = 0

        if test < 1.0e-5:
            tolerance = 1.0e-9
        
        return not math.isclose(np.linalg.det(self._A[:, basis]), 0, abs_tol=tolerance)



    def is_feasible_basis(self, basis):
        """
        Tests if the given basis is feasible.

        Returns
        -------
        result : bool
            Whether or not the basis is feasible.

        """
        return self.is_basis(basis) and (self.compute_basic_solution(basis) >= 0).all()



    def compute_basic_solution(self, basis):
        """
        Computes the basic solution corresponding to the specified basis.

        basis : array-like of int
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



    def to_canonical_form(self, basis, show_steps=True, in_place=False):
        """
        Converts the linear program into canonical form for the given basis.

        Parameters
        ----------
        basis : array-like of int
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
            copy = self.copy()
            
            return copy.to_canonical_form(basis, show_steps, True)

        if not self._is_sef:
            raise ArithmeticError()

        show_steps and self.__append_to_steps(('1.01', basis))

        if not self.is_basis(basis):
            raise ValueError()

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])

        return self.__to_canonical_form(basis, show_steps)



    def __to_canonical_form(self, basis, show_steps):
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
        result : LinearProgrammingModel
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
        Creates the associated duality program. 

        Parameters
        ----------
        show_steps : bool, optional (default=True)
            Whether steps should be stored or not for this operation.

        Returns
        -------
        result : LinearProgram
            The computed result of the method.

        """
        
#          max          |   min
#--------------------|-------------
#              <=   >= 0
#   constraint  =   free variable
#              >=   <= 0
#
#    variable  >=0  >=
#              free =  
#              <=0  <=  constraint

        if not self.z == 0:
            raise ValueError()

        LinearProgram(self._A.T, self._c, self._b, 0)



    def two_phase_simplex(self, show_steps=True, in_place=False):
        """
        Computes the optimal solution for the linear program or returns a certificate of unboundedness
        using the simplex algorithm.

        Parameters
        ----------
        show_steps : bool, optional (default=True)
            Whether steps should be stored or not for this operation.

        in_place : bool, optional (default=True)
            Whether the operation should return a copy or be performed in place.

        Returns
        -------
        result : tuple (ndarray of float, LinearProgram)
            The copy of the linear program or self in canonical form.

        """
        if not self._is_sef:
            raise ArithmeticError()
        
        if not in_place:
            copy = self.copy()

            return copy.simplex_solution(show_steps, True)
        
        indices = np.where(self._b < 0)
        self._A[indices] *= -1
        self._b[indices] *= -1

        rows, columns = self._A.shape
        auxiliary_columns = rows + columns

        auxiliary_A = np.c_[self._A, np.eye(rows)]
        auxiliary_b = np._b.copy()

        auxiliary_c = np.zeros(self._A.shape[0])

        auxiliary_c[columns:] = 1

        auxiliary_program = LinearProgram()
        
        solution = (None, None)

        while solution[0]:
            solution = auxiliary_program.simplex_iteration(basis, in_place=True)
        # Find starting basis using auxilliary linear program
        # Loop over simplex iteration helper
        # TODO: make helper for simplex_iteration
        
        pass



    def simplex_solution(self, basis, show_steps=True, in_place=False): 
        """
        Computes simplex iterations until termination. 

        Parameters
        ----------
        basis : array-like of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        show_steps : bool, optional (default=True)
            Whether steps should be stored or not for this operation.

        in_place : bool, optional (default=True)
            Whether the operation should return a copy or be performed in place.

        Returns
        -------
        result : tuple (ndarray of float, ndarray of int, LinearProgram)

        """
        if not self._is_sef:
            raise ArithmeticError()
        
        if not in_place:
            copy = self.copy()

            return copy.simplex_solution(basis, show_steps, True)
        
        solution = None
        optimality_certificate = None
        counter = 0
        starting_A = self._A
        starting_c = self._c

        show_steps and self.__append_to_steps(("5.02", counter))
        show_steps and self.__append_to_steps(("5.01", self))

        while not isinstance(solution, np.ndarray) and not isinstance(solution, float):
            solution, basis, current = self.simplex_iteration(basis, show_steps, True)

            counter += 1

            show_steps and self.__append_to_steps(("5.02", counter))
            show_steps and self.__append_to_steps(("5.01", current))
        
        if isinstance(solution, float):
            basis = None
            
            show_steps and self.__append_to_steps("5.05")
        else:
            converted_basis = [i - 1 for i in basis]
            optimality_certificate = np.linalg.inv(starting_A[:, converted_basis].T) @ starting_c[converted_basis] #TODO optimize

            show_steps and self.__append_to_steps(("5.03", solution))
            show_steps and self.__append_to_steps(("5.04", basis))
            show_steps and self.__append_to_steps(("5.06", optimality_certificate))
        
        return solution, basis, optimality_certificate



    def simplex_iteration(self, basis, show_steps=True, in_place=False):
        """
        Computes a single iteration of the simplex algorithm with Bland's rule.

        Parameters
        ----------
        basis : array-like of int
            The column indices of the coefficient matrix that forms a basis. Use math indexing for format.

        show_steps : bool, optional (default=True)
            Whether steps should be stored or not for this operation.

        in_place : bool, optional (default=True)
            Whether the operation should return a copy or be performed in place.

        Returns
        -------
        result : tuple (ndarray of float, ndarray of int, LinearProgram)
            The copy of the linear program or self in canonical form. The first parameter indicates the
            solution vector, if any, and the second gives the current basis. If solution vector is infinity,
            the given linear program is unbounded.

        """
        if not self._is_sef:
            raise ArithmeticError()
        
        if not in_place:
            copy = self.copy()

            return copy.simplex_iteration(basis, show_steps, True)

        if not self.is_basis(basis): #basis might need to be sorted
            raise ValueError()

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])
        negative_indices = np.where(self._b < 0)

        self._b[negative_indices] *= -1
        self._A[negative_indices] *= -1

        self.__to_canonical_form(basis, show_steps)

        x = self.__compute_basic_solution(basis)

        N = [i for i in range(self._A.shape[1]) if not i in basis]

        if (self._c[N] <= 0).all():
            return x, np.array([i + 1 for i in basis]), self

        k = None

        for i in N:
            if self._c[i] > 0:
                k = i

                break

        Ak = self._A[:, k]

        if (Ak <= 0).all():
            return math.inf, np.array([i + 1 for i in basis]), self # optimize conversion back to math indexing

        t = np.amin([self._b[i] / Ak[i] for i in range(len(Ak)) if Ak[i] > 0])
        computed = self._b - t * Ak

        for i in range(len(computed)): 
            if math.isclose(computed[i], 0):
                basis.remove(basis[i])
                
                break

        basis.append(k)
        basis.sort()

        return None, np.array([i + 1 for i in basis]), self



    def verify_infeasibility(self, y):
        """
        Verifies the certificate of infeasibility.

        Parameters
        ----------
        y : array-like of int, float

        Returns
        -------
        result : bool
            Whether or not the certificate is valid. #review needed

        """
        if not self._is_sef: # Requires SEF?
            raise ArithmeticError()

        y_transpose = y.T

        yTA = y_transpose @ self._A
        yTb = y_transpose @ self._b

        return (yTA >= 0).all() and (yTb < 0).all()



    def verify_unboundedness(self, x, d):
        """
        Verifies the certificate of unboundedness.

        Parameters
        ----------
        x : array-like of int, float

        d : array-like of int, float

        Returns
        -------
        result :

        """
        Ad = self._A @ d
        cd = self._c @ d

        return all([
            np.allclose(Ad, 0),
            (d >= 0).all(),
            (cd > 0).all(),
            self.is_feasible(x)
        ])



    def verify_optimality(self, certificate):
        """
        Verifies the certificate of optimality.

        Parameters
        ----------
        certificate : array-like of float
            The certificate

        Returns
        -------
        result : bool
            Whether the given certificate certifies optimality or not.

        """
        # BUG: simplex with in_place=True overrides original program
        certificate = self.__to_ndarray(certificate)

        if not self._is_sef: # Requires SEF?
            raise ArithmeticError()
        
        return (self._c - certificate @ self._A <= 0).all()



    def is_feasible(self, x, show_steps=True):
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

            if not i in self._free_variables and x[i] < 0:
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



    def graph_feasible_region(self):
        """
        Graphs the feasible region of the linear program. Only supports 2 dimensional visualization.
        Constraints for the program must be in the form Ax <= b or Ax >= b.

        """
        if not self._A.shape[1] == 2:
            raise ArithmeticError()

        if not len(self._inequality_indices) == self._b.shape[0]:
            raise ArithmeticError()

        if not all(i == "<=" for i in list(self._inequality_indices.values())):
            if not all(i == ">=" for i in list(self._inequality_indices.values())):
                raise ArithmeticError()

        inequality = self.inequalities[0]

        A = np.array([[0, 0], [0, 0]])
        b = np.array([0, 0])

        shape = self._A.shape

        points = []

        # for tracking if an inconsistent system of equations has been given
        equations = {}

        # graph_single_line is called when only one line is provided, or two lines are provided but
        # one is horizontal/vertical
        def graph_single_line(newA, newb, ineq, custom_edge = False, value = 0, horizontal = False):
            """
            Given A, b, and an inequality, graphs the region above/below the given line according to the
            given inequality. If custom_edge is true, one of the edges can be specified at the given
            value and whether it's horizontal or vertical.

            This method directly graphs the line. Return after calling this method.

            Invariant: A is a tuple and b is an int. newA and newb are used in order to avoid naming conflicts.

            Method: find x- and y- intercepts, then set outer bounds based on intercepts
            if custom_edge is true, then the value given is set as one of the outer bounds
                - if horizontal is true, given value is a y-value; otherwise it's an x-value
            """

            if ineq == "=":
                print("Error: given inequality in graph_single_line is '='. Blame the programmer.")
                exit()

            points = []  # clear points in case some were added previously

            # check if the given line will ever intercept x or y axes
            has_x_inter = newA[0] != 0
            has_y_inter = newA[1] != 0
            x_intercept = 0
            y_intercept = 0

            # calculate intercepts
            if has_x_inter:
                x_intercept = newb / newA[0]  # since Ax = b <=> a_0x + a_1y = b; x = b/a_0 - a_1 * 0
            if has_y_inter:
                y_intercept = newb / newA[1]

            # if custom edge is given, modify intercepts accordingly
            # TODO: fix calculation of intercepts
            if custom_edge:
                if has_x_inter and horizontal:
                    x_intercept = value / newA[0]
                if has_y_inter and not horizontal:
                    y_intercept = value / newA[1]


            # horizontal line given, draw a rectangle
            if not has_x_inter:
                # set boundaries
                y_top = y_intercept
                y_bottom = -y_intercept
                x_left = -y_intercept
                x_right = y_intercept

                if ineq == "<=":
                    if y_intercept < 0:
                        y_bottom = y_intercept * 2
                        x_left = y_bottom
                    elif y_intercept == 0:
                        y_bottom = -10
                        x_left = -10

                    if custom_edge and not horizontal:
                        x_right = value
                        x_left = -value

                        if value < 0:
                            x_left = value * 2
                        elif value == 0:
                            x_left = -10

                else:  # inequality is >=
                    y_top = y_intercept * 2
                    y_bottom = y_intercept
                    x_left = y_intercept
                    x_right = y_intercept * 2

                    if y_intercept < 0:
                        y_top = -y_intercept
                        x_right = y_top
                    if y_intercept == 0:
                        y_bottom = 10
                        x_left = 10

                    if custom_edge and not horizontal:
                        x_right = value * 2
                        x_left = value

                        if value < 0:
                            x_right = -value
                        elif value == 0:
                            x_right = 10
                    

                # add points sequentially
                points.append([x_right, y_top])
                points.append([x_right, y_bottom])
                points.append([x_left, y_bottom])
                points.append([x_left, y_top])


            # vertical line given, draw a rectangle
            elif not has_y_inter:
                # set boundaries
                y_top = x_intercept
                y_bottom = -x_intercept
                x_left = -x_intercept
                x_right = x_intercept

                if ineq == "<=":
                    if x_intercept < 0:
                        y_bottom = x_intercept * 2
                        x_left = y_bottom
                    elif x_intercept == 0:
                        y_bottom = -10
                        x_left = -10

                    if custom_edge and horizontal:
                        y_top = value
                        y_bottom = -value

                        if value < 0:
                            y_bottom = value * 2
                        elif value == 0:
                            y_bottom = -10

                else:  # inequality is >=
                    y_top = x_intercept * 2
                    y_bottom = x_intercept
                    x_left = x_intercept
                    x_right = x_intercept * 2

                    if x_intercept < 0:
                        y_top = -x_intercept
                        x_right = y_top
                    if x_intercept == 0:
                        y_bottom = 10
                        x_left = 10

                    if custom_edge and horizontal:
                        y_top = value * 2
                        y_bottom = value

                        if value < 0:
                            y_top = -value
                        elif value == 0:
                            y_top = 10
                    

                # add points sequentially
                points.append([x_right, y_top])
                points.append([x_right, y_bottom])
                points.append([x_left, y_bottom])
                points.append([x_left, y_top])


            # neither horizontal nor vertical, draw a triangle
            elif ineq == "<=":
                point1 = [x_intercept, 0]
                point2 = [0, y_intercept]
                point3 = [0, 0]  # changed later, if necessary

                if custom_edge:
                    if horizontal:
                        point1 = [x_intercept, value]
                    else:
                        point2 = [value, y_intercept]
                    
                    point3 = [x_intercept, y_intercept]
                else:
                    if x_intercept < 0:
                        point3[0] = x_intercept
                    if y_intercept < 0:
                        point3[1] = y_intercept

                points.append(point1)
                points.append(point2)
                points.append(point3)
                

            else:  # inequality is >=
                point1 = [x_intercept, 0]
                point2 = [0, y_intercept]
                point3 = [x_intercept, y_intercept]  # changed later, if necessary

                if custom_edge:
                    if horizontal:
                        point1 = [x_intercept, value]
                        point3 = [0, value]
                    else:
                        point2 = [value, y_intercept]
                        point3 = [value, 0]
                else:
                    if x_intercept < 0:
                        point3[0] = 0
                    if y_intercept < 0:
                        point3[1] = 0

                points.append(point1)
                points.append(point2)
                points.append(point3)


            points.append(points[0])  # add the first point again to create a closed loop

            xs, ys = zip(*points)

            plt.figure()
            plt.plot(xs, ys)
            plt.grid()
            plt.fill(xs, ys)
            plt.show()



        if shape[0] == 1:
            # only one line was given, call graph_single_line
            graph_single_line(self._A[0], self._b, inequality)
            return


        # get intersect points of inequalities/lines
        # points are sorted later, only if necessary
        for i in range(shape[0]):
            for j in range(shape[0]):
                if i < j:
                    A[0, :] = self._A[i, :]
                    A[1, :] = self._A[j, :]

                    b[0] = self._b[i]
                    b[1] = self._b[j]

                    line1 = (self._A[i, 0], self._A[i, 1])
                    line2 = (self._A[j, 0], self._A[j, 1])

                    # check if equation already exists and gives different answer
                    if equations.__contains__(line1) and not math.isclose(equations[line1], b[0]):
                        print("Error: inconsistent system of equations")  # make more specific later
                        exit()
                    elif equations.__contains__(line2) and not math.isclose(equations[line2], b[1]):
                        print("Error: inconsistent system of equations")  # make more specific later
                        exit()

                    equations[line1] = b[0]
                    equations[line2] = b[1]

                    # if both lines are horizontal/vertical, skip point
                    if A[0][0] == 0 and A[1][0] == 0:
                        continue
                    elif A[0][1] == 0 and A[1][1] == 0:
                        continue

                    point = np.linalg.solve(A, b)
                    points.append(point)


        # check if each point satisfies every inequality
        new_points = []
        for point in points:
            valid_point = True

            for i in range(len(self._A)):
                value = point[0] * self._A[i, 0] + point[1] * self._A[i, 1]

                if self.inequalities[i] == "<=":
                    if value > self._b[i] and not math.isclose(value, self._b[i]):
                        valid_point = False
                elif self.inequalities[i] == ">=":
                    if value < self._b[i] and not math.isclose(value, self._b[i]):
                        valid_point = False
                else:
                    if not math.isclose(value, self._b[i]):
                        valid_point = False

            if valid_point:
                new_points.append(point)

        points = copy.deepcopy(new_points)


        # if fewer than 3 points exist, need to add boundaries on edges (can't draw infinitely)
        if len(points) <= 2:

            # only occurs when all but one line is parallel
            # results from inconsistent system, should be caught above
            if len(points) == 2:
                print("Error, two points found but system was not inconsistent??? This code is garbo.")
                exit()

            # occurs when only 2 lines are present
            # inconsistent system can cause this, but should be caught above
            elif len(points) == 1:
                """ 
                METHOD:
                # polygon will be 3- to 5-sided, depending on slope of lines provided
                # depending if sign is >= or <=, pick higher/lower of one line's intercept with the preset x-max
                (x-max is absolute value of first point x-value * 5, same for y-max)
                then add x-max, y-max to points (or min for <=), then other line's intercept (check if intercept is beyond -x-max)
                exit out of if statement at this point, regular code will handle rest
                x-max, y-max is to set a bound on the projection
                """

                if not shape[0] == 2:
                    print("Error:", shape[0], "lines are present - logic went wrong... who wrote this code?!")
                    exit()

                # cannot draw region if inequalities given are equations
                if inequality == "=":
                    print("Error: ")
                    exit()

                point = points[0]

                # find x, y min and max
                x_range = abs(point[0] * 10)
                x_range = max(x_range, 10)  # set minimum range in case point is on x or y axis
                x_max = point[0] + x_range
                x_min = point[0] - x_range

                y_range = abs(point[1] * 5)
                y_range = max(y_range, 5)
                y_max = point[1] + y_range
                y_min = point[1] - y_range

                print("x_max is", x_max)
                print("x_min is", x_min)
                print("y_max is", y_max)
                print("y_min is", y_min)

                # get equations for the two lines
                line1 = (self._A[0, 0], self._A[0, 1])
                line2 = (self._A[1, 0], self._A[1, 1])

                # list for each line's intercepts with edge of screen (x/y max and min)
                l1points = []
                l2points = []

                
                # calculate intercepts of both lines with all edges
                # then add intercepts and corners in order, clockwise
                # both lines cannot be both horizontal or both vertical, otherwise point would not be found
                # NOTE: if any line is horizontal or vertical, call graph_single line with the horizontal/vertical line as an edge instead

                # calculate line1
                # line1 is horizontal
                if line1[0] == 0:
                    graph_single_line(line2, self._b[1], inequality, True, self._b[0], True)
                    return

                # line1 is vertical
                elif line1[1] == 0:
                    graph_single_line(line2, self._b[1], inequality, True, self._b[0], False)
                    return

                else:
                    # find intercept with right edge
                    A[0, :] = np.array([line1[0], line1[1]])
                    A[1, :] = np.array([1, 0])  # create vertical line at x_max
                    b[0] = self._b[0, :]
                    b[1] = np.array([x_max])
                    r_point = np.linalg.solve(A, b)

                    # find intercept with bottom edge
                    A[0, :] = np.array([line1[0], line1[1]])
                    A[1, :] = np.array([0, 1])  # create horizontal line at y_min
                    b[0] = self._b[0, :]
                    b[1] = np.array([y_min])
                    b_point = np.linalg.solve(A, b)

                    # find intercept with left edge
                    A[0, :] = np.array([line1[0], line1[1]])
                    A[1, :] = np.array([1, 0])  # create vertical line at x_min
                    b[0] = self._b[0, :]
                    b[1] = np.array([x_min])
                    l_point = np.linalg.solve(A, b)

                    # find intercept with top edge
                    A[0, :] = np.array([line1[0], line1[1]])
                    A[1, :] = np.array([0, 1])  # create horizontal line at y_max
                    b[0] = self._b[0, :]
                    b[1] = np.array([y_max])
                    t_point = np.linalg.solve(A, b)

                    l1points.append(r_point)
                    l1points.append(l_point)
                    l1points.append(b_point)
                    l1points.append(t_point)

                    print("l1points are", l1points)

                # line 2
                # line2 is horizontal
                if line2[0] == 0:
                    graph_single_line(line1, self._b[0], inequality, True, self._b[1], True)
                    return

                # line2 is vertical
                elif line2[1] == 0:
                    graph_single_line(line1, self._b[0], inequality, True, self._b[1], False)
                    return

                else:
                    # find intercept with right edge
                    A[0, :] = np.array([line2[0], line2[1]])
                    A[1, :] = np.array([1, 0])  # create vertical line at x_max
                    b[0] = self._b[1, :]
                    b[1] = np.array([x_max])
                    r_point = np.linalg.solve(A, b)

                    # find intercept with bottom edge
                    A[0, :] = np.array([line2[0], line2[1]])
                    A[1, :] = np.array([0, 1])  # create horizontal line at y_min
                    b[0] = self._b[1, :]
                    b[1] = np.array([y_min])
                    b_point = np.linalg.solve(A, b)

                    # find intercept with left edge
                    A[0, :] = np.array([line2[0], line2[1]])
                    A[1, :] = np.array([1, 0])  # create vertical line at x_min
                    b[0] = self._b[1, :]
                    b[1] = np.array([x_min])
                    l_point = np.linalg.solve(A, b)

                    # find intercept with top edge
                    A[0, :] = np.array([line2[0], line2[1]])
                    A[1, :] = np.array([0, 1])  # create horizontal line at y_max
                    b[0] = self._b[1, :]
                    b[1] = np.array([y_max])
                    t_point = np.linalg.solve(A, b)

                    l2points.append(r_point)
                    l2points.append(l_point)
                    l2points.append(b_point)
                    l2points.append(t_point)

                    print("l2points are", l2points)


                if inequality == "<=":
                    '''
                    Method: find the points that intersect x_max on each line and choose the lowest one
                    If this point is below y_min, choose the point on that line that intersects with y_min instead and append to points

                    All cases with vertical and horizontal lines have already been filtered out above and set to graph_single_line
                    '''

                    # find right point(s)
                    l1_right_point = ((x_min, y_max))  # set to other extreme
                    l2_right_point = ((x_min, y_max))
                    l1_rightmost = False  # tracks if line1 or line2 is the rightmost

                    # find the point on each line that intersects with the right edge
                    for point1 in l1points:
                        if math.isclose(point1[0], x_max):
                            l1_right_point = point1
                            break  # there should only be at most one point on the right edge, otherwise I messed up bad

                    for point2 in l2points:
                        if math.isclose(point2[0], x_max):
                            l2_right_point = point2
                            break

                    # NOTE: vertical/horizontal line cases have been moved to len(points) == 0 case
                    # find lower one of l1_right_point and l2_right_point, select as right point
                    if l1_right_point[1] < l2_right_point[1]:
                        l1_rightmost = True
                        right_point = l1_right_point
                    else:
                        right_point = l2_right_point

                    if right_point[1] < y_min:  
                        # if point of intersect with right edge is below bottom edge, then pick point on line with bottom edge instead
                        if l1_rightmost:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_min):
                                    right_point = point1
                        else:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_min):
                                    right_point = point2
                    elif right_point[1] > y_max:
                        # if point of intersect with right edge is above top edge, pick point on line w/ top edge instead
                        # add top right corner after
                        if l1_rightmost:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_max):
                                    right_point = point1
                        else:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_max):
                                    right_point = point2

                    points.append(right_point)
                    print("Added point", right_point, "on right")

                    if math.isclose(right_point[1], y_max):
                        points.append((x_max, y_max))  # add top right corner to polygon if necessary
                        print("Added top right corner at", (x_max, y_max))

                    if not math.isclose(right_point[1], y_min):
                        points.append((x_max, y_min))  # add bottom right corner to polygon if necessary
                        print("Added bottom right corner at", (x_max, y_min))


                    # find left point(s)
                    left_point = ((x_max, y_max))  # set to other extreme
                    if l1_rightmost:
                        # find the point on l2 that is on x_min
                        for point2 in l2points:
                            if math.isclose(point2[0], x_min):
                                left_point = point2
                                break
                        
                        # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                        if left_point[1] > y_max:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_max):
                                    left_point = point2
                        elif left_point[1] < y_min:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_min):
                                    left_point = point2
                    else:
                        # same logic as above, but for l1 instead
                        for point1 in l1points:
                            if math.isclose(point1[0], x_min):
                                left_point = point1
                                break

                        # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                        if left_point[1] > y_max:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_max):
                                    left_point = point1
                        elif left_point[1] < y_min:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_min):
                                    left_point = point1

                    if not math.isclose(left_point[1], y_min):
                        points.append((x_min, y_min))  # add bottom left corner to polygon if necessary
                        print("Added bottom left corner at", (x_min, y_min))

                    if math.isclose(left_point[1], y_max):
                        points.append((x_min, y_max))  # add top left corner to polygon if necessary
                        print("Added top left corner at", (x_min, y_max))

                    points.append(left_point)
                    print("Added point", left_point, "on left")


                elif inequality == ">=":
                    '''
                    Same method as above, only swap "lowest" and "highest"
                    For more detailed comments, see above
                    '''
                    # find right point(s)
                    l1_right_point = ((x_min, y_min))  # set to other extreme
                    l2_right_point = ((x_min, y_min))
                    l1_rightmost = False

                    # find the point on each line that intersects with the right edge
                    for point1 in l1points:
                        if math.isclose(point1[0], x_max):
                            l1_right_point = point1
                            break
                    for point2 in l2points:
                        if math.isclose(point2[0], x_max):
                            l2_right_point = point2
                            break

                    # find higher one of l1_right_point and l2_right_point, select as right point
                    if l1_right_point[1] > l2_right_point[1]:
                        l1_rightmost = True
                        right_point = l1_right_point
                    else:
                        right_point = l2_right_point

                    if right_point[1] > y_max:
                        # if point of intersect with right edge is above top edge, pick point on line w/ top edge instead
                        if l1_rightmost:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_max):
                                    right_point = point1
                        else:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_max):
                                    right_point = point2
                    elif right_point[1] < y_min:  
                        # if point of intersect with right edge is below bottom edge, then pick point on line with bottom edge instead
                        # add bottom right corner after
                        if l1_rightmost:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_min):
                                    right_point = point1
                        else:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_min):
                                    right_point = point2

                    points.append(right_point)
                    print("Added point", right_point, "on right")

                    if math.isclose(right_point[1], y_min):
                        points.append((x_max, y_min))  # add bottom right corner to polygon if necessary
                        print("Added top right corner at", (x_max, y_min))

                    if not math.isclose(right_point[1], y_max):
                        points.append((x_max, y_max))  # add top right corner to polygon if necessary
                        print("Added top right corner at", (x_max, y_max))

                    
                    # find left point(s)
                    left_point = ((x_max, y_min))  # set to other extreme
                    if l1_rightmost:
                        # find the point on l2 that is on x_min
                        # if this point is above y_max or below y_min, then pick the point on either that is highest in x
                        for point2 in l2points:
                            if math.isclose(point2[0], x_min):
                                left_point = point2
                                break

                        # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                        if left_point[1] > y_max:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_max):
                                    left_point = point2
                        elif left_point[1] < y_min:
                            for point2 in l2points:
                                if math.isclose(point2[1], y_min):
                                    left_point = point2
                    else:
                        # same logic as above, but for l1 instead
                        for point1 in l1points:
                            if math.isclose(point1[0], x_min):
                                left_point = point1
                                break
                        
                        # if this point is above y_max or below y_min, then pick the point on the top/bottom edge respectively
                        if left_point[1] > y_max:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_max):
                                    left_point = point1
                        elif left_point[1] < y_min:
                            for point1 in l1points:
                                if math.isclose(point1[1], y_min):
                                    left_point = point1

                    if not math.isclose(left_point[1], y_max):
                        points.append((x_min, y_max))  # add top left corner to polygon if necessary
                        print("Added top left corner at", (x_min, y_max))

                    if math.isclose(left_point[1], y_min):
                        points.append((x_min, y_min))  # add bottom left corner to polygon if necessary
                        print("Added bottom left corner at", (x_min, y_min))

                    points.append(left_point)
                    print("Added point", left_point, "on left")
                    

                # code should not get to this point, since the filter was added before any calculations above
                else:
                    print("Error: inequality is = symbol, even though it should have been filtered out already.")
                    exit()

            
            # no intersect points - occurs when only 1 line is present
            # this clause should never be executed, since the 1 line case should not be solvable with np.linalg.solve
            # inconsistent system can cause this, but should be caught above
            else:
                print("Error: somehow no points were found. The person who wrote this reallllly messed up.")
                exit()


        # 3 or more points exist - don't need to add boundaries or other points, but need to sort points so they
        # are in a clockwise order for drawing properly
        # need to sort these points due to how pyplot takes input to draw polygons
        else:
            """
            Method: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
            """
            origin = points[0]  # set origin to the first point in the list
            refvec = [0, 1]  # reference vector for calculations

            def find_cw_angle_and_distance(point):
                # get vector between point and the origin
                v = [point[0] - origin[0], point[1] - origin[1]]
                # get length of vector
                v_len = math.hypot(v[0], v[1])

                # if the length is zero, there is no angle nor distance - return
                if v_len == 0:
                    return -math.pi, 0

                # normalize the vector in order to find the directional angle
                norm = [v[0] / v_len, v[1]/v_len]
                dot_product = norm[0] * refvec[0] + norm[1] * refvec[1]
                diff_product = refvec[1] * norm[0] - refvec[0] * norm[1]

                angle = math.atan2(diff_product, dot_product)

                # convert negative angles to positive angles
                if angle < 0:
                    return 2 * math.pi + angle, v_len
                return angle, v_len


            # use new function with sorted function to sort points list
            points = sorted(points, key = find_cw_angle_and_distance)


        points.append(points[0])  # add the first point again to create a closed loop

        xs, ys = zip(*points)

        plt.figure()
        plt.plot(xs, ys)
        plt.grid()
        plt.fill(xs, ys)
        plt.show()



    def evaluate(self, x):
        """
        Evaluates the objective function with a given vector. Does not check if x satisfies the constraints.

        Returns
        -------
        result : float

        """
        x = self.__to_ndarray(x)
        
        if not self.__is_vector_of_size(x, self._c.shape[0]):
            raise ValueError()
        
        return self._c @ x + self._z



    def value_of(self, x):
        """
        Computes the value of a given x vector. The vector must satisfy the constraints.
        
        Returns
        -------
        result : float
        
        """
        x = self.__to_ndarray(x)

        if not self.is_feasible(x):
            raise ValueError()
        
        return self.evaluate(x)



    def copy(self):
        """
        Creates a copy of the current model.

        Returns
        -------
        result : LinearProgram

        """
        p = LinearProgram(self._A.copy(), self._b.copy(), self._c.copy(), self._z, self._objective)

        p._inequality_indices = self._inequality_indices.copy()
        p._free_variables = self._free_variables.copy()
        p._steps = self._steps.copy()
        p._is_sef = self._is_sef

        return p



    def to_sef(self, show_steps=True, in_place=False):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        if not in_place:
            copy = self.copy()

            return copy.to_sef(show_steps, True)

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

        for i in range(len(self._free_variables)):
            index = self._free_variables[i] + i

            self._c = np.insert(self._c, index + 1, -self._c[index])
            self._A = np.insert(self._A, index + 1, -self._A[:, index], axis=1)

        self._free_variables = []

        for i in range(self._b.shape[0]):
            if i in self._inequality_indices:
                operator = self._inequality_indices[i]
                self._A = np.c_[self._A, np.zeros(self._A.shape[0])]
                self._c = np.r_[self._c, 0]

                if (operator == ">="):
                    self._A[i, -1] = -1
                elif (operator == "<="):
                    self._A[i, -1] = 1

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
            copy = self.copy()

            return copy.clear_steps()

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
                        elif not type(column) in [int, float]:
                            raise ValueError()
                elif not type(row) in [int, float]:
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
        
        basis.sort()

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



    def __get_free_variables(self):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        return list(map(lambda i: i + 1, self._free_variables))


    
    def __format_steps(self):
        """
        Converts expression to standard equality form.

        Returns
        -------
        result : LinearProgram

        """
        return functools.reduce((lambda previous, current: f"{previous}\n{current['text']}"), self.steps, "").strip()



    def __convert_indices(self, indices, min_value=None, max_value=None):
        """ 
        Converts from math indexing to array indexing.
        
        min_value is the closed lower bound allowed for minimum index value.
        max_value is the open upper bound allowed for minimum index value.

        """
        indices = list(map(lambda i: i - 1, indices))

        if len(indices) > 0:
            conditions = [
                not min_value == None and min(indices) < min_value,
                max_value and max(indices) >= max_value,
                any(not type(i) == int for i in indices)
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
        #return self.__is_in_rref()
