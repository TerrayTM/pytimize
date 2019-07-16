import sys
import numpy as np

sys.path.append("./enums")

from objective import Objective
from math import isclose, inf

# To do: redo pydoc comment style
# To do: update pydoc comments
# To do: check all and any expressions
# To do: for make independent rows, check for sef at end
class LinearProgrammingModel:
    def __init__(self, A, b, c, z, objective=Objective.max, inequalities=None, free_variables=None):
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

        objective : Objective, optional (default=Objective.max)
            The objective of the linear programming model.

        inequalities : array-like of {"=", ">=", "<="}, optional (default=None)
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

        if not isinstance(objective, Objective):
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
        self._is_sef = sef_condition and len(free_variables) == 0 and objective == Objective.max
        self._free_variables = free_variables


    # TODO: replace >= and <= in output with actual symbols (Google to find them)
    def __str__(self):
        # TODO: test the function :D
        output = ""
        shape = self._A.shape
        row_length = 0

        # find max length of row for formatting
        for i in range(shape[0]):
            length = len(str(self._A[i, :]))
            if length > row_length:
                row_length = length

        for i in range(shape[0]):
            spaces = row_length + 5  # format output nicely
            output += str(self._A[i, :])
            spaces -= len(str(self._A[i, :]))

            if i == shape[0] // 2:
                output += "x"
                spaces -= 1

            if i in self._inequality_indices:
                spaces -= 1
            
            output += " " * spaces
            
            if i in self._inequality_indices:
                output += self._inequality_indices[i]
            else:
                output += "="

            output += f"   {str(self._b[i])}\n"

        return f"Max {self._c}x + {self._z}\nSubject To:\n\n{output}"



    def is_canonical_form_for(self, basis):
        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])

        return all([
            self.is_basis(basis),
            np.allclose(self._A[basis], np.eye(len(basis))),
            np.allclose(self._c[basis], 0)
        ])


    
    def is_basic_solution(self, x, basis):
        if not self._is_sef:
            raise Error() # raise error if not in SEF form ?

        x = self.__to_ndarray(x)

        if not self.__is_vector_of_size(x, self._c):
            raise ValueError()

        # TODO Check if basis is changed by reference
        if not self.is_basis(basis):
            raise ValueError()

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])

        for i in range(self._c.shape[0]):
            if not i in basis and not isclose(self._c[i], 0):
                return False

        return np.allclose(self._A @ x, self._b)



    def is_feasible_basic_solution(self, x, basis):
        x = self.__to_ndarray(x)

        if not self.__is_vector_of_size(x, self._c):
            raise ValueError()

        return (x >= 0).all() and self.__is_basic_solution(x, basis) 



    def is_basis(self, basis):
        if not self._is_sef:
            raise Error() # requires sef form

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis)

        if not self._A.shape[0] == len(basis):
            raise ValueError()

        return not isclose(np.linalg.det(self._A[:, basis]), 0)



    def is_feasible_basis(self, basis):
        return self.is_basis(basis) and (self.compute_basic_solution(basis) >= 0).all()



    def compute_basic_solution(self, basis):
        if not self.is_basis(basis):
            raise Error()
        
        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis)

        return self.__compute_basic_solution(basis)


    #Move this to private methods
    def __compute_basic_solution(self, basis):
        components = np.linalg.inv(self._A[:, basis]) @ self._b
        solution = np.zeros(self._c.shape[0])
        
        basis.sort()

        for index, value in zip(basis, components):
            solution[index] = value
        
        return solution



    def to_canonical_form(self, basis, show_steps=True, in_place=False):
        """
        Computes the canonical form of the formulation in terms of the given base indices.

        :param basis: An array of integer indices denoting the columns that form a base.
                             Use standard math index numbering.

        :return: LinearProgrammingModel

        """
        if not in_place:
            copy = self.copy()
            
            return copy.to_canonical_form(basis, show_steps, True)

        if not self._is_sef:
            self.to_sef(True)  # To do: add test cases

        basis = self.__convert_indices(basis, 0, self._c.shape[0])

        A_b = self._A[:, basis]
        c_b = self._c[basis]
        A_b_inverse = np.linalg.inv(A_b)
        y_transpose = (A_b_inverse.T @ c_b).T

        A = A_b_inverse @ self._A
        b = A_b_inverse @ self._b
        c = self._c - y_transpose @ self._A
        z = y_transpose @ self._b + self._z

        self._A = A
        self._b = b
        self._c = c
        self._z = z

        return self



    def compute_simplex_solution(self, show_steps=True, in_place=False):
        pass



    #Separate this function into one private and one public
    def compute_simplex_iteration(self, basis, show_steps=True, in_place=False):
        if not self._is_sef:
            raise Error() #not sef
        
        if not in_place:
            copy = self.copy()

            return copy.compute_simplex_iteration(basis, True)

        basis = self.__array_like_to_list(basis)
        basis = self.__convert_indices(basis, 0, self._c.shape[0])
        
        self.to_canonical_form(basis, in_place=True)

        x = self.__compute_basic_solution(basis)

        N = [i for i in range(self._A.shape[1]) if not i in basis]

        if (self._c[:, N] <= 0).all():
            return x, self

        k = None

        for i in N:
            if self._c[:, i] > 0:
                k = i

                break

        Ak = self._A[:, k]

        if (Ak <= 0).all():
            return inf, self

        t = self._b / Ak
        t = np.amin(t[t > 0])

        basis.append(k)
        basis.remove(t)

        return None, self



    def verify_infeasibility(self, y):
        """
        Verifies the certificate of infeasibility.

        :param y:

        :return: A boolean value indicating if the certificate is valid.

        """
        y_transpose = y.T

        yTA = y_transpose @ self._A
        yTb = y_transpose @ self._b

        return (yTA >= 0).all() and (yTb < 0).all()



    def verify_unboundedness(self, x, d):
        """
        Verifies the certificate of unboundedness.

        :param x:
        :param d:

        :result: A boolean value indicating if the certificate is valid.

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

        """
        pass



    def is_feasible(self, x, show_steps=True):
        """
        Checks if the given vector "x" is a feasible solution.

        :param x: A row vector.

        :return: A boolean value indicating if the solution is feasible.

        """
        x = self.__to_ndarray(x)

        if not self.__is_vector_of_size(x, self._c.shape[0]):
            raise ValueError()

        show_steps and self._steps.append(f"Is {x} feasible?")

        if self._is_sef:
            all_nonnegative = (x >= 0).all()
            satisfy_constraints = np.allclose(self._A @ x, self._b)
            is_feasible = all_nonnegative and satisfy_constraints

            show_steps and is_feasible and self._steps.extend([
                f"{x} is feasible because:", 
                "* P is in SEF.", 
                f"* All entries of {x} are nonnegative.",
                "* Constraints are satisfied (Ax = b).",
            ])
            show_steps and not is_feasible and self._steps.extend(list(filter(None, [
                f"{x} is not feasible because:",
                "* P is in SEF.",
                f"* Some entries of {x} is negative." if not all_nonnegative else None,
                "* Constraints are not satisfied (Ax ≠ b)." if not satisfy_constraints else None
            ])))

            return is_feasible

        for i in range(x.shape[0]):
            row = self._A[i, :]
            value = row @ x

            if not i in self._free_variables and x[i] < 0:
                show_steps and self._steps.extend([
                    f"{x} is not feasible because:",
                    f"* Entry at index {i + 1} is negative and it is not a free variable."
                ])

                return False

            if i in self._inequality_indices:
                current = self._inequality_indices[i]

                if current == "<=" and value > self._b[i]:
                    show_steps and self._steps.extend([
                        f"{x} is not feasible because:",
                        f"* {row} • {x} = {value} and {value} is not ≤ {self._b[i]}."
                    ])

                    return False
                elif current == ">=" and value < self._b[i]:
                    show_steps and self._steps.extend([
                        f"{x} is not feasible because:",
                        f"* {row} • {x} = {value} and {value} is not ≥ {self._b[i]}."
                    ])

                    return False
            elif not isclose(value, self._b[i]):
                show_steps and self._steps.extend([
                    f"{x} is not feasible because:",
                    f"* {row} • {x} = {value} and {value} ≠ {self._b[i]}."
                ])

                return False
        
        show_steps and self._steps.extend([
            f"{x} is feasible because:",
            "* Constraints are satisfied.",
            "* All entries are either nonnegative or is a free variable."
        ])

        return True



    def evaluate(self, x):
        """
        Evaluates the objective function with a given x vector. Does not 
        check if x satisfies the constraints.

        :return: float
        """
        x = self.__to_ndarray(x)
        
        if not self.__is_vector_of_size(x, self._c[0]):
            raise ValueError()
        
        return self._c @ x + self._z



    def value_of(self, x):
        """
        Computes the value of a given x vector. The vector must satisfy the constraints.
        
        :return: float
        """
        x = self.__to_ndarray(x)

        if not self.is_feasible(x):
            raise ValueError()
        
        return self.evaluate(x)



    def copy(self):
        """
        Creates a copy of the current model.

        :return: LinearProgrammingModel

        """
        p = LinearProgrammingModel(self._A.copy(), self._b.copy(), self._c.copy(), self._z, self._objective)

        p._inequality_indices = self._inequality_indices.copy()
        p._free_variables = self._free_variables.copy()
        p._steps = self._steps.copy()
        p._is_sef = self._is_sef

        return p



    def to_sef(self, in_place=False):
        """
        Converts expression to standard equality form.

        :return: LinearProgrammingModel

        """
        if not in_place:
            copy = self.copy()

            return copy.to_sef(in_place=True)

        if self._is_sef:
            return self

        if self._objective == Objective.min:
            self._c = -self._c
            self._objective = Objective.max

        for i in range(len(self._free_variables)):
            free_variables = self._free_variables[i]
            index = free_variables + i

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

        self.inequality_indices = {}
        self._is_sef = True

        return self



    def is_solution_optimal(self, x):
        self.is_feasible(x)



    def clear_steps(self, in_place=False):
        if not in_place:
            copy = self.copy()

            return copy.clear_steps()

        self._steps = []

        return self



    def __to_ndarray(self, source):
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



    def __get_inequalities(self):
        inequalities = []

        for i in range(self._b.shape[0]):
            if i in self._inequality_indices:
                inequalities.append(self._inequality_indices[i])
            else:
                inequalities.append("=")

        return inequalities



    def __get_free_variables(self):
        return list(map(lambda i: i + 1, self._free_variables))



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
        return isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == dimension



    def __array_like_to_list(self, array_like):
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
                if isclose(arr[row, col], 0):
                    # have not found a non-zero entry yet, search rest of row
                    continue

                elif has_zero_row:
                    # row has non-zero entries but there is a zero row above it
                    return False

                elif isclose(arr[row, col], 1):
                    # found a leading one, check rest of column for zeros
                    row_has_leading_one = True
                    for r in range(shape[0]):
                        if not r == row and not isclose(arr[r, col], 0):
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
            if not isclose(arr[row, col], 1):
                i = 0
                while isclose(arr[row, col], 0):
                    # If number in row,col is 0, find a lower row with a non-zero entry
                    # If all lower rows have a 0 in the column, move on to the next column
                    if isclose(i + row, shape[0]):
                        i = 0
                        col += 1
                        if isclose(col, shape[1]):
                            break
                        continue
                    if not isclose(arr[row + i, col], 0):
                        # Found a lower row with non-zero entry, swap rows
                        arr[[row, row + i]] = arr[[row + i, row]]
                        break
                    i += 1
                
                if isclose(col, shape[1]):
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
            if isclose(col, shape[1]):
                break

        return arr




    def __make_indep(self):
        """
        Converts the augmented matrix [A|b] to be linearly independent.
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
                    if isclose(arr[i, col], 0):
                        if isclose(arr[j, col], 0):
                            # both zero entries, move on to next column
                            continue
                        # one row has a zero while other doesn't, move on to next row
                        duplicate = False
                        break
                    elif isclose(arr[j, col], 0):
                        # one row has a zero while other doesn't
                        duplicate = False
                        break
                    
                    if col == 0:
                        multiple = arr[i, col] / arr[j, col]
                    elif not isclose(arr[i, col] / arr[j, col], multiple):
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
        """ Gets the constraint coefficient matrix. """
        return self._A



    @property
    def b(self):
        """ Gets the constraint vector. """
        return self._b



    @property
    def c(self):
        """ Gets the coefficient vector. """
        return self._c



    @property
    def z(self):
        """ Gets the constant. """
        return self._z



    @property
    def inequalities(self):
        """ Gets the constraint inequalities. """
        return self.__get_inequalities()



    @property
    def objective(self):
        """ Gets the objective. """
        return self._objective



    @property
    def is_sef(self):
        """ Gets if the model is in standard equality form. """
        return self._is_sef



    @property
    def steps(self):
        """ Gets the steps of all operations performed. """
        return self._steps
    


    @property
    def free_variables(self):
        """ Gets the free variable indices. """
        return self.__get_free_variables()



    @property
    def is_in_rref(self):
        """ Gets if the constraint is in RREF. Model must be in SEF. """
        return self.__is_in_rref()

