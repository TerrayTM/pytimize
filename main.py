import sys
import numpy as np

sys.path.append('./enums');

from objective import Objective

class LinearProgrammingProblem:
    def __init__(self, A, b, c, z, objective=Objective.max, operators=None, free_vars=[]):
        """
        Constructs a LP (P) formulation of the form 'func{cTx + z : Ax = b, vars >= 0}'
        where 'func' denotes either min or max, 'T' denotes the transpose operator, and
        'vars' denotes the variables that are constrained to be greater than or equal to 0.

        :param A: An m x n constraint coefficient matrix.
        :param b: A column vector with n float entries.
        :param c: A row vector with n float entries.
        :param z: A float constant.
        :para objective: Describes whether the objective is to maximize or minimize. Can only be 'max' or 'min'.
        :param operators: An array with m entries denoting the constraint type of each equation.
                          Entries can be '=', '>=' or '<='. If left empty, array will be autofilled with equal signs.
        :param free_vars: An array of integer indices describing the variables that are free.
                          Use standard math index numbering (meaning the first variable starts at index 1).

        """
        A = self.__to_ndarray(A)
        b = self.__to_ndarray(b)
        c = self.__to_ndarray(c)

        if not type(z) is float and not type(z) is int:
            raise TypeError()
        if not operators:
            operators = ['=' for i in range(b.shape[0])]

        self._A = A
        self._b = b
        self._c = c
        self._z = z
        self._objective = objective
        self._is_sef = all(
            operator == '=' for operator in operators) and len(free_vars) == 0
        self._operators = operators
        self._free_vars = [free_var - 1 for free_var in free_vars]

        if len(self._free_vars) > 0 and (min(self._free_vars) < 0 or max(self._free_vars) >= A.shape[1]):
            raise IndexError()



    def __str__(self):
        # To do: better display for formulation
        return f"A:\n{str(self._A)}\n\nb:\n{self._b}\n\nc:\n{self._c}\n\nz:\n{self._z}\n\nOperators:\n{self._operators}"



    def to_canonical_form(self, base_indices, show_steps=True):
        """
        Computes the canonical form of the formulation in terms of the given base indices.

        :param base_indices: An array of integer indices denoting the columns that form a base.
                             Use standard math index numbering.

        :return: Self

        """
        A_b = self._A[:, base_indices]

        c_b = self._c[base_indices]
        A_b_inverse = np.linalg.inv(A_b)
        y_transpose = (A_b_inverse.T @ c_b).T
        constant = y_transpose @ self._b + self._c
        coefficient_matrix = self._c.T - y_transpose @ self._A  # not matrix should be row vector

        self._A = A_b_inverse @ self._A
        self._b = A_b_inverse @ self._b

        self._z = constant
        self._c = coefficient_matrix

        return self



    def compute_simplex_solution(self):
        pass


    #Basis is in mathematical index
    #Separate this function into one private and one public
    def compute_simplex_iteration(self, basis):
        basis = map(lambda i: i + 1, basis)
        
        if min(basis) < 0 or max(basis) >= self._A.shape[1]:
            raise TypeError()
        
        self.to_canonical_form(basis)

        #x is basic feasible solution for B
        x = None

        N = [i for i in range(self._A.shape[1]) if not i in basis]

        if (self._c[:, N] <= 0).all():
            return x

        k = None

        for i in N:
            if self._c[:, i] > 0:
                k = i

                break
        
        if (A[:, k] <= 0).all():
            #unbounded
            return
        
        return self



    def verify_infeasibility(self, y):
        """
        Verifies the certificate of infeasibility.

        :param y:

        :return: A boolean value indicating if the certificate is valid.

        """
        y_transpose = y.T

        yTA = y_transpose @ self._A
        yTb = y_transpose @ self._b

        return all([
            (yTA >= 0).all(),
            (yTb < 0).all()
        ])



    def verify_unboundedness(self, x, d):
        """
        Verifies the certificate of unboundedness.

        :param x:
        :param d:

        :result: A boolean value indicating if the certificate is valid.

        """
        Ad = self._A @ d
        cTd = self._c.T @ d

        return all([
            (Ad == 0).all(),
            (d >= 0).all(),
            (cTd > 0).all(),
            self.is_solution_feasible(x)
        ])



    def verify_optimality(self, certificate):
        """
        Verifies the certificate of optimality.

        """
        pass



    def is_solution_feasible(self, x):
        """

        Checks if the given vector 'x' is a feasible solution.

        :param x: A row vector.

        :return: A boolean value indicating if the solution is feasible.

        """
        if not self.__is_vector_of_size(x, self._c.shape[0]):
            return False

        if self._is_sef:
            return (x >= 0).all() and np.array_equal(self._A @ x, self._b)

        for i in range(self._A.shape[0]):
            value = self._A[i, :] @ x
            
            if not i in self._free_vars and x[i] < 0:
                return False
            
            operator = self._operators[i]
            b = self._b[i]

            fail_conditions = [
                operator == '=' and value != b,
                operator == '>=' and value < b,
                operator == '<=' and value > b
            ]

            if any(fail_conditions):
                return False
        
        return True
    


    def evaluate_objective_function(self, x):
        pass
    


    def evaluate_value(self, x):
        pass



    def to_sef(self):
        """
        Converts expression to standard equality form.

        :return: Self

        """
        if self._is_sef:
            return

        if self._objective == 'min':
            self._c = -self._c
            self._objective = 'max'

        # ASSUME FREEE VARS IS IN ASC ORDER
        for i in range(len(self._free_vars)):
            free_var = self._free_vars[i]
            index = free_var + i

            self._c = np.insert(self._c, index + 1, -self._c[index])
            self._A = np.insert(self._A, index + 1, -self._A[:, index], axis=1)
        
        self._free_vars = []

        for i in range(len(self._operators)):
          operator = self._operators[i]

          if (not operator == '='):
            self._A = np.c_[self._A, np.zeros(self._A.shape[0])]
            self._c = np.r_[self._c, 0]

            if (operator == '>='):
              self._A[i, -1] = -1
            elif (operator == '<='):
              self._A[i, -1] = 1
            
            self._operators[i] = '='

        self._is_sef = True

        return self



    def is_solution_optimal(self, x):
        self.is_solution_feasible(x)
        pass



    def __to_ndarray(self, source):
        if isinstance(source, np.ndarray):
            # add dtype validation
            return source

        if isinstance(source, list):
        # Add array dimension validation
            return np.array(source, dtype=float)

        raise TypeError()



    def __is_vector_of_size(self, x, n):
        return all([
            type(x) is np.ndarray,
            x.ndim == 1,
            x.shape[0] == n
        ])



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
                if arr[row, col] == 0:
                    # haven't found a non-zero entry yet, search rest of row
                    continue

                elif has_zero_row:
                    # row has non-zero entries but there is a zero row above it
                    return False

                elif arr[row, col] == 1:
                    # found a leading one, check rest of column for zeros
                    row_has_leading_one = True
                    for r in range(shape[0]):
                        if r != row and arr[r, col] != 0:
                            return False
                    break

                else:
                    # row has non-zero entries but first number is not 1
                    return False

            if not row_has_leading_one:
                # row was empty
                has_zero_row = True
                
        return True



    def __rref(self, arr):
        """
        Returns an array that has been row reduced into row reduced echelon 
        form (RREF) using the Gauss-Jordan algorithm. 

        :param arr: An m x n matrix.

        :return: An m x n matrix in RREF that is row equivalent to the given matrix.

        """
        shape = arr.shape
        col = 0
        for row in range(shape[0]):
            # Get a 1 in the row,col entry
            if arr[row, col] != 1:
                i = 0
                while arr[row, col] == 0:
                    # If number in row,col is 0, find a lower row with a non-zero entry
                    # If all lower rows have a 0 in the column, move on to the next column
                    if i + row == shape[0]:
                        i = 0
                        col += 1
                        if col == shape[1]:
                            break
                        continue
                    if arr[row + i, col] != 0:
                        # Found a lower row with non-zero entry, swap rows
                        arr[[row, row + i]] = arr[[row + i, row]]
                        break
                    i += 1
                
                if col == shape[1]:
                    # Everything left is 0
                    break

                # Divide row by row,col entry to get a 1
                num = arr[row, col]
                for i in range(shape[1]):
                    arr[row, i] /= num

            # Subtract row from all other rows to get 0s in rest of col
            for i in range(shape[0]):
                if i != row:
                    multiple = arr[i, col] / arr[row, col]
                    for j in range(shape[1]):
                        arr[i, j] -= arr[row, j] * multiple
                        
            col += 1
            if col == shape[1]:
                break

        return arr



    def get_A(self):
        """ Returns constraint coefficient matrix. """
        return self._A



    def get_b(self):
        """ Returns constraint vector. """
        return self._b



    def get_c(self):
        """ Returns objective function coefficient vector. """
        return self._c



    def get_z(self):
        """ Returns objective function constant. """
        return self._z



    def get_operators(self):
        """ Returns the operators on the constraint. """
        return self._operators



    def get_objective(self):
        """ Returns the objective of this model. """
        return self._objective



    def get_is_sef(self):
        """ Returns if model is in SEF. """
        return self._is_sef




A = np.array([
    [1, 1, 1, 0, 0],
    [2, 1, 0, 1, 0],
    [-1, 1, 0, 0, 1]
])

c = np.array([2, 3, 0, 0, 0])

b = np.array([6, 10, 4]).T

z = 0.0

base_indices = np.array([0, 1, 3])

x = LinearProgrammingProblem(A, b, c, z)

x.to_canonical_form(base_indices)



a = np.array([[1, 2, 3], [2, 4, 6]])

b= np.zeros(3) #use 0 instead of np.zeros


A = np.array([
    [1,5,3],
    [2,-1,2],
    [1,2,-1]
])

b=np.array([
5,4,2
]).T

c=np.array([-1,2,-4]).T

lp = LinearProgrammingProblem(A=A, b=b, c=c, z=0, operators=['>=', '<=', '='], free_vars=[3], objective='min')


print(lp.to_sef())
