import sys
import numpy as np

sys.path.append("../optimization")

from optimizathon import LinearProgram

A = np.array([
    [1, 0, 2, 7, -1], 
    [0, 1, -4, -5, 3]
])
b = np.array([2, 1])
c = np.array([0, 0, 4, -11, -1])
z = 17

p = LinearProgram(A, b, c, z)
optimal_solution, optimal_basis, optimality_certificate = p.simplex_solution([1, 2])

print("Optimal Solution: {}".format(optimal_solution))
print("Optimal Basis: {}".format(optimal_basis))
print("Optimality Certificate: {}".format(optimality_certificate))

