import numpy as np

from pytimize.programs import LinearProgram

A = np.array([[1, 0, 2, 7, -1], [0, 1, -4, -5, 3]])
b = np.array([2, 1])
c = np.array([0, 0, 4, -11, -1])
z = 17

p = LinearProgram(A, b, c, z)

# Simplex method with initial basis
optimal_solution, optimal_basis, optimality_certificate = p.simplex([1, 2])

print("\nSimplex ================================================")
print(f"Optimal Solution: {optimal_solution}")
print(f"Optimal Basis: {optimal_basis}")
print(f"Optimality Certificate: {optimality_certificate}")

# Two phase simplex method
optimal_solution, optimal_basis, optimality_certificate = p.two_phase_simplex()

print("\nTwo Phase Simplex ======================================")
print(f"Optimal Solution: {optimal_solution}")
print(f"Optimal Basis: {optimal_basis}")
print(f"Optimality Certificate: {optimality_certificate}")
