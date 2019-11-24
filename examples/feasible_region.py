import sys
import numpy as np

sys.path.append("../optimization")

from main import LinearProgram

A = np.array([
    [5, -1], 
    [1, 1],
    [-1, 0], 
    [0, -1]
])
b = np.array([10, 19, 0, 0])
c = np.array([1, 1])
z = 0

p = LinearProgram(A, b, c, z, inequalities=["<=", "<=", "<=", "<="])
p.graph_feasible_region()
