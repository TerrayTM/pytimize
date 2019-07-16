import sys
import numpy as np

sys.path.append("../optimization")
sys.path.append("../optimization/enums")

from main import LinearProgrammingModel
from unittest import TestCase, main
from math import isclose

class TestComputeBasicSolution(TestCase):
    def test_compute_basic_solution(self):
        A = [
            [2, 1, 2, -1, 0, 0],
            [1, 0, -1, 2, 1, 0],
            [3, 0, 3, 1, 0, 1]
        ]
        b = [2, 1, 1]
        c = [1, 2, 3, 4, 5, 6]
        z = 5

        p = LinearProgrammingModel(A, b, c, z)

        test_one = p.compute_basic_solution(np.array([1, 5, 6]))
        test_one_expected = np.array([1, 0, 0, 0, 0, -2])
        # basic solution can have negative entries but not feasible basic solutions
        test_two = p.compute_basic_solution(np.array([2, 5, 6]))
        test_two_expected = np.array([0, 2, 0, 0, 1, 1])

        self.assertTrue(np.allclose(test_one, test_one_expected))    #add test message
        self.assertTrue(np.allclose(test_two, test_two_expected))


if __name__ == "__main__":
    main()
