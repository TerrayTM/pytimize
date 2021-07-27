import numpy as np
import math

from ... import UnconstrainedProgram
from unittest import TestCase, main

class TestSteepestDescent(TestCase):
    def test_steepest_descent(self) -> None:
        def f(w):
            x, y = w
            return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

        def df(w):
            x, y = w
            return np.array([
                400 * x ** 3 - 400 * x * y + 2 * x - 2,
                200 * (y - x ** 2)
            ])

        program = UnconstrainedProgram(f)
        self.assertTrue(np.all(program.steepest_descent(np.array([0.5, 0.5]), df) - np.array([0.99, 0.99]) < 0.1))

if __name__ == "__main__":
    main()
