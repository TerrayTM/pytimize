import numpy as np
import math

from ... import UnconstrainedProgram
from unittest import TestCase, main

class TestNewtonMethod(TestCase):
    def test_newton_method(self) -> None:
        def f(w):
            x, y = w
            return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

        def df(w):
            x, y = w
            return np.array([
                400 * x ** 3 - 400 * x * y + 2 * x - 2,
                200 * (y - x ** 2)
            ])

        def d2f(w):
            x, y = w
            return np.array([
                [1200 * x ** 2 - 400 * y + 2, -400 * x],
                [-400 * x, 200]
            ])

        program = UnconstrainedProgram(f)
        self.assertTrue(np.all(program.newton_method(np.array([0.5, 0.5]), df, d2f) - np.array([0.99, 0.99]) < 0.1))

if __name__ == "__main__":
    main()
