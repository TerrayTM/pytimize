import numpy as np

from ...linear import maximize, minimize, x, variables
from ....programs import LinearProgram
from unittest import TestCase, main

class TestIntegration(TestCase):
    def test_max_program(self):
        p = maximize(x[1] + x[2]).subject_to(
            x[1] <= 5,
            x[2] <= 5
        ).whereAllNegative()

        self.assertIsInstance(p.program, LinearProgram)
        self.assertTrue(np.allclose(p.solve(), [0, 0]))

if __name__ == "__main__":
    main()
