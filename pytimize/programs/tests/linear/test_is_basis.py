import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

class TestIsBasis(TestCase):
    def test_is_basis(self) -> None:
        A = np.array([
            [2, 1, 2, -1, 0, 0],
            [1, 0, -1, 2, 1, 0],
            [3, 0, 3, 1, 0, 1]
        ])
        b = np.array([2, 1, 1])
        c = np.array([1, 1, 1, 1, 1, 1])
        z = 0
        
        p = LinearProgram(A, b, c, z, inequalities=["<=", "<=", "<="])
        
        self.assertTrue(p.is_basis([2, 5, 6]), "Should form a basis.")
        self.assertTrue(p.is_basis([1, 2, 3]), "Should form a basis.")
        self.assertTrue(p.is_basis([1, 5, 6]), "Should form a basis.")

        self.assertFalse(p.is_basis([1, 3, 5]), "Should not form a basis.")
        self.assertFalse(p.is_basis([1, 6]), "Should not form a basis.")
        self.assertFalse(p.is_basis([1]), "Should not form a basis.")
        self.assertFalse(p.is_basis([2, 5, -1]), "Should not form a basis.")
        self.assertFalse(p.is_basis([2, 5, 100]), "Should not form a basis.")
        self.assertFalse(p.is_basis([2, 6, 6]), "Should not form a basis.")

if __name__ == "__main__":
    main()
