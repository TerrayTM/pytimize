from unittest import TestCase, main

import numpy as np

from ...linear import maximize, minimize, x
from ...linear._objective import ObjectiveFunction


class TestObjective(TestCase):
    def test_objective(self) -> None:
        objective = minimize(x[1] + x[2] + x[3] + 1)

        self.assertIsInstance(
            objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            objective.objective, "min", "Should create the correct objective type."
        )
        self.assertEqual(objective.constant, 1, "Should create correct constant.")
        self.assertTrue(
            np.allclose(objective.coefficients, [1, 1, 1]),
            "Should create correct coefficients.",
        )

        objective = maximize(2 * x[1] + 2 * x[2] + 2 * x[3] + 2)

        self.assertIsInstance(
            objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            objective.objective, "max", "Should create the correct objective type."
        )
        self.assertEqual(objective.constant, 2, "Should create correct constant.")
        self.assertTrue(
            np.allclose(objective.coefficients, [2, 2, 2]),
            "Should create correct coefficients.",
        )

    def test_maximize_zero(self) -> None:
        objective = maximize(0)

        self.assertIsInstance(
            objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            objective.objective, "max", "Should create the correct objective type."
        )
        self.assertEqual(objective.constant, 0, "Should create correct constant.")
        self.assertIsNone(objective.coefficients, "Should create correct coefficients.")

    def test_minimize_zero(self) -> None:
        objective = minimize(0)

        self.assertIsInstance(
            objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            objective.objective, "min", "Should create the correct objective type."
        )
        self.assertEqual(objective.constant, 0, "Should create correct constant.")
        self.assertIsNone(objective.coefficients, "Should create correct coefficients.")

    def test_number_objective(self) -> None:
        max_objective = maximize(5)
        min_objective = minimize(5)

        self.assertIsInstance(
            max_objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            max_objective.objective, "max", "Should create the correct objective type."
        )
        self.assertEqual(max_objective.constant, 5, "Should create correct constant.")
        self.assertIsNone(
            max_objective.coefficients, "Should create correct coefficients."
        )
        self.assertIsInstance(
            min_objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            min_objective.objective, "min", "Should create the correct objective type."
        )
        self.assertEqual(min_objective.constant, 5, "Should create correct constant.")
        self.assertIsNone(
            min_objective.coefficients, "Should create correct coefficients."
        )

        max_objective = maximize(-5)
        min_objective = minimize(-5)

        self.assertIsInstance(
            max_objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            max_objective.objective, "max", "Should create the correct objective type."
        )
        self.assertEqual(max_objective.constant, -5, "Should create correct constant.")
        self.assertIsNone(
            max_objective.coefficients, "Should create correct coefficients."
        )
        self.assertIsInstance(
            min_objective, ObjectiveFunction, "Should create objective functions."
        )
        self.assertEqual(
            min_objective.objective, "min", "Should create the correct objective type."
        )
        self.assertEqual(min_objective.constant, -5, "Should create correct constant.")
        self.assertIsNone(
            min_objective.coefficients, "Should create correct coefficients."
        )


if __name__ == "__main__":
    main()
