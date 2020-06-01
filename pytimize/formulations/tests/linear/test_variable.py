from ...linear import x, variables
from ...linear._variable import Variable
from ...linear._equation import LinearEquation
from ...linear._constraint import VariableConstraint
from unittest import TestCase, main

class TestVariable(TestCase):
    def test_variable(self) -> None:
        a = x[1]
        b = x[2]
        c = x[3]

        self.assertIsInstance(a, LinearEquation, "Should create variables.")
        self.assertIsInstance(b, LinearEquation, "Should create variables.")
        self.assertIsInstance(c, LinearEquation, "Should create variables.")

    def test_variable_constraint(self) -> None:
        le = x <= 0
        ge = x >= 0

        self.assertIsInstance(le, VariableConstraint, "Should be variable constraint.") 
        self.assertIsInstance(ge, VariableConstraint, "Should be variable constraint.")
        self.assertEqual(le.positive_variables, None, "Should be all negative.")
        self.assertEqual(le.negative_variables, [], "Should be all negative.")
        self.assertEqual(ge.positive_variables, [], "Should be all positive.")
        self.assertEqual(ge.negative_variables, None, "Should be all positive.")

    def test_invalid_variable_constraint(self) -> None:
        with self.assertRaises(ValueError, msg="Should throw exception if invalid variable constraint is given."):
            x <= 5
        
        with self.assertRaises(ValueError, msg="Should throw exception if invalid variable constraint is given."):
            x >= 5

    def test_named_variables(self) -> None:
        a, b, c = variables(3)

        self.assertIsInstance(a, Variable, "Should create variables.")
        self.assertIsInstance(b, Variable, "Should create variables.")
        self.assertIsInstance(c, Variable, "Should create variables.")

    def test_variable_index_zero(self) -> None:
        with self.assertRaises(ValueError, msg="Should throw exception if index starts at 0."):
            x[0]

    def test_variable_index_negative(self) -> None:
        with self.assertRaises(ValueError, msg="Should throw exception if index is negative."):
            x[-1]

    def test_variable_init(self) -> None:
        with self.assertRaises(NotImplementedError, msg="Should throw exception if x is used incorrectly."):
            x()

if __name__ == "__main__":
    main()
