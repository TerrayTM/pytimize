from unittest import TestCase, main

import numpy as np

from ... import LinearParser


class TestParse(TestCase):
    def test_parse(self) -> None:
        result = LinearParser.parse("1.25x_5+0.25x_4-3x_8")
        expected = np.array([0, 0, 0, 0.25, 1.25, 0, 0, -3])

        self.assertTrue(np.allclose(result, expected), "Should parse correctly.")


if __name__ == "__main__":
    main()
