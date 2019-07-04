import sys
import numpy as np

sys.path.append('../optimization')

from main import LinearProgrammingProblem

def test_simple():
    A = np.array([
        [1, 1, 1, 0, 0],
        [2, 1, 0, 1, 0],
        [-1, 1, 0, 0, 1]
    ])
    b = np.array([6, 10, 4])
    c = np.array([2, 3, 0, 0, 0])
    z = 0

    expected_A = np.array([
        [1, 0, 0.5, 0, -0.5],
        [0, 1, 0.5, 0, 0.5],
        [0, 0, -1.5, 1, 0.5]
    ])
    expected_b = np.array([1, 5, 3])
    expected_c = np.array([0, 0, -2.5, 0, -0.5])
    expected_z = 17

    basis = [1, 2, 4]

    p = LinearProgrammingProblem(A, b, c, z)
    p.to_canonical_form(basis)
    
    assert np.array_equal(p.A, expected_A), "Should compute correct coefficient matrix."
    assert np.array_equal(p.b, expected_b), "Should compute correct constraints."
    assert np.array_equal(p.c, expected_c), "Should compute correct coefficient vector."
    assert p.z == expected_z, "Should compute correct constant."

def run_tests():
    tests = [
        test_simple
    ]

    for test in tests:
        test()

    print(f"{__file__}: {len(tests)} test{('', 's')[len(tests) > 1]} passed!")

if __name__ == "__main__":
    run_tests()