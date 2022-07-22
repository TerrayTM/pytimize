from typing import Callable, List, Union

import numpy as np

Vector = Union[np.ndarray, List[float]]
Matrix = Union[np.ndarray, List[List[float]]]


class UnconstrainedProgram:
    def __init__(self, f: Callable[[Vector], float], objective: str = "min"):
        """
        Constructs an unconstrained program of the form `<objective> <function>`
        where objective denotes whether this is a maximization or minimization problem, and function
        denotes the target of the program.

        Parameters
        ----------
        f : Callable[[Vector], float]
            The function to optimize for.

        objective : str (default="min")
            The objective of the program. Must be either `max` or `min`.

        """
        if objective == "max":
            self._f = lambda x: -f(x)
        self._f = f
        self._minimize = objective == "min"

    def steepest_descent(
        self,
        x: Vector,
        df: Callable[[Vector], Vector],
        alpha: float = 1,
        c: float = 0.7,
        p: float = 0.5,
        tolerance: float = 10e-5,
        max_iteration: int = 10e4,
    ):
        """
        Computes the minimizer or maximizer of the function using steepest descent method
        with backtracking.

        Parameters
        ----------
        x : Vector
            The starting point to begin steepest descent.

        df : Callable[[Vector], Vector]
            The gradient function of the target function.

        alpha : float (default=1)
            The initial step length multiplier.

        c : float (default=0.7)
            Parameter controlling exit condition of backtrack.

        p : float (default=0.5)
            Multiplier of length reduction per backtrack iteration.

        tolerance : float (default=10e-5)
            The accuracy tolerance of the final answer.

        max_iteration : int (default=10e4)
            The maximum number of iterations.

        """
        for _ in range(int(max_iteration)):
            d = -df(x)
            a = alpha

            while self._f(x + a * d) > self._f(x) + c * a * df(x) @ d:
                a *= p

            x += a * d

            if np.linalg.norm(df(x)) <= tolerance:
                break

        return x if self._minimize else -x

    def newton_method(
        self,
        x: Vector,
        df: Callable[[Vector], float],
        d2f: Callable[[Vector], Matrix],
        tolerance: float = 10e-5,
        max_iteration: int = 10e4,
    ):
        """
        Computes the minimizer or maximizer of the function using Newton's method.

        Parameters
        ----------
        x : Vector
            The starting point to begin Newton's method.

        df : Callable[[Vector], Vector]
            The gradient function of the target function.

        d2f : Callable[[Vector], Matrix]
            The Hessian function of the target function.

        tolerance : float (default=10e-5)
            The accuracy tolerance of the final answer.

        max_iteration : int (default=10e4)
            The maximum number of iterations.

        """

        for _ in range(int(max_iteration)):
            d = np.linalg.solve(d2f(x), -df(x))
            x += d

            if np.linalg.norm(df(x)) <= tolerance:
                break

        return x if self._minimize else -x
