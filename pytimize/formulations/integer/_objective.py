from typing import Union

from ..linear._equation import LinearEquation
from ..linear._objective import ObjectiveFunction as LinearObjective


class ObjectiveFunction(LinearObjective):
    pass


def minimize(equation: Union[LinearEquation, float]) -> ObjectiveFunction:
    if isinstance(equation, float) or isinstance(equation, int):
        equation = LinearEquation({}, equation)

    return ObjectiveFunction(equation, "min")


def maximize(equation: Union[LinearEquation, float]) -> ObjectiveFunction:
    if isinstance(equation, float) or isinstance(equation, int):
        equation = LinearEquation({}, equation)

    return ObjectiveFunction(equation, "max")
