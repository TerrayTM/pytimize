from ..linear._objective import ObjectiveFunction as LinearObjective
from ..linear._equation import LinearEquation
from typing import Union

class ObjectiveFunction(LinearObjective):
    pass



def minimize(equation: Union[LinearEquation, float]) -> ObjectiveFunction:
    if isinstance(equation, float):
        equation = LinearEquation({}, equation)

    return ObjectiveFunction(equation, "min")



def maximize(equation: Union[LinearEquation, float]) -> ObjectiveFunction:
    if isinstance(equation, float):
        equation = LinearEquation({}, equation)

    return ObjectiveFunction(equation, "max")
