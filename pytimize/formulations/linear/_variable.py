from typing import Tuple, Union

from ._constraint import LinearConstraint, VariableConstraint
from ._equation import LinearEquation


class MetaVariable(type):
    def __getitem__(cls, key: Union[int, slice]) -> LinearEquation:
        if isinstance(key, int):
            if key <= 0:
                raise ValueError("Variable indexing must starts at 1.")

            key -= 1
        elif isinstance(key, tuple):
            # TODO implement
            pass

        return LinearEquation({key: 1})

    def __le__(self, other: int) -> VariableConstraint:
        if not other == 0:
            raise ValueError()

        return VariableConstraint(negative_variables=[])

    def __ge__(self, other: int) -> VariableConstraint:
        if not other == 0:
            raise ValueError()

        return VariableConstraint(positive_variables=[])


class Variable:
    def __init__(self, key: int):
        self._key = key

    def __mul__(self, other: float) -> LinearEquation:
        if isinstance(other, float) or isinstance(other, int):
            return LinearEquation({self._key: other})

    def __rmul__(self, other: float) -> LinearEquation:
        return self * other

    def __imul__(self, other: float) -> LinearEquation:
        return self * other

    def __pos__(self) -> LinearEquation:
        return LinearEquation({self._key: 1})

    def __neg__(self) -> LinearEquation:
        return LinearEquation({self._key: -1})

    def __add__(self, other) -> LinearEquation:
        return self.equation + other

    def __radd__(self, other) -> LinearEquation:
        return self.equation + other

    def __iadd__(self, other) -> LinearEquation:
        return self.equation + other

    def __sub__(self, other) -> LinearEquation:
        return self + -other

    def __rsub__(self, other) -> LinearEquation:
        return -self + other

    def __isub__(self, other) -> LinearEquation:
        return self + -other

    def __le__(self, other: Union[LinearEquation, float]) -> LinearConstraint:
        if isinstance(other, int) or isinstance(
            other, float
        ):  # Could be optimized TODO
            if other == 0:
                return VariableConstraint(
                    negative_variables=[self._key + 1]
                )  # index needs checking

        return self.equation <= other

    def __ge__(self, other: Union[LinearEquation, float]) -> LinearConstraint:
        if isinstance(other, int) or isinstance(other, float):
            if other == 0:
                return VariableConstraint(positive_variables=[self._key + 1])

        return self.equation >= other

    def __eq__(self, other: Union[LinearEquation, float]) -> LinearConstraint:
        return self.equation == other

    @property
    def equation(self):
        return LinearEquation({self._key: 1})

    @property
    def key(self):
        return self._key


class x(object, metaclass=MetaVariable):
    def __init__(self):
        raise NotImplementedError(
            "Incorrect usage of `x`. Please refer to the Pytimize formulation docs."
        )


def variables(count: int) -> Tuple[Variable]:
    return tuple(Variable(i) for i in range(count))
