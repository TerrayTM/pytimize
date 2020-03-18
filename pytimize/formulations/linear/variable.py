from . import LinearConstraint, LinearEquation

class Variable:
    def __init__(self, label):
        self._label = label
        self._coefficient = 1 

    def __add__(self, other):
        if isinstance(other, Variable):
            if other.label == self._label:
                self._coefficient += other.coefficient
                return self
            
            return LinearEquation({
                self._label: self._coefficient,
                other.label: other.coefficient
            })
        elif isinstance(other, LinearEquation):
            return other.add_variable(self._label, self._coefficient)
        elif isinstance(other, LinearConstraint):
            return other.add_variable(self._label, self._coefficient)
        raise Exception()

    def __mul__(self, other): 
        if isinstance(other, float) or isinstance(other, int): 
            self._coefficient *= other
            return self

    def __rmul__(self, other):
        return self * other 

    @property
    def label(self):
        return self._label

    @property
    def coefficient(self):
        return self._coefficient

class MetaVariable(type):
    def __getitem__(cls, slice):
        return Variable(slice)

class x(object, metaclass=MetaVariable):
    pass
