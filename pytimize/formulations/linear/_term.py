from ._equation import LinearEquation
from ._constraint import LinearConstraint

class Term:
    def __init__(self, label: int):
        self._label = label
        self._coefficient = 1



    def __neg__(self):
        self._coefficient = -self._coefficient
        
        return self



    def __pos__(self):
        return self



    def __add__(self, other):
        if isinstance(other, Term):
            if other.label == self._label:
                self._coefficient += other.coefficient
                return self
            
            return LinearEquation({ 
                self._label: self._coefficient,
                other.label: other.coefficient
            })
        elif isinstance(other, LinearEquation):
            return other.add_variable(self._label, self._coefficient)
        elif isinstance(other, int) or isinstance(other, float): 
            return LinearEquation({ self._label: self._coefficient }, other)
        raise Exception()



    def __radd__(self, other):
        pass



    def __iadd__(self, other):
        pass



    def __mul__(self, other): 
        if isinstance(other, float) or isinstance(other, int): 
            self._coefficient *= other
            return self



    def __rmul__(self, other):
        return self * other 



    def __imult__(self, other):
        return self * other


    def __le__(self, other):
        pass 



    def __ge__(self, other):
        pass 



    def __eq__(self, other):
        pass



    def to_equation(self):
        return LinearEquation({ self._label: self._coefficient })



    @property
    def label(self):
        return self._label



    @property
    def coefficient(self):
        return self._coefficient
