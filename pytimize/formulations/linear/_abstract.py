from ...programs import LinearProgram

class AbstractLinearProgram:
    def __init__(self, A, b, c, z):
        self._A = A 
        self._b = b
        self._c = c
        self._z = z
        self._negative_variables = []
        self._positive_variables = []



    def where(self, *variable_constraints):
        pass

    def fill(self, length):
        pass

    def compile(self):
        return LinearProgram(self._A, self._b, self._c, self._z)
