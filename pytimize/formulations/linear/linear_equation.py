class LinearEquation:
    def __init__(self, variables):
        self._variables = variables
    
    def add_variable(self, label, coefficient):
        previous = self._variables.setdefault(label, 1)
        previous[label] = previous + coefficient

    def __add__(self, other):
        exit()

    def __radd__(self, other): 
        pass 

    def __neg__(self):
        pass

    def __sub__(self, other):
        pass 

    def __rsub__(self, other):
        pass
