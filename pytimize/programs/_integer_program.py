from . import LinearProgram

class IntegerProgram(LinearProgram):
    def __init__(self): #Constructor should take same parameters as Linear Program's 
        pass  # Also add new parameter called integer list, which is the indices of
    # the variables that has to be an integer
    # for example integers=[1,2,3] refers to x1 x2 x3 needs to be integers
    # anything else not in list can be noninteger

    #TODO implement constructor
    #TODO branch and bound method
    #TODO LP relaxation method (returns LP without integer constraint)
    #TODO cutting plane method
    #TODO override evaluate of base to take consideration of integers
    #TODO add integer constraint to __str__