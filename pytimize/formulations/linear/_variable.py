from . import Term

class MetaVariable(type):
    def __getitem__(cls, slice):
        return Term(slice)



class x(object, metaclass=MetaVariable):
    pass
