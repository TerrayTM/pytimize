class TypeCheck:
    @staticmethod
    def expect(variable, types, parameter) -> None:
        if not isinstance(variable, type_name):
            raise ValueError(f"The parameter '{parameter}' needs to be of type '{type_name.__name__}'.")
