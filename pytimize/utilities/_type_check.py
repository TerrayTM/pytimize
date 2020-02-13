from typing import get_origin, get_args, _SpecialForm, Any
from inspect import signature, _empty
from functools import wraps
from collections import deque

def _validate_type(input_value: Any, required_type: Any) -> bool:
    """
    Checks if the given value is of given type.

    Parameters
    ----------
    input_value : Any
        The input value that needs to be checked.

    required_type : Any
        The expected type of the input value.

    Returns
    -------
    result : bool
        Whether or not the value is of the expected type.

    """
    queue = deque([(input_value, required_type)])

    while len(queue) > 0:
        value, expected_type = queue.pop()

        if expected_type == _empty:
            raise NotImplementedError("Type annotation is missing for parameter item.")
        
        if expected_type in [int, str, float, bool]:
            if not isinstance(value, expected_type):
                return expected_type == float and isinstance(value, int)
        else:
            origin = get_origin(expected_type)
            
            if isinstance(origin, _SpecialForm):
                pass #TODO

            if not isinstance(value, origin):
                return False

            argument_types = get_args(expected_type)

            if origin == list or origin == set:
                queue.extend((item, argument_types[0]) for item in value)
            elif origin == tuple:
                if not len(argument_types) == len(value):
                    return False

                queue.extend(zip(value, argument_types))
            else:
                raise NotImplementedError(f"Type check for `{origin}` is not implemented.")

    return True



def typecheck(method: Any) -> Any:
    """
    Decorator for type checking function parameters. Throws an error if
    any input parameters does not match its type annotation.

    Parameters
    ----------
    method : Any
        The method that the decorator wraps on.

    Returns
    -------
    result : Any
        A decorated function. 

    """
    @wraps(method)
    def wrapped_method(*args, **kwargs) -> Any:
        parameter_info = signature(method).parameters

        kwargs.update(dict(zip(parameter_info.keys(), args)))

        for variable, value in kwargs.items():
            expected_type = parameter_info[variable].annotation

            if not _validate_type(value, expected_type):
                raise TypeError(f"Parameter `{variable}` expects to have type `{expected_type}`.")
        
        return method(**kwargs)

    return wrapped_method
