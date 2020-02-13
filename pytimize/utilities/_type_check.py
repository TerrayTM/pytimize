from typing import get_origin, get_args, _SpecialForm, Any, Union
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
    queue = deque([(input_value, required_type, None)])
    union_types = set()
    union_id = 0

    while len(queue) > 0:
        value, expected_type, ids = queue.pop()

        if expected_type == _empty:
            raise NotImplementedError("Type annotation is missing for parameter item.")
        
        if expected_type in [int, str, float, bool]:
            if not isinstance(value, expected_type):
                if ids is None:
                    return expected_type == float and isinstance(value, int)
            elif ids is not None:
                union_types.update(ids)
        else:
            origin = get_origin(expected_type)

            if isinstance(origin, _SpecialForm):
                if origin == Union:
                    base_copy = [union_id]

                    if ids is not None:
                        base_copy += ids.copy()

                    queue.extend((value, allowed_type, base_copy.copy()) for allowed_type in get_args(expected_type))
                    
                    union_id +=1
                    
                    continue
                else:
                    raise NotImplementedError(f"Type check for special form `{origin}` is not supported.")

            if origin is None: 
                if ids is not None:
                    union_types.update(ids)
        
                continue

            if not isinstance(value, origin):
                if ids is None:
                    return False
                else:
                    continue

            argument_types = get_args(expected_type)

            if origin == list or origin == set:
                queue.extend((item, argument_types[0], None if ids is None else ids.copy()) for item in value)
            elif origin == tuple:
                argument_count = len(argument_types)

                if argument_count == 0:
                    raise NotImplementedError(f"Type annotation for collection type `{origin}` is incomplete.")

                if not argument_count == len(value):
                    if ids is None:
                        return False
                    else:
                        continue

                copy = [None if ids is None else ids.copy() for i in range(argument_count)]

                queue.extend(zip(value, argument_types, copy))
            else:
                raise NotImplementedError(f"Type check for `{origin}` is not implemented.")

    return all(i in union_types for i in range(union_id))



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
