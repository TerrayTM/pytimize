_steps = {
    "1.1": "Convert to Canonical Form for Basis Indices of {}",
    "1.2": "The corresponding basis of A is \n{}.",
    "1.3": "The corresponding basis of c is {}",
    "1.4": "The inverse of Ab is '
}

def render_descriptor(key, arguments):
    if not key in _steps:
        raise KeyError()

    return _steps[key].format(*arguments)
