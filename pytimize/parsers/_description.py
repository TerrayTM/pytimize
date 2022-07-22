from numpy import ndarray

_steps = {
    "1.01": "Convert to Canonical Form for Base Indices of {}",
    "1.02": "Basis:\n{}",
    "1.03": "Corresponding Coefficient Entries: {}",
    "1.04": "Basis Inverse:\n{}",
    "1.05": "y Vector:\n{}",
    "2.01": "Is {} Feasible?",
    "2.02": "{} is Feasible:",
    "2.03": "* P is in SEF",
    "2.04": "* All Entries of {} is Nonnegative",
    "2.05": "* Constraints are Satisfied (Ax = b)",
    "2.06": "{} is Not Feasible:",
    "2.07": "* Some Entries of {} are Negative",
    "2.08": "* Constraints are Not Satisfied (Ax ≠ b)",
    "2.09": "* Entry at Index {} is Negative",
    "2.10": "* Entry at Index {} is Not a Free Variable",
    "2.11": "* {} • {} = {} and {} is Not ≤ {}",
    "2.12": "* {} • {} = {} and {} is Not ≥ {}",
    "2.13": "* {} • {} = {} and {} is Not ≠ {}",
    "2.14": "* Constraints are Satisfied",
    "2.15": "* All Entries are Either Nonnegative or is a Free Variable",
    "3.01": "Convert to SEF",
    "3.02": "Take Negative of Coefficient Vector to Set Objective to Maximization",
    "3.03": "{} => {}",
    "3.04": "Free Variables at Index x",
    "4.01": "Is {} A Basic Solution",
    "4.02": "Column {} Is Not a Zero",
    "4.03": "Ax ≠ b",
    "4.04": "{} is A Basic Solution for Basis {}",
    "4.05": "{} is Not A Basic Solution for Basis {}",
    "4.06": "Ax = b",
    "4.07": "Columns of Basis are Zero",
    "5.01": "{}",
    "5.02": "Iteration {} =================================",
    "5.03": "Solution: {}",
    "5.04": "Optimal Basis: {}",
    "5.05": "The Program is Unbounded",
    "5.06": "Optimality Certificate: {}",
}

_cleanup_rules = [("[ ", "["), (" ]", "]"), ("[[", "["), ("]] ", "]"), (" [", "[")]

# TODO minor format bug
def render_descriptor(key, arguments):
    if key not in _steps:
        raise KeyError()

    for i in range(len(arguments)):
        if isinstance(arguments[i], ndarray):
            text = " ".join(filter(None, str(arguments[i]).split(" ")))

            for rule in _cleanup_rules:
                text = text.replace(rule[0], rule[1])

            arguments[i] = text

    return _steps[key].format(*arguments)


def assert_correctness(steps):
    pass
