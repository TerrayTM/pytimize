<p align="center">
  <a href="https://pytimize.terrytm.com">
    <img alt="pytimize" src="https://terrytm.com/files/pytimize.png" width="700">
  </a>
</p>
<p align="center">
  Python optimization library for mathematical programming.
</p>
<p align="center">
  <a href="https://pypi.org/project/pytimize/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pytimize?color=green&label=PyPI%20Package"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img alt="Apache" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Build" src="https://terrytm.com/api/wain.php?route=badge&name=pytimize">
</p>

Introduction
------------
Pytimize is a python library for
- Formulating and solving complex linear, integer, and nonlinear programs. 
- Performing combinatorial optimization with directed/undirected graphs and flows.
- Visualizing polyhedrons and displaying computation process.

Install using `pip install pytimize`!

Documentation
-------------
Coming soon!

Example
-------
The following shows a code snippet for constructing a linear program and solving
it with two phase simplex. For more detailed examples, please see `pytimize/examples`.
```python
>>> from pytimize.formulations.linear import variables, minimize
>>> a, b, c, d, e = variables(5)
>>> p = minimize(4*c - 11*d - e + 17).subject_to(
        a + 2*c + 7*d <= 2 + e,
        b - 4*c - 5*d >= 1 - 3*e
    ).where(
        a >= 0,
        b >= 0,
        c >= 0,
        d <= 0,
        e <= 0
    )
>>> p
Min [0. 0. 4. -11. -1.]x + 17.
Subject To:

[1.  0.   2.   7.  -1.]     ≤   [2.]
[0.  1.  -4.  -5.   3.]x    ≥   [1.]
x₄, x₅ ≤ 0
x₁, x₂, x₃ ≥ 0

>>> p.dual()
Max [2. 1.]x
Subject To:

[ 1.   0.]     ≤   [  0.]
[ 0.   1.]     ≤   [  0.]
[ 2.  -4.]x    ≤   [  4.]
[ 7.  -5.]     ≥   [-11.]
[-1.   3.]     ≥   [ -1.]
x₁ ≤ 0
x₂ ≥ 0

>>> p.to_sef(in_place=True)
Max [0. 0. -4. -11. -1. 0. 0.]x + 17.
Subject To:

[1.  0.   2.  -7.   1.  1.   0.]     =   [2.]
[0.  1.  -4.   5.  -3.  0.  -1.]x    =   [1.]
x ≥ 0

>>> solution, optimal_basis, certificate = p.two_phase_simplex()
>>> solution, optimal_basis, certificate
(array([2., 1., 0., 0., 0., 0., 0.]), [1, 2], array([0., 0.])
>>> p.verify_optimality(certificate)
True
>>> p.optimal_value()
17.0
```

You can also formulate the exact same program by specifying the objective 
function and constraints in matrix form:
```python
>>> from pytimize.programs import LinearProgram
>>> import numpy as np
>>> A = np.array([
      [1, 0, 2, 7, -1], 
      [0, 1, -4, -5, 3]
    ])
>>> b = np.array([2, 1])
>>> c = np.array([0, 0, 4, -11, -1])
>>> z = 17
>>> p = LinearProgram(A, b, c, z, "min", ["<=", ">="], negative_variables=[4, 5])
>>> p
Min [0. 0. 4. -11. -1.]x + 17
Subject To:

[1.  0.   2.   7.  -1.]     ≤   [2.]
[0.  1.  -4.  -5.   3.]x    ≥   [1.]
x₄, x₅ ≤ 0
x₁, x₂, x₃ ≥ 0
```

Contributing
------------
Pytimize is a work in progress project. Contributions are welcome on a pull request basis.

Credits
-------
Pytimize is created and maintained by Terry Zheng, Jonathan Wang, and Colin He.
Logo is designed by Kayla Estacio.
