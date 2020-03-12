<p align="center">
  <a href="https://pytimize.terrytm.com">
    <img alt="pytimize" src="https://terrytm.com/files/pytimize.png" width="700">
  </a>
</p>
<p align="center">
  Python optimization library for mathematical programming.
</p>

Introduction
------------
Pytimize is a python library for
- Formulating and solving complex linear, integer, and nonlinear programs. 
- Performing combinatorial optimization with directed/undirected graphs and flows.
- Visualizing polyhedrons and displaying computation process.

Documentation
-------------
Coming soon!

Example
-------
The following shows a code snippet for constructing a linear program and solving
it. For more detailed examples, please see `pytimize/examples`.
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
>>> p = LinearProgram(A, b, c, z, "min", ["<=", ">="])
>>> print(p)
Min [0. 0. 4. -11. -1.]x + 17
Subject To:

[1. 0. 2.  7.  -1.]     ≤   [2.]
[0. 1. -4. -5. 3. ]x    ≥   [1.]
x ≥ 0

>>> p.solve()
array([0.    , 0.    , 0.    , 0.4375, 1.0625])
```

Contributing
------------
Pytimize is a work in progress project. Contributions are welcome on a pull request basis.

Credits
-------
Pytimize is created and maintained by Terry Zheng, Jonathan Wang, and Colin He.
Logo is designed by Kayla Estacio.
