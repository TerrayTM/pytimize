# optimization
Python optimization library for mathematical programming.

*This library is a work in progress.*

Created By Terry Zheng and Jonathan Wang

Currently Implemented Methods:
* Convert to Canonical Form
* Verify Infeasibility
* Verify Unboundedness
* Check Solution Feasible
* Convert to SEF
* Simplex Iteration
* Show Computation Steps
* Graph Feasible Region
* Two Phase Simplex
* Bland's Rule
* Shortest Path Solver
* Duality Theory

Planned Features:
* Integer Programming
* Branch and Bound Solver
* Cutting Plane Solver
* And More!

### Running Tests

Run All Tests

`python -m unittest discover`

Run Specific Test

`python -m unittest <module path>`

For example: `python -m unittest optimizathon.programs.tests.linear_program.test_copy`
