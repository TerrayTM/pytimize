<p align="center">
  <a href="">
    <img alt="pytimize" src="https://terrytm.com/files/images/8/image_5e29b7102548e269600364rXiWsmSY4.png" width="300">
  </a>
</p>

<p align="center">
  Python optimization library for mathematical programming.
</p>


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
* Shortest Path Linear Program
* Graph Parser
* Linear Constraint Parser
* Undirected Graph

Planned Features:
* Integer Programming
* Nonlinear Programming
* Branch and Bound Solver
* Network Flow Theory
* Min Cut Max Flow Theorem
* Cycle Detection
* Directed Graph
* Minimum Spanning Tree
* Transshipment Solver
* Min Cost Perfect Matching
* KKT Conditions
* Cutting Plane Solver
* And More!

### Running Tests

Run All Tests

`python -m unittest discover`

Run Specific Test

`python -m unittest <module path>`

For example: `python -m unittest pytimize.programs.tests.linear_program.test_copy`
