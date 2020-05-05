import math
import random
import numpy as np

from ..programs._linear import LinearProgram
from ..programs._integer import IntegerProgram
from typing import List, Tuple, Dict, Set, Optional, Iterator, Union, Callable, Any
from collections import deque

class UndirectedGraph:
  def __init__(self, graph: Optional[Dict[str, Set[str]]]=None, edges: Optional[Dict[Tuple[str, str], float]]=None, vertices: Optional[Dict[str, float]]=None) -> None:
    """
    Constructs a undirected graph with support for edge and vertex weights.
    If `graph`, `edges`, `vertices`, or a combination of them is provided, the 
    graph will be built respectively with any nonspecified weights set to 0.

    Parameters
    ----------
    graph : Optional[Dict[str, Set[str]]] (default=None)
      The graph dictionary where key is the vertex and value is the set of 
      vertices that are connected to that vertex.

    edges : Optional[Dict[Tuple[str, str], float]] (default=None)
      The edges dictionary where key is the edge and value is the weight. 

    vertices : Optional[Dict[str, float]] (default=None)
      The vertices dictionary where key is the vertex and value is the weight.

    """
    self._graph = {}
    self._edges = {}
    self._vertices = {}

    if graph is not None:
      for vertex, connections in graph.items():
        if len(connections) == 0:
          if not self.has_vertex(vertex):
            self.add_vertex(vertex)
        else:
          for connection in connections:
            edge = vertex, connection

            if self.has_edge(edge):
              continue

            self.add_edge(edge)

    if edges is not None: 
      for edge, weight in edges.items():
        if not self.has_edge(edge):
          self.add_edge(edge)

        self.set_edge_weight(edge, weight)

    if vertices is not None:
      for vertex, weight in vertices.items():
        if not self.has_vertex(vertex): 
          self.add_vertex(vertex)

        self.set_vertex_weight(vertex, weight)



  def __repr__(self) -> str: # TODO use table
    """
    Generates a string representation of the graph.

    Returns
    -------
    result : str
        A string representation of the graph.

    """
    adjacency = set()

    for vertex, connections in self._graph.items():
      formatted = '-'.join(sorted(connections))

      adjacency.add(f"{vertex}: {formatted}")

    return "\n".join(sorted(adjacency))



  def __eq__(self, other: "UndirectedGraph") -> bool:
    """
    Compares if two graphs are the same.

    Returns
    -------
    result : bool
        Whether or not the other is the same graph. 

    """
    return other._graph == self._graph and other._edges == self._edges and other._vertices == self._vertices 



  def add_edge(self, edge: Tuple[str, str], weight: float=0) -> None:
    """
    Adds an edge to the graph. Any newly created vertices will have a weight of 0.

    Parameters
    ----------
    edge : Tuple[str, str]
      The identifier of the edge.

    weight : float (default=0)
      The weight of the edge.

    """
    if weight < 0:
      raise ValueError("Weight cannot be negative.")

    if edge[0] == edge[1]:
      raise ValueError("Cannot form an edge with the same vertex.")

    if not edge[0] or not edge[1]:
      raise ValueError("Vertices cannot be an empty string.")
    
    edge = tuple(sorted(edge))

    if self.has_edge(edge):
      raise ValueError("The given edge is already in graph.")

    self._graph.setdefault(edge[0], set()).add(edge[1])
    self._graph.setdefault(edge[1], set()).add(edge[0])
    self._edges.setdefault(edge, weight)
    self._vertices.setdefault(edge[0], 0)
    self._vertices.setdefault(edge[1], 0)



  def remove_edge(self, edge: Tuple[str, str]) -> None:
    if not self.has_edge(edge):
      raise ValueError("The given edge is not in graph.")

    self._graph[edge[0]].remove(edge[1])
    self._graph[edge[1]].remove(edge[0])



  def has_edge(self, edge: Tuple[str, str]) -> bool:
    """
    Checks if given edge is in graph.

    Parameters
    ----------
    edge : Tuple[str, str]
      The identifier of the edge.

    Returns
    -------
    result : bool
        Whether or not the given edge is in graph.

    """
    return tuple(sorted(edge)) in self._edges



  def add_vertex(self, vertex: str, weight: float=0) -> None:
    """
    Adds a vertex to the graph.

    Parameters
    ----------
    vertex : str
      The identifier of the vertex.

    weight : float (default=0)
      The weight of the vertex.

    """
    if weight < 0:
      raise ValueError("Weight cannot be negative.")

    if self.has_vertex(vertex):
      raise ValueError("The given vertex is already in graph.")

    self._graph.setdefault(vertex, set())
    self._vertices.setdefault(vertex, weight)



  def remove_vertex(self):
    pass



  def has_vertex(self, vertex: str) -> bool:
    """
    Checks if given vertex is in graph.

    Parameters
    ----------
    vertex : str
      The identifier of the vertex.

    Returns
    -------
    result : bool
        Whether or not the given vertex is in graph.

    """
    return vertex in self._vertices



  def set_edge_weight(self, edge: Tuple[str, str], weight: float) -> None:
    if not self.has_edge(edge):
      raise ValueError("The given edge is not in graph.")

    self._edges[tuple(sorted(edge))] = weight



  def set_vertex_weight(self, vertex: str, weight: float) -> None:
    if not self.has_vertex(vertex):
      raise ValueError("The given vertex is not in graph.")

    self._vertices[vertex] = weight



  def get_edge_weight(self, edge: Tuple[str, str]) -> float:
    """
    Gets the weight of an edge.

    Parameters
    ----------
    edge : Tuple[str, str]
      The identifier of the edge.

    Returns
    -------
    result : float
      The weight of the edge.

    """
    if not self.has_edge(edge):
      raise ValueError("The given edge is not in graph.")

    return self._edges[tuple(sorted(edge))]



  def get_vertex_weight(self, vertex: str) -> float:
    """
    Gets the weight of a vertex.

    Parameters
    ----------
    vertex : str
      The identifier of the vertex.

    Returns
    -------
    result : float
      The weight of the vertex.

    """
    if not self.has_vertex(vertex):
      raise ValueError("The given vertex is not in graph.")

    return self._vertices[vertex]



  def shortest_path(self, start: str, end: str) -> List[Tuple[str, str]]:
    if start not in self._graph:
      raise ValueError("Starting vertex is not in graph.")

    if end not in self._graph:
      raise ValueError("Ending vertex is not in graph.")

    potential = {}
    visited = {start}
    directed_graph = {}

    while end not in visited:
      best_slack = math.inf
      best_edge = None
      edges = set()

      for vertex in visited:
        for edge in self._graph[vertex]:
          if edge[0] not in visited:
            edge_hash = self._hash_edge((edge[0], vertex))
            slack = edge[1] - (potential[edge_hash] if edge_hash in potential else 0)

            #print(edge_hash, slack)

            edges.add(edge_hash)

            if slack < best_slack:
              best_slack = slack
              best_edge = vertex, edge[0], edge_hash
      
            #print(visited, best_slack)

      for edge_hash in edges:
        previous = potential.setdefault(edge_hash, 0)
        potential[edge_hash] = previous + best_slack
      
      visited.add(best_edge[1])
      directed_graph.setdefault(best_edge[1], []).append(best_edge[0])

    current = end
    path = []
    
    while not current == start:
      previous = current
      current = directed_graph[current][0]

      path.insert(0, (current, previous))

    return path



  def formulate_shortest_path(self, start: str, end: str, disjoint_cuts=True) -> "LinearProgram":
    if start not in self._graph:
      raise ValueError("Starting vertex is not in graph.")

    if end not in self._graph:
      raise ValueError("Ending vertex is not in graph.")

    vertices = self.vertices

    vertices.remove(start)
    vertices.remove(end)

    power_set = []
    # TODO: BFS Disjoint st-cuts
    for i in range(2 ** len(vertices)):
      current_set = set()

      for j in range(len(vertices)):
        if i & (1 << j):
          current_set.add(vertices[j])

      current_set.add(start)
      power_set.append(current_set)

    graph_edges = self.edges
    A = {self._hash_edge((edge[0], edge[1])): np.zeros(len(power_set)) for edge in graph_edges}

    for i, group in enumerate(power_set):
      edges = set()

      for vertex in group:
        edges = edges.union(map(lambda x: self._hash_edge((vertex, x[0])), self._graph[vertex]))

      for edge in filter(lambda x: x[0] not in group or x[1] not in group, edges):
        A[edge][i] = 1

    ordered_A = []
    ordered_edges = [] #ordered edges must point back to actual edges not hash TODO

    for edge, coefficients in A.items():
      ordered_A.append(coefficients)
      ordered_edges.append(edge)

    result_A = np.vstack(ordered_A).T
    edge_positions = {edge: i for i, edge in enumerate(ordered_edges)}
    result_c = np.zeros(len(ordered_edges))
    constraints = result_A.shape[0]

    for edge in graph_edges:
      result_c[edge_positions[self._hash_edge((edge[0], edge[1]))]] = edge[2]

    return LinearProgram(result_A, np.ones(constraints), result_c, 0, "min", [">="] * constraints)



  def delta(self, vertices: Union[Set[str], str]) -> Set[str]:
    """
    Gets a set of edges where every edge has exactly one endpoint in given `vertices`.

    Parameters
    ----------
    vertices : Union[Set[str], str]
      The vertex or vertices of the operation.

    Returns
    -------
    result : Set[str]
      The set of edges that satisfies the above condition.

    """
    if not isinstance(vertices, set):
      vertices = set(vertices)

    return set(filter(lambda x: (x[0] in vertices and x[1] not in vertices) or \
      (x[1] in vertices and x[0] not in vertices), self._edges.keys()))



  def formulate_max_stable_set(self) -> Tuple[IntegerProgram, Dict[str, int]]:
    if self.is_empty():
      return # TODO Return max 0 unconstrained

    mapping = {vertex: i for i, vertex in enumerate(self._vertices)}
    rows = len(self._edges)
    columns = len(mapping)
    A = np.empty(columns)

    for a, b in self._edges:
      row = np.zeros(columns)
      row[mapping[a]] = 1
      row[mapping[b]] = 1
      A = np.vstack((A, row))

    A = A[1:]
    b = np.ones(rows)
    c = np.zeros(columns)
    inequalities = ["<="] * (rows + columns) # Includes x <= 1 columns

    for vertex, weight in self._vertices.items():
      c[mapping[vertex]] = weight

    # TODO this could be optimized by setting x <= 1 all at once
    for i in range(columns):
      row = np.zeros(columns)
      row[i] = 1
      A = np.vstack((A, row))
      b = np.hstack((b, 1))

    return IntegerProgram(A, b, c, 0, inequalities=inequalities), mapping



  def dfs(self, start: str) -> List[str]:
    return list(self.walk(start, "dfs"))



  def bfs(self, start: str) -> List[str]:
    return list(self.walk(start, "bfs"))



  def walk(self, start: str, traversal: str="bfs", order: Optional[Callable[[str], Any]]=None) -> Iterator[str]:
    """
    Walks through the graph and returns its vertices. Traversal can be breath first,
    depth first, or random search. If the graph is not connected `bfs` and `dfs` may not
    return all the vertices. 

    Parameters
    ----------
    start : str
      The starting vertex.

    traversal : str (default="bfs")
      The type of graph traversal. Options are `bfs`, `dfs`, or `rng`.

    order : Optional[Callable[[str], Any]] (default=None)
      The sorting order of vertices to output. This callable takes a tuple
      of previous vertex and current vertex. If given none, lexical order based
      on current vertex is used.

    Returns
    -------
    result : Iterator[str]
      The iterator giving the current vertex of the walk.

    """
    # TODO Add dfs
    if not self.has_vertex(start):
      raise ValueError("The starting vertex is not in graph.")

    if traversal == "rng":
      vertices = list(self._vertices.keys())

      random.shuffle(vertices)

      return iter(vertices)

    visited = set()
    queue = deque([start])
    
    while len(queue) > 0:
      current = queue.popleft()

      if current in visited:
        continue

      visited.add(current)
      queue.extend(sorted(self._graph[current], key=order))

      yield current



  def is_connected(self) -> bool:
    """
    Checks if the graph is connected.

    Returns
    -------
    result : bool
      Whether or not the graph is connected.

    """
    if self.is_empty():
      return False

    return len(self.bfs(next(iter(self._vertices)))) == len(self._vertices)



  def is_cyclic(self) -> bool:
    """
    Checks if the graph is cyclic.

    Returns
    -------
    result : bool
      Whether or not the graph is cyclic.

    """
    if self.is_empty():
      return False
    
    unexplored = set(self._vertices.keys())

    while len(unexplored) > 0:
      queue = deque([next(iter(unexplored))])
      visited = set()

      while len(queue) > 0: 
        current = queue.popleft()

        if current in visited:
          return True

        visited.add(current)
        queue.extend(self._graph[current])

      unexplored = unexplored.difference(visited)

    return False



  def copy(self) -> "UndirectedGraph":
    """
    Creates a deep copy of the graph.

    Returns
    -------
    result : UndirectedGraph
        The copy of the graph.

    """
    g = UndirectedGraph() 

    g._graph = self.graph
    g._vertices = self.vertices
    g._edges = self.edges

    return g



  def partitions(self) -> List["UndirectedGraph"]:
    pass



  def is_empty(self): 
    return len(self._vertices) == 0



  def degree(self, vertices: Union[Set[str], str]) -> int: 
    """
    Computes the degree of a vertex or vertices. Degree is defined as the number of
    edges that has exactly one endpoint inside `vertices`.

    Parameters
    ----------
    vertices : Union[Set[str], str]
      The vertex or vertices of the operation.

    Returns
    -------
    result : int
      The degree of the vertices.

    """
    return len(self.delta(vertices))



  def has_spanning_tree(self) -> bool:
    return self.is_connected()



  def minimum_spanning_tree(self) -> bool:
    pass



  def is_tree(self) -> bool:
    """
    Checks if the graph is a tree. By definition, a graph is a tree if it is
    connected and it has no cycles.

    Returns
    -------
    result : bool
      Whether or not the graph is cyclic.

    """
    if self.is_empty():
      return False

    queue = deque([next(iter(self._vertices))])
    visited = set()

    while len(queue) > 0:
      current = queue.popleft() 

      if current in visited:
        return False

      visited.add(current)
      queue.extend(set(self._graph[current]).difference(visited))

    return len(visited) == len(self._vertices)



  def is_binary_tree(self) -> bool:
    pass



  def _hash_edge(self, edge: Tuple[str, str]) -> str:
    a, b = edge
    
    if a > b:
      a, b = b, a
    
    return f"{a}{b}"



  @property
  def edges(self) -> Dict[Tuple[str, str], float]:
    """
    Gets the edges of the graph.

    Returns
    -------
    result : Dict[Tuple[str, str], float]
      The edges dictionary where key is the edge and value is the weight. 

    """
    return self._edges.copy()



  @property
  def graph(self) -> Dict[str, Set[str]]:
    """
    Gets the graph representation.

    Returns
    -------
    result : Dict[str, Set[str]]
      The graph dictionary where key is the vertex and value is the set of 
      vertices that are connected to that vertex.

    """
    return self._graph.copy()



  @property
  def vertices(self) -> Dict[str, float]:
    """
    Gets the vertices of the graph.

    Returns
    -------
    result : Dict[str, float]
      The vertices dictionary where key is the vertex and value is the weight.

    """
    return self._vertices.copy()
