import math
import numpy as np 

from ..programs._linear import LinearProgram
from ..programs._integer import IntegerProgram
from typing import List, Tuple, Dict, Set, Optional, Iterator, Union

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
    visited = { start }
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

    graph_edges = self.__get_edges()
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
    edge_positions = { edge: i for i, edge in enumerate(ordered_edges) }
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
    if len(self._graph) == 0:
      return None #TODO this can't be none

    mapping_A = { vertex: i for i, vertex in enumerate(self._graph.keys()) }
    columns = len(mapping_A)
    stack_A = np.empty(columns)
    seen = set()

    for vertex in self._graph.keys():
      row = np.zeros(columns)
      row[mapping_A[vertex]] = 1
      
      for connected in self.delta(vertex): # TODO Should be one item or set
        row[mapping_A[connected]] = 1
      
      hash = " ".join(str(i) for i in row)
      
      if not hash in seen:
        seen.add(hash)

        stack_A = np.vstack((stack_A, row))

    result_A = stack_A[1:]
    result_b = np.ones(result_A.shape[0])
    result_c = np.ones(result_A.shape[1]) #TODO implemented node weights
    inequalities = ["<="] * result_A.shape[0]

    return IntegerProgram(result_A, result_b, result_c, 0, inequalities=inequalities), mapping_A



  # TODO: note if graph is segmented then dfs and bfs shouldn't work
  def dfs(self, start: str) -> List[str]:
    return list(self.walk_dfs)



  def bfs(self, start: str) -> List[str]:
    return list(self.walk_bfs)


  
  def walk_dfs(self, start: str) -> Iterator[Tuple[Tuple[Tuple[str, str], float], Tuple[str, float]]]:
    """
    """
    pass



  def walk_bfs(self, start: str) -> Iterator[Tuple[Tuple[Tuple[str, str], float], Tuple[str, float]]]:
    """
    """
    pass



  def is_connected(self) -> bool:
    pass



  def has_cycle(self) -> bool:
    pass 



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



  def is_tree(self) -> bool:
    pass



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
