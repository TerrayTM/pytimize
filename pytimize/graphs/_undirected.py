import math
import numpy as np 

from ..programs import LinearProgram, IntegerProgram
from typing import List, Tuple, Dict, Set

class UndirectedGraph: #TODO validation 
  def __init__(self, edges: List[Tuple[str, str, float]]=None) -> None:
    self._graph = {}

    if edges is not None:
      if not isinstance(edges, list):
        raise TypeError("Edges must be a list.")

      for edge in edges:
        if not isinstance(edge, tuple):
          raise TypeError("Each entry of edges must be a tuple.")

        if not len(edge) == 3:
          raise TypeError("Each tuple must have 3 items.")

        if not isinstance(edge[0], str) or not isinstance(edge[1], str):
          raise TypeError("First two entries of tuple must be strings.")

        self.add_edge((edge[0], edge[1]), edge[2])



  def add_edge(self, edge: Tuple[str, str], weight: float=0) -> None:
    if edge[0] in self._graph and any(node[0] == edge[1] for node in self._graph[edge[0]]):
      raise ValueError("The given edge is already in graph.")

    if weight < 0:
      raise ValueError("Weight cannot be negative.")

    if edge[0] == edge[1]:
      raise ValueError("Cannot form an edge with the same vertex.")

    if not edge[0] or not edge[1]:
      raise ValueError("Vertices cannot be empty string.")

    if not isinstance(edge[0], str) or not isinstance(edge[1], str):
      raise ValueError("Vertices must be of type string.")

    self._graph.setdefault(edge[0], set()).add((edge[1], weight))
    self._graph.setdefault(edge[1], set()).add((edge[0], weight))



  def remove_edge(self, edge: Tuple[str, str]) -> None:
    if not self.has_edge(edge):
      raise ValueError("The given edge is not in graph.")

    self._graph[edge[0]].remove(edge[1])
    self._graph[edge[1]].remove(edge[0])



  def has_edge(self, edge: Tuple[str, str]) -> bool:
    return edge[0] in self._graph and edge[1] in self._graph[edge[0]]



  def has_vertex(self, vertex: str) -> bool:
    return vertex in self._graph



  def get_weight(self, edge: Tuple[str, str]) -> float:
    if not self.has_edge(edge):
      raise ValueError("The given edge is not in graph.")

    return self._graph[edge[0]][edge][1]



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
            edge_hash = self.__hash_edge((edge[0], vertex))
            slack = edge[1] - (potential[edge_hash] if edge_hash in potential else 0)

            #print(edge_hash, slack)

            edges.add(edge_hash)

            if slack < best_slack:
              best_slack = slack
              best_edge = (vertex, edge[0], edge_hash)
      
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



  def formulate_shortest_path(self, start: str, end: str) -> "LinearProgram":
    if start not in self._graph:
      raise ValueError("Starting vertex is not in graph.")

    if end not in self._graph:
      raise ValueError("Ending vertex is not in graph.")

    vertices = self.vertices

    vertices.remove(start)
    vertices.remove(end)

    power_set = []

    for i in range(2 ** len(vertices)):
      current_set = set()

      for j in range(len(vertices)):
        if i & (1 << j):
          current_set.add(vertices[j])

      current_set.add(start)
      power_set.append(current_set)

    graph_edges = self.__get_edges()
    A = { self.__hash_edge((edge[0], edge[1])): np.zeros(len(power_set)) for edge in graph_edges }

    for i, group in enumerate(power_set):
      edges = set()

      for vertex in group:
        edges = edges.union(map(lambda x: self.__hash_edge((vertex, x[0])), self._graph[vertex]))

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
      result_c[edge_positions[self.__hash_edge((edge[0], edge[1]))]] = edge[2]

    return LinearProgram(result_A, np.ones(constraints), result_c, 0, "min", [">=" for i in range(constraints)])


  def delta(self, vertices):
    connected = []

    for inner in vertices:
      for outer in self._graph[inner]:
        if outer not in vertices:
          connected.append(outer[0])

    return connected



  def formulate_max_stable_set(self) -> Tuple[IntegerProgram, List[str]]:
    mapping_A = { vertex: i for i, vertex in enumerate(self._graph.keys()) }
    stack_A = []
    seen = set()

    for vertex in self._graph.keys():
      row = np.zeros(len(mapping_A))
      row[mapping_A[vertex]] = 1
      
      for connected in self.delta({ vertex }): # TODO Should be one item or set
        row[mapping_A[connected]] = 1
      
      hash = " ".join(str(i) for i in row)
      
      if not hash in seen:
        seen.add(hash)
        stack_A.append(row)

    result_A = np.array(stack_A)
    result_b = np.ones(result_A.shape[0])
    result_c = np.ones(result_A.shape[1]) #TODO implemented node weights
    inequalities = ["<="] * result_A.shape[0]

    return IntegerProgram(result_A, result_b, result_c, 0, inequalities=inequalities), mapping_A



  def __hash_edge(self, edge: Tuple[str, str]) -> str:
    a, b = edge[0], edge[1]
    
    if a < b:
      a, b = b, a
    
    return f"{a}{b}"



  def __get_edges(self) -> List[Tuple[str, str, float]]:
    result = set()

    for vertex, connections in self._graph.items():
      for connection in connections:
        a = vertex
        b = connection[0]

        if a < b:
          a, b = b, a
        
        result.add((a, b, connection[1]))
    
    return result



  @property
  def edges(self) -> Set[Tuple[str, str, float]]:
    """
    Gets the edges of the graph.

    Returns
    -------
    result : list of tuples (str, str, float)

    """
    return self.__get_edges()



  @property
  def graph(self) -> Dict[str, Set[Tuple[str, float]]]:
    """
    Gets the graph representation as an adjacency list.

    Returns
    -------
    result : Dict[str, List[Set[str, float]]]

    """
    return self._graph



  @property
  def vertices(self) -> List[str]:
    """
    Gets the vertices of the graph.

    Returns
    -------
    result : list of str

    """
    return list(self._graph.keys())
