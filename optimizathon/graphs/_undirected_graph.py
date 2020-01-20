import math

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
    if edge[0] in self._graph.keys() and edge[1] in self._graph[edge[0]]:
      raise ValueError("The given edge is already in graph.")

    if weight < 0:
      raise ValueError("Weight cannot be negative.")

    if edge[0] == edge[1]:
      raise ValueError("Cannot form an edge with the same vertex.")

    if not edge[0] or not edge[1]:
      raise ValueError("Edge name cannot be empty string.") 

    self._graph.setdefault(edge[0], set()).add((edge[1], weight))
    self._graph.setdefault(edge[1], set()).add((edge[0], weight))



  def remove_edge(self, edge: Tuple[str, str]) -> None: #TODO change ALL not ... in to ... not in
    if not self.has_edge(edge):
      raise ValueError("The given edge is not in graph.")

    self._graph[edge[0]].remove(edge[1])
    self._graph[edge[1]].remove(edge[0])



  def has_edge(self, edge: Tuple[str, str]) -> bool:
    return edge[0] in self._graph.keys() and edge[1] in self._graph[edge[0]]



  def has_vertex(self, vertex: str) -> bool:
    return vertex in self._graph.keys()



  def get_weight(self, edge: Tuple[str, str]) -> float:
    if not self.has_edge(edge):
      raise ValueError("The given edge is not in graph.")

    return self._graph[edge[0]][edge][1]



  def shortest_path(self, start: str, end: str) -> List[Tuple[str, str]]:
    if start not in self._graph.keys():
      raise ValueError("Starting vertex is not in graph.")

    if end not in self._graph.keys():
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
            edge_hash = [edge[0], vertex]

            edge_hash.sort()
            
            edge_hash = "".join(edge_hash)
            slack = edge[1] - (potential[edge_hash] if edge_hash in potential.keys() else 0)

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



  def create_shortest_path_program(self, start: str, end: str) -> "LinearProgram":
    if start not in self._graph.keys():
      raise ValueError("Starting vertex is not in graph.")

    if end not in self._graph.keys():
      raise ValueError("Ending vertex is not in graph.")

    vertices = self.vertices

    vertices.remove(start)
    vertices.remove(end)

    power_set = set()

    for i in 2 ** range(len(vertices)):
      current_set = set()

      for j in range(len(vertices)):
        if i & (1 << j):
          current_set.add(vertices[j])

      current_set.add(start)
      power_set.add(current_set)
    
    for group in power_set:
      edges = set()

      for edge in group:
        edges = edges.union(self._graph[edge])

      edges = filter(lambda x: x[0] not in group, edges) # [(a,0)]

    pass



  def __get_edges(self) -> List[Tuple[str, str]]:
    result = set()

    for vertex, connections in self._graph:
      for connection in connections:
        result.add((vertex, connection[0], connection[1]))
    
    return list(result)



  @property
  def edges(self) -> List[Tuple[str, str, float]]:
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
