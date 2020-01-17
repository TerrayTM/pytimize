import math

from typing import List, Tuple

class UndirectedGraph: #TODO validation 
  def __init__(self, edges: List[Tuple(str, str, float)]=None):
    self.graph = {}

    if edges is not None:
      if not isinstance(edges, float):
        raise TypeError("Edges must be a list.")

      for edge in edges:
        if not isinstance(edge, tuple):
          raise TypeError("Each entry of edges must be a tuple.")

        if not len(edge) == 3:
          raise TypeError("Each tuple must have 3 items.")

        if not isinstance(edge[0], str) or not isinstance(edge[1], str):
          raise TypeError("First two entries of tuple must be strings.")

        self.add_edge((edge[0], edge[1]), edge[2])



  def add_edge(self, edge: Tuple(str, str), weight: float=0):
    if edge[0] in self.graph.keys() and edge[1] in self.graph[edge[0]]:
      raise ValueError("The given edge is already in graph.")

    if weight < 0:
      raise ValueError("Weight cannot be negative.")

    self.graph.setdefault(edge[0], set()).add((edge[1], weight))
    self.graph.setdefault(edge[1], set()).add((edge[0], weight))



  def remove_edge(self, edge: Tuple(str, str)): #TODO change ALL not ... in to ... not in
    if edge[0] not in self.graph.keys() or edge[1] not in self.graph[edge[0]]:
      raise ValueError("The given edge is not in graph.")

    self.graph[edge[0]].remove(edge[1])
    self.graph[edge[1]].remove(edge[0])



  def has_edge(self, edge: Tuple(str, str)):
    return edge[0] in self.graph.keys() and edge[1] in self.graph[edge[0]]



  def has_vertex(self, vertex: str):
    return vertex in self.graph.keys()



  def shortest_path(self, start: str, end: str):
    if start not in self.graph.keys():
      raise ValueError("Starting vertex is not in graph.")

    if end not in self.graph.keys():
      raise ValueError("Ending vertex is not in graph.")

    potential = {}
    visited = { start }
    directed_graph = {}

    while end not in visited:
      best_slack = math.inf
      best_edge = None
      edges = set()

      for vertex in visited:
        for edge in self.graph[vertex]:
          if edge[0] not in visited:
            edge_hash = [edge[0], vertex]

            edge_hash.sort()
            
            edge_hash = "".join(edge_hash)
            slack = edge[1] - (potential[edge_hash] if edge_hash in potential.keys() else 0)

            print(edge_hash, slack)

            edges.add(edge_hash)

            if slack < best_slack:
              best_slack = slack
              best_edge = (vertex, edge[0], edge_hash)
      
            print(visited, best_slack)

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

      path.insert(0, current + previous)

    return path



  def to_linear_program(self):
    pass



  def __get_edges(self):
    result = set()

    for vertex, connections in self.graph:
      for connection in connections:
        result.add((vertex, connection))
    
    return list(result)



  @property
  def edges(self):
      """
      Gets the edges of the graph.

      Returns
      -------
      result : list of tuples (str, str)

      """
      return self.__get_edges()



  @property
  def vertices(self):
      """
      Gets the vertices of the graph.

      Returns
      -------
      result : list of str

      """
      return list(self.graph.keys())
