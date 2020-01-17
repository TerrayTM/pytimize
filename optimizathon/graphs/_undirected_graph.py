from math import inf
  
class UndirectedGraph:
  def __init__(self, edges: list=None):
    self.graph = {}

  def add_edge(self, source, target, weight): #not called source and targe since unordered check if edge exists then throw err
    self.graph.setdefault(source, set()).add((target, weight))
    self.graph.setdefault(target, set()).add((source, weight))

  def get_vertices(self):
    print(self.graph)
    pass

  def get_edges(self):
    pass

  def shortest_path(self, start, end):
    potential = {}
    visited = {start}
    directed_graph = {}

    while end not in visited:
      best_slack = inf
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
