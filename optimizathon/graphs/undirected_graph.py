from math import inf
  
class UndirectedGraph(Graph):
  def __init__(self, edges: list=None):
    self.graph = {}

    if edges is not None:
      for edge in edges:
        self.add_edge(edge)

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


# g = UndirectedGraph()
# g.add_edge('s', 'a', 6)
# g.add_edge('a', 't', 4)
# g.add_edge('t', 'b', 5)
# g.add_edge('s', 'b', 2)
# g.add_edge('c', 's', 4)
# g.add_edge('b', 'c', 1)
# g.add_edge('a', 'c', 1)
# g.add_edge('c', 't', 2)


# print(g.shortest_path('s', 't'))


# g = UndirectedGraph()
# g.add_edge('s', 'a', 3)
# g.add_edge('a', 'b', 4)
# g.add_edge('b', 't', 1)
# g.add_edge('t', 'd', 2)
# g.add_edge('d', 'c', 2)
# g.add_edge('c', 'b', 2)
# g.add_edge('a', 'c', 1)
# g.add_edge('s', 'c', 4)


# print(g.shortest_path('s', 't'))

# g = UndirectedGraph()

# g.add_edge('a','c',6)
# g.add_edge('a','b',4)
# g.add_edge('b','c',2)
# g.add_edge('c','g',5)
# g.add_edge('b','g',3)
# g.add_edge('c','d',1)
# g.add_edge('d','g',2)
# g.add_edge('d','e',5)
# g.add_edge('e','f',6)
# g.add_edge('f','d',1)
# g.add_edge('f','g',4)
# g.add_edge('g','h',5)
# g.add_edge('h','i',3)
# g.add_edge('i','g',2)
# g.add_edge('j','i',9)
# g.add_edge('j','k',6)
# g.add_edge('k','g',4)
# g.add_edge('k','l',1)
# g.add_edge('l','b',5)

# print(g.shortest_path('j', 'c')) #['jk', 'kg', 'gd', 'dc']

# print(g.shortest_path('g', 'e')) #['gd', 'de']
# print(g.shortest_path('a', 'c')) #['gd', 'de']


class NonlinearProgram:
  def __init__():
    pass

  def gradient_descent_solver(self):
    pass
