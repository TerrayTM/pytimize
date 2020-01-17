
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
