import math

from typing import Iterable, Union, List, Tuple, Dict, Set, Optional
from collections import deque

class DirectedGraph:
    def __init__(self, graph: Optional[Dict[str, Set[str]]]=None, arcs: Optional[Dict[Tuple[str, str], float]]=None, nodes: Optional[Dict[str, float]]=None) -> None:
        """
        Constructs a directed graph with support for arc and node weights.
        If `graph`, `arcs`, `nodes`, or a combination of them is provided, the 
        graph will be built respectively with any nonspecified weights set to 0.

        Parameters
        ----------
        graph : Optional[Dict[str, Set[str]]] (default=None)
            The graph dictionary where key is the node and value is the set of 
            nodes that are connected to that node.

        arcs : Optional[Dict[Tuple[str, str], float]] (default=None)
            The arcs dictionary where key is the arc and value is the weight. 

        nodes : Optional[Dict[str, float]] (default=None)
            The nodes dictionary where key is the node and value is the weight.

        """
        self._graph = {}
        self._arcs = {}
        self._nodes = {}

        if graph is not None:
            for node, connections in graph.items():
                if len(connections) == 0:
                    if not self.has_node(node):
                        self.add_node(node)
                else:
                    for connection in connections:
                        arc = (node, connection)

                        if self.has_arc(arc):
                            continue

                        self.add_arc(arc)

        if arcs is not None: 
            for arc, weight in arcs.items():
                if not self.has_arc(arc):
                    self.add_arc(arc, weight)

                self.set_arc_weight(arc, weight)

        if nodes is not None:
            for node, weight in nodes.items():
                if not self.has_node(node): 
                    self.add_node(node)

                self.set_node_weight(node, weight)



    def set_arc_weight(self, arc: Tuple[str, str], weight: float) -> None:
        if not self.has_arc(arc):
            raise ValueError("The given edge is not in graph.")

        self._arcs[arc] = weight



    def set_node_weight(self, node: str, weight: float) -> None:
        if not self.has_node(node):
            raise ValueError("The given vertex is not in graph.")

        self._nodes[node] = weight



    def add_arc(self, arc, weight=0, demands=None):
        """
        Adds an arc to the digraph. If adding the arc creates one or more new nodes, 
        `demands` can be set to initialize the potential of those new nodes. In such case 
        if `demands` is none, then by default the new nodes will have potentials of 0.

        Parameters
        ----------
        arc : Tuple[str, str]
            The identifier of the arc.

        weight : float (default=0)
            The weight of the arc.

        demands : Union[float, Tuple[float, float]] (default=None)
            The potential assigned to the new nodes, if any are created. If both endpoints of
            the arc are new nodes, then a tuple of demands is expected, one entry for each end. 

        """
        if self.has_arc(arc):
            raise ValueError("The given arc already exists in digraph.")

        if arc[0] == arc[1]:
            raise ValueError("Cannot create an arc with the same endpoints.")

        self._graph.setdefault(arc[0], set()).add(arc[1])
        self._arcs.setdefault(arc, weight)

        a_exists = self.has_node(arc[0])
        b_exists = self.has_node(arc[1])

        if not a_exists and not b_exists:
            if demands and not isinstance(demands, tuple):
                raise TypeError("Demands must be a tuple of floats when both endpoints are new nodes.")

            self.add_node(arc[0], demands[0] if demands else 0)
            self.add_node(arc[1], demands[1] if demands else 0)
        elif not a_exists:
            self.add_node(arc[0], demands[0] if demands else 0)
        elif not b_exists:
            self.add_node(arc[1], demands[1] if demands else 0)



    def add_node(self, node, demand=0):
        """
        Adds a node to the digraph.

        Parameters
        ----------
        node : str
            The identifier of the node.

        demand : float (default=0)
            The demand of the node.

        """
        if self.has_node(node):
            raise ValueError("The given node already exists in digraph.")
        
        self._graph.setdefault(node, set())
        self._nodes.setdefault(node, demand)



    # TODO fix bug in undirected graph
    def remove_arc(self, arc, remove_nodes=False):
        """
        Removes an arc from the digraph.

        Parameters
        ----------
        arc : Tuple[str, str]
            The identifier of the arc.

        remove_nodes : bool (default=False)
            Should the endpoint nodes of the arc also be removed
            if their connection count is 0.

        """
        if not self.has_arc(arc):
            raise ValueError("Target arc does not exist in digraph.")

        del self._arcs[arc]
        self._graph[arc[0]].remove(arc[1])

        if remove_nodes:
            if self.connections(arc[0]) == 0:
                del self._graph[arc[0]]
                del self._nodes[arc[0]]

            if self.connections(arc[1]) == 0:
                del self._graph[arc[1]]
                del self._nodes[arc[1]]



    def remove_node(self, node):
        """
        Removes a node from the digraph.

        Parameters
        ----------
        node : str
            The identifier of the node.

        demand : float (default=0)
            The demand of the node.

        """
        pass



    def cut(self, nodes: Union[Iterable[str], str]) -> Set[Tuple[str, str]]:
        """
        Gets a set of arcs where every arc has its starting node in `nodes`.

        Parameters
        ----------
        nodes : Union[Iterable[str], str]
            The node or nodes of the operation.

        Returns
        -------
        result : Set[Tuple[str, str]]
            The list of arcs that satisfies the above condition.

        """
        if not isinstance(nodes, set):
            nodes = set(nodes)

        return set(filter(lambda arc: arc[0] in nodes and arc[1] not in nodes, self._arcs.keys()))



    def ncut(self, nodes: Union[Iterable[str], str]) -> Set[Tuple[str, str]]:
        """
        Gets a set of arcs where every arc has its ending node in `nodes`.

        Parameters
        ----------
        nodes : Union[Iterable[str], str]
            The node or nodes of the operation.

        Returns
        -------
        result : Set[Tuple[str, str]]
            The list of arcs that satisfies the above condition.

        """

        if not isinstance(nodes, set):
            nodes = set(nodes)

        return list(filter(lambda arc: arc[1] in nodes and arc[0] not in nodes, self._arcs.keys()))



    def has_node(self, node):
        """
        Checks if the given node exists in digraph.

        Parameters
        ----------
        node : str
            The identifier of the node.

        Returns
        -------
        result : bool
            If the node is in the digraph or not.

        """
        return node in self._nodes



    def has_arc(self, arc):
        """
        Checks if the given arc exists in digraph.

        Parameters
        ----------
        arc : Tuple[str, str]
            The identifier of the arc.

        Returns
        -------
        result : bool
            If the arc is in the digraph or not.

        """
        return arc in self._arcs



    def indegree(self, nodes: Union[Iterable[str], str]) -> int:
        """
        Computes the indegree of a node or a set of nodes.

        Parameters
        ----------
        nodes : Union[Iterable[str], str]
            The node identifier or a set of node identifiers. 

        Returns
        -------
        result : int
            The indegree of the node or a set of nodes.

        """
        return len(self.ncut(nodes))



    def outdegree(self, nodes: Union[Iterable[str], str]) -> int:
        """
        Computes the outdegree of a node or a set of nodes.

        Parameters
        ----------
        nodes : Union[Iterable[str], str]
            The node identifier or a set of node identifiers. 

        Returns
        -------
        result : int
            The outdegree of the node or a set of nodes.

        """
        return len(self.cut(nodes))



    def connections(self, nodes):
        """
        Counts the number of connections a node or a set of nodes have. A connection
        of a node is defined as an arc connecting the node to its neighbour regardless
        of direction.

        Parameters
        ----------
        nodes : Union[str, List[str]]
            The node identifier or a set of node identifiers. 

        Returns
        -------
        result : int
            The number of connections of the node or a set of nodes.

        """
        return self.indegree(nodes) + self.outdegree(nodes)



    def create_residual(self, flow: Dict[Tuple[str, str], float]) -> "DirectedGraph":
        if any(arc not in self._arcs or value < 0 for arc, value in flow.items()):
            raise ValueError()

        if any(value < 0 for value in self._arcs.values()):
            raise ValueError()

        residual = self._compute_residual(flow)
        arcs = {}

        for node, endpoints in residual.items():
            for endpoint, value in endpoints.items():
                arcs[(node, endpoint)] = value

        return DirectedGraph(arcs=arcs)



    def _compute_residual(self, flow: Dict[Tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
        residual = {}

        for arc, capacity in self._arcs.items():
            u, v = arc
            value = flow.get(arc, 0)

            if capacity < value:
                raise ArithmeticError("Given flow is infeasible.")

            if capacity > value:
                residual.setdefault(u, {})[v] = capacity - value

            if value > 0:
                residual.setdefault(v, {})[u] = value

        return residual



    def preflow_push(self, source: str, sink: str) -> Tuple[float, Dict[Tuple[str, str], float]]:
        """
        Computes the maximum flow from `source` to `sink` using the Preflow Push algorithm.
        The directed graph must specify nonnegative arc weights as capcities.

        Parameters
        ----------
        source: str
            The source node of where flow is sent.

        sink: str
            The sink node of where flow is received.

        Returns
        -------
        result : Tuple[float, Dict[Tuple[str, str], float]]
            A tuple representing the maximal flow value and the actual flow itself.

        """
        if not self.has_node(source):
            raise ValueError("Source node does not exist.")

        if not self.has_node(sink):
            raise ValueError("Sink node does not exist.")

        if source == sink:
            raise ValueError("Source cannot be the same as the sink.")

        index = 0
        excess = {}
        source_cut = self.cut(source)
        flow = {arc: self._arcs[arc] if arc in source_cut else 0 for arc in self._arcs}
        residual = self._compute_residual(flow)
        height = {node: len(self._nodes) if node == source else 0 for node in self._nodes}
        nodes = list(filter(lambda node: not node == source and not node == sink, self._nodes))

        for node in nodes:
            excess[node] = sum(flow[arc] for arc in self.ncut(node))
            excess[node] -= sum(flow[arc] for arc in self.cut(node))

        while index < len(nodes):
            node = nodes[index]

            if excess[node] > 0:
                pushed = False

                for endpoint, value in residual[node].items():
                    if height[endpoint] == height[node] - 1:
                        pushed = True
                        is_reversed = False
                        arc = (node, endpoint)
                        push_value = min(excess[node], value)
                        u, v = node, endpoint

                        if arc not in flow:
                            is_reversed = True
                            arc = (endpoint, node)
                            flow[arc] -= push_value
                        else:
                            flow[arc] += push_value

                        if is_reversed:
                            u, v = v, u

                        residual.setdefault(u, {})[v] = self._arcs[arc] - flow[arc]
                        residual.setdefault(v, {})[u] = flow[arc]

                        if residual[u][v] == 0:
                            del residual[u][v]
                    
                        if residual[v][u] == 0:
                            del residual[v][u]

                        if len(residual[u]) == 0:
                            del residual[u]
                
                        if len(residual[v]) == 0:
                            del residual[v]

                        if not node == source and not node == sink:
                            excess[node] -= push_value

                        if not endpoint == source and not endpoint == sink:
                            excess[endpoint] += push_value

                        break

                if not pushed:
                    max_height = math.inf

                    for endpoint in residual[node]:
                        max_height = min(height[endpoint], max_height) 

                    height[node] = max_height + 1

                index = 0
            else:
                index += 1

            value = sum(flow[arc] for arc in self.cut(source))
            value -= sum(flow[arc] for arc in self.ncut(source))

        return value, flow



    def _bfs_path(self, source, target, graph):
        queue = deque([source])        
        seen = set([source])
        parent = {}

        while len(queue) > 0:
            current = queue.popleft()

            for endpoint in graph.get(current, {}):
                if endpoint in seen:
                    continue
                
                parent[endpoint] = current

                if endpoint == target:
                    return parent
                else:
                    seen.add(endpoint)
                    queue.append(endpoint)

        return None



    def ford_fulkerson(self, source: str, sink: str) -> Tuple[float, Dict[Tuple[str, str], float]]:
        """
        Computes the maximum flow from `source` to `sink` using the Ford Fulkerson algorithm.
        The directed graph must specify nonnegative arc weights as capcities.

        Parameters
        ----------
        source: str
            The source node of where flow is sent.

        sink: str
            The sink node of where flow is received.

        Returns
        -------
        result : Tuple[float, Dict[Tuple[str, str], float]]
            A tuple representing the maximal flow value and the actual flow itself.

        """
        if not self.has_node(source):
            raise ValueError("Source node does not exist.")

        if not self.has_node(sink):
            raise ValueError("Sink node does not exist.")

        if source == sink:
            raise ValueError("Source cannot be the same as the sink.")

        flow = {arc: 0 for arc in self._arcs}
        residual = self._compute_residual(flow)
        path = self._bfs_path(source, sink, residual)

        while path is not None:
            u = path[sink]
            v = sink
            min_residual = math.inf

            while u is not None:
                min_residual = min(min_residual, residual[u][v])
                v = u
                u = path.get(u, None)

            u = path[sink]
            v = sink

            while u is not None:
                arc = (u, v)

                if arc not in flow:
                    arc = (v, u)
                    flow[arc] -= min_residual
                else:
                    flow[arc] += min_residual

                residual.setdefault(u, {})[v] = self._arcs[arc] - flow[arc]
                residual.setdefault(v, {})[u] = flow[arc]

                if residual[u][v] == 0:
                    del residual[u][v]
            
                if residual[v][u] == 0:
                    del residual[v][u]

                if len(residual[u]) == 0:
                    del residual[u]
                
                if len(residual[v]) == 0:
                    del residual[v]

                v = u
                u = path.get(u, None)

            path = self._bfs_path(source, sink, residual)

        value = sum(flow[arc] for arc in self.cut(source))
        value -= sum(flow[arc] for arc in self.ncut(source))

        return value, flow



    def formulate_max_flow(self):
        pass


    def is_digraph_connected(self):
        pass

    def all_indegrees(self):
        pass

    def all_outdegrees(self):
        pass

    def adjacency_matrix(self):
        pass 

    def incidence_matrix(self):
        pass

    def minimum_spanning_tree(self):
        pass

    def has_dicycle(self):
        pass

    def is_diwalk(self):
        pass

    def is_dipath(self):
        pass

    def is_dicycle(self):
        pass

    def is_tree_solution(self):
        pass 

    def is_tree_flow(self): 
        pass

    @property
    def arcs(self):
        return list(self._arcs.keys())

    @property
    def nodes(self):
        return list(self._nodes.keys())
 
