class DirectedGraph:
    def __init__(self):
        """
        Constructs a directed graph with support for arc weights and node demands.

        """
        self._graph = {}
        self._arcs = {}
        self._nodes = {}



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
            if their conenction count is 0.

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



    def  remove_node(self, node):
        """
        Adds a node to the digraph.

        Parameters
        ----------
        node : str
            The identifier of the node.

        demand : float (default=0)
            The demand of the node.

        """
        pass



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
        return node in self._nodes.keys()



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
        return arc in self._arcs.keys()



    def indegree(self, nodes):
        """
        Counts the indegree of a node or a set of nodes.

        Parameters
        ----------
        nodes : Union[str, List[str]]
            The node identifier or a set of node identifiers. 

        Returns
        -------
        result : int
            The indegree of the node or a set of nodes.

        """
        if isinstance(nodes, str):
            if not self.has_node(nodes):
                raise ValueError("Given node does not exist in digraph.")

            nodes = [nodes]
        else:
            raise ValueError("Given node does not exist in digraph.")

        degree = 0
        nodes = set(nodes)

        for node, arcs in self._graph:
            if not node in nodes:
                degree += len(arcs.difference(nodes))

        return degree



    def outdegree(self, nodes):
        """
        Counts the outdegree of a node or a set of nodes.

        Parameters
        ----------
        nodes : Union[str, List[str]]
            The node identifier or a set of node identifiers. 

        Returns
        -------
        result : int
            The outdegree of the node or a set of nodes.

        """
        if isinstance(nodes, str):
            if not self.has_node(nodes):
                raise ValueError("Given node does not exist in digraph.")

            return len(self._graph[nodes])

        nodes = set(nodes)
        degree = 0

        for node in nodes:
            degree += len(self._graph[node].difference(nodes))
        
        return degree



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
    def edges(self):
        pass

    @property
    def vertices(self):
        pass
 
