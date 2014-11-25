# -*- coding: utf-8 -*-

import graphlab as gl


def networkx_to_graphlab(g):
    p = gl.SGraph()
    # Add nodes
    p = p.add_vertices(map(lambda x: gl.Vertex(x[0], attr=x[1]), g.nodes(data=True)))
    # Add edges
    p = p.add_edges(map(lambda x: gl.Edge(x[0], x[1], attr=x[2]), g.edges(data=True)))
    if not g.is_directed:  # undirected
        p = p.add_edges(map(lambda x: gl.Edge(x[1], x[0], attr=x[2]), g.edges(data=True)))
    return p

