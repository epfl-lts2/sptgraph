# -*- coding: utf-8 -*-

import networkx as nx
import pandas as pd


def gen_graph(directed):
    g = nx.Graph()

    if directed:
        g = nx.DiGraph()

    # Add 5 nodes
    for i in xrange(1, 6):
        g.add_node(i, node_weight=i)

    # Add edges
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(1, 3, weight=2.0)
    g.add_edge(1, 4, weight=3.0)
    g.add_edge(3, 4, weight=4.0)
    g.add_edge(2, 5, weight=5.0)

    return g


def gen_signal():
    data = [
        # Node A
        {"baseID": 1, "layer": 0, "value": 0.1},
        {"baseID": 1, "layer": 1, "value": 0.1},
        {"baseID": 1, "layer": 2, "value": 0.1},
        {"baseID": 1, "layer": 3, "value": 0.1},
        # Node B
        {"baseID": 2, "layer": 1, "value": 0.2},
        {"baseID": 2, "layer": 2, "value": 0.2},
        # Node C
        {"baseID": 3, "layer": 1, "value": 0.3},
        {"baseID": 3, "layer": 2, "value": 0.3},
        # Node D
        {"baseID": 4, "layer": 1, "value": 0.4},
        {"baseID": 4, "layer": 2, "value": 0.4},
    ]
    return pd.DataFrame(data)
