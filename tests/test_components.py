#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_components
----------------------------------

Tests for `components` module.
"""

import unittest
import graphlab as gl

from sptgraph import sptgraph
from sptgraph import components

from data import *  # import test data


class TestComponents(unittest.TestCase):

    def setUp(self):
        pass

    # @unittest.skip('Skipping undirected self edge')
    def test_undirected_self_edge(self):
        # Undirected, self-edges:
        g = sptgraph.create_spatio_temporal_graph(gen_graph(False), gen_signal(), True, verbose=False)
        h, cc = components.find_connected_components(g)
        comps = components.extract_components(h, cc)
        self.assertEqual(1, len(comps))

    def test_directed_no_self_edge(self):
        # Directed no self-edge
        g = sptgraph.create_spatio_temporal_graph(gen_graph(True), gen_signal(), False, verbose=False)

        # The graph has only 1 connected component
        h, cc = components.find_connected_components(g)
        comps = components.extract_components(h, cc)
        self.assertEqual(1, len(comps))

        # We remove the edge (7, 12) to create 2 weakly connected components
        nodes = g.vertices
        edges = g.edges.add_row_number('eid')
        to_remove = g.get_edges(7, 12)
        edges = edges[edges['eid'] != to_remove['eid'][0]]
        g = gl.SGraph(nodes, edges)

        h, cc = components.find_connected_components(g)
        comps = components.extract_components(h, cc)
        self.assertEqual(2, len(comps))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

