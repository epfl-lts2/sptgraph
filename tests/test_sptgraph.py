#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sptgraph
----------------------------------

Tests for `sptgraph` module.
"""

import unittest

from sptgraph import sptgraph

from sptgraph import utils

from data import *  # import test data


class TestSptgraph(unittest.TestCase):

    def setUp(self):
        import graphlab as gl

    # @unittest.skip('Skipping create_node_signal')
    def test_create_node_signal(self):
        node_signal = sptgraph.create_node_signal(gen_signal(), 'baseID', 'layer', False)
        node_signal = node_signal.sort('baseID')
        self.assertEqual('15', node_signal['layers'][0])
        self.assertEqual('6', node_signal['layers'][1])
        self.assertEqual('6', node_signal['layers'][2])
        self.assertEqual('6', node_signal['layers'][3])

    # @unittest.skip('Skipping networkx_to_graphlab')
    def test_networkx_to_graphlab(self):
        g = utils.networkx_to_graphlab(gen_graph(False))
        self.assertEqual(5, len(g.vertices))
        self.assertEqual(10, len(g.edges))

        g = utils.networkx_to_graphlab(gen_graph(True))
        self.assertEqual(5, len(g.vertices))
        self.assertEqual(5, len(g.edges))

    # @unittest.skip('Skipping create_signal_graph')
    def test_create_signal_graph(self):
        node_signal = sptgraph.create_node_signal(gen_signal(), 'baseID', 'layer', False)

        for is_directed in [True, False]:
            g = utils.networkx_to_graphlab(gen_graph(is_directed))
            sg = sptgraph.merge_signal_on_graph(g, node_signal, 'baseID', 'layer', use_fast=False, verbose=False)

            # Node 5 is never activated
            self.assertEqual(4, len(sg.vertices))

            if not is_directed:
                self.assertEqual(8, len(sg.edges))
            else:
                self.assertEqual(4, len(sg.edges))

            actual_columns = set(sg.vertices.column_names())
            expected_columns = {'layers', 'node_weight', '__id'}
            self.assertItemsEqual(expected_columns, actual_columns)

    def test_create_spatio_temporal_graph(self):
        # Undirected, self-edges:
        g = utils.networkx_to_graphlab(gen_graph(False))
        #   2-2
        #  / X \
        # 1-1-1-1
        # |\|X|/ |
        # | 3-3  |
        # | |X|  |
        # L 4-4--|
        h = sptgraph.create_spatio_temporal_graph(g, gen_signal(), True, verbose=False)
        self.assertEqual(13, h.vertices['__id'].max())
        self.assertEqual(3, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(10, len(h.vertices))
        self.assertEqual(20, len(h.edges))

        # Directed self-edge
        g = utils.networkx_to_graphlab(gen_graph(True))
        h = sptgraph.create_spatio_temporal_graph(g, gen_signal(), True, verbose=False)
        self.assertEqual(13, h.vertices['__id'].max())
        self.assertEqual(3, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(10, len(h.vertices))
        self.assertEqual(13, len(h.edges))

        # Undirected no self-edge
        g = utils.networkx_to_graphlab(gen_graph(False))
        h = sptgraph.create_spatio_temporal_graph(g, gen_signal(), False, verbose=False)
        self.assertEqual(13, h.vertices['__id'].max())
        self.assertEqual(3, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(10, len(h.vertices))
        self.assertEqual(14, len(h.edges))

        # Directed no self-edge
        g = utils.networkx_to_graphlab(gen_graph(True))
        h = sptgraph.create_spatio_temporal_graph(g, gen_signal(), False, verbose=False)
        self.assertEqual(12, h.vertices['__id'].max())
        self.assertEqual(2, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(8, len(h.vertices))
        self.assertEqual(7, len(h.edges))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

