#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sptgraph
----------------------------------

Tests for `sptgraph C++` module.
"""

import unittest
import graphlab as gl

from sptgraph import sptgraph
from sptgraph.ext import sptgraph_fast
from sptgraph import utils

from data import *  # import test data

class TestSptgraphFast(unittest.TestCase):

    def setUp(self):
        pass

#    @unittest.skip('Skipping create_node_signal')
    def test_create_node_signal(self):
        node_signal = sptgraph.create_node_signal(gen_signal(), 'baseID', 'layer', False)
        self.assertEqual('15', node_signal['layers'][0])
        self.assertEqual('6', node_signal['layers'][1])
        self.assertEqual('6', node_signal['layers'][2])
        self.assertEqual('6', node_signal['layers'][3])

#    @unittest.skip('Skipping networkx_to_graphlab')
    def test_networkx_to_graphlab(self):
        g = utils.networkx_to_graphlab(gen_graph(False))
        self.assertEqual(5, len(g.vertices))
        self.assertEqual(10, len(g.edges))

        g = utils.networkx_to_graphlab(gen_graph(True))
        self.assertEqual(5, len(g.vertices))
        self.assertEqual(5, len(g.edges))

#    @unittest.skip('Skipping create_signal_graph')
    def test_create_signal_graph(self):
        node_signal = sptgraph.create_node_signal(gen_signal(), 'baseID', 'layer', False)

        for is_directed in [True, False]:
            g = utils.networkx_to_graphlab(gen_graph(is_directed))
            sg = sptgraph.reduce_graph_to_signal(g, node_signal, 'baseID', 'layer', verbose=False)

            # Node 5 is never activated
            self.assertEqual(4, len(sg.vertices))

            if not is_directed:
                self.assertEqual(8, len(sg.edges))
            else:
                self.assertEqual(4, len(sg.edges))

            actual_columns = set(sg.vertices.column_names())
            expected_columns = {'layers', 'node_weight', '__id'}
            self.assertItemsEqual(expected_columns, actual_columns)

#    @unittest.skip('Skipping aggregate layers')
    def test_aggregate_layers(self):
        signal = gl.SFrame(gen_signal())
        nb_layers = signal['layer'].max() + 1  # starts at 0

        # Python 'slow'
        original = sptgraph.create_node_signal(signal, 'baseID', 'layer', False)
        # Fast c++ version
        res = sptgraph_fast.aggregate_layers(signal, 'baseID', 'layer', nb_layers)

        # Transform output to compare
        l1 = original['layers'].apply(int)
        l2 = res['layers'].apply(utils.reform_layer_int_from_blocks)
        m = l1 == l2
        self.assertTrue(m.all(), 'Layers should be equal')

#    @unittest.skip('Skipping rebuilt_bitset')
    def test_rebuild_bitset(self):
        signal = gl.SFrame(gen_signal())
        nb_layers = signal['layer'].max() + 1  # starts at 0
        res = sptgraph_fast.aggregate_layers(signal, 'baseID', 'layer', nb_layers)
        l1 = res['layers'].apply(utils.reform_layer_int_from_blocks)
        l2 = map(lambda x: int(x, 2), res['layers'].apply(sptgraph_fast.flex_bitset_to_flex_string))
        m = l1 == gl.SArray(l2)
        self.assertTrue(m.all(), 'Layers should be equal')

    def test_sptgraph_undirected_self_edges(self):
        # Most complex case
        #   2-2
        #  / X \
        # 1-1-1-1
        # |\|X|/ |
        # | 3-3  |
        # | |X|  |
        # L 4-4--|
        h_ref = sptgraph.create_spatio_temporal_graph(gen_graph(False), gen_signal(), True, verbose=False, force_python=True)
        h = sptgraph.create_spatio_temporal_graph(gen_graph(False), gen_signal(), True, verbose=False, force_python=False)
        self.assertEqual(13, h.vertices['__id'].max())
        self.assertEqual(3, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(10, len(h.vertices))
        self.assertEqual(20, len(h.edges))
        self.check_equal_graphs(h_ref, h)

    def test_sptgraph_directed_self_edges(self):
        h_ref = sptgraph.create_spatio_temporal_graph(gen_graph(True), gen_signal(), True, verbose=False, force_python=True)
        h = sptgraph.create_spatio_temporal_graph(gen_graph(True), gen_signal(), True, verbose=False)
        self.assertEqual(13, h.vertices['__id'].max())
        self.assertEqual(3, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(10, len(h.vertices))
        self.assertEqual(13, len(h.edges))
        self.check_equal_graphs(h_ref, h)

    def test_sptgraph_undirected_no_self_edges(self):
        h_ref = sptgraph.create_spatio_temporal_graph(gen_graph(False), gen_signal(), False, verbose=False, force_python=True)
        h = sptgraph.create_spatio_temporal_graph(gen_graph(False), gen_signal(), False, verbose=False)
        self.assertEqual(13, h.vertices['__id'].max())
        self.assertEqual(3, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(10, len(h.vertices))
        self.assertEqual(14, len(h.edges))
        self.check_equal_graphs(h_ref, h)

    def test_sptgraph_directed_no_self_edges(self):
        h_ref = sptgraph.create_spatio_temporal_graph(gen_graph(True), gen_signal(), False, verbose=False, force_python=True)
        h = sptgraph.create_spatio_temporal_graph(gen_graph(True), gen_signal(), False, verbose=False)
        self.assertEqual(12, h.vertices['__id'].max())
        self.assertEqual(2, h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(8, len(h.vertices))
        self.assertEqual(7, len(h.edges))
        self.check_equal_graphs(h_ref, h)



    def check_equal_graphs(self, h_ref, h):
        self.assertEqual(h_ref.vertices['__id'].max(), h.vertices['__id'].max())
        self.assertEqual(h_ref.vertices['layer'].max(), h.vertices['layer'].max())  # layer starts at 0
        self.assertEqual(len(h_ref.vertices), len(h.vertices))
        self.assertEqual(len(h_ref.edges), len(h.edges))
        self.assertEqual(h_ref.vertices['__id'], h.vertices['__id'], 'Node ids should be equal')
        self.assertEqual(h_ref.vertices['baseID'], h.vertices['baseID'], 'Node base ids should be equal')
        self.assertEqual(h_ref.vertices['layer'], h.vertices['layer'], 'Node layers should be equal')
        self.assertEqual(h_ref.edges['__src_id'], h.edges['__src_id'], 'Edges sources should be equal')
        self.assertEqual(h_ref.edges['__dst_id'], h.edges['__dst_id'], 'Edges destination should be equal')

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

