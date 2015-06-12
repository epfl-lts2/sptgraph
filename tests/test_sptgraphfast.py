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

    @unittest.skip('Skipping aggregate layers')
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

    @unittest.skip('Skipping  rebuilt_bitset')
    def test_rebuild_bitset(self):
        signal = gl.SFrame(gen_signal())
        nb_layers = signal['layer'].max() + 1  # starts at 0
        res = sptgraph_fast.aggregate_layers(signal, 'baseID', 'layer', nb_layers)
        l1 = res['layers'].apply(utils.reform_layer_int_from_blocks)
        l2 = map(lambda x: int(x, 2), res['layers'].apply(sptgraph_fast.flex_bitset_to_flex_string))
        m = l1 == gl.SArray(l2)
        self.assertTrue(m.all(), 'Layers should be equal')

    def test_sptgraph(self):
        signal = gl.SFrame(gen_signal())
        nb_layers = signal['layer'].max() + 1  # starts at 0
        node_signal = sptgraph_fast.aggregate_layers(signal, 'baseID', 'layer', nb_layers)
        g = utils.networkx_to_graphlab(gen_graph(True))
        sg = sptgraph.reduce_graph_to_signal(g, node_signal, 'baseID', 'layer', verbose=False)
        h = sptgraph_fast.build_sptgraph(sg, 'baseID', 'layer', False)
#        print h.edges['sp_edges']
        print h

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

