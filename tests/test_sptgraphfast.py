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

    # @unittest.skip('Skipping create_node_signal')
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

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

