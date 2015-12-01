# -*- coding: utf-8 -*-

import time

import networkx as nx
import pandas as pd

import graphlab as gl
from bitsets import bitset

# Project's imports
import utils
import sptgraph_impl
import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

try:
    import ext.sptgraph_fast as fast
    HAS_FAST_MODULE = True
except ImportError:
    HAS_FAST_MODULE = False


def create_node_signal(signal, baseid_name, layer_name, verbose=True):
    """Create signal on the node from pandas DataFrame or Graphlab SFrame"""

    def layers_to_long_str(x):
        """Convert layer number into bitstring then long str and add them as attributes on the nodes"""
        return str(layer_set(tuple(x['layers'])))

    start = time.time()
    if verbose:
        LOGGER.info('Create node signal')

    nb_layers = signal[layer_name].max() + 1
    # Create the bitspace
    layer_set = bitset('Layers', tuple(range(nb_layers)))
    # Aggregate layers per node
    node_signal = signal.groupby(baseid_name, {'layers': gl.aggregate.CONCAT(layer_name)})
    layer_bitstring = node_signal.apply(layers_to_long_str)
    # Remove old layers column and replace it with the bitstring one
    node_signal = node_signal.remove_column('layers').add_column(layer_bitstring, 'layers')

    if verbose:
        LOGGER.info('Create node signal done in: %s seconds', time.time() - start)

    return node_signal


def create_node_signal_fast(signal, baseid_name, layer_name, verbose=True):
    start = time.time()
    nb_layers = signal[layer_name].max() + 1  # starts at 0
    res = fast.aggregate_layers(signal, baseid_name, layer_name, nb_layers)
    if verbose:
        LOGGER.info('Create node signal (fast) done in: %s seconds', time.time() - start)
    return res


def merge_signal_on_graph(g, node_signal, baseid_name, layer_name, excluded_ids=None, use_fast=True,
                          verbose=True, remove_self=True):
    start = time.time()
    if verbose:
        LOGGER.info('Start reducing graph to minimum')

    p = g
    if isinstance(g, nx.Graph) or isinstance(g, nx.DiGraph):  # convert if needed
        start2 = time.time()
        if verbose:
            LOGGER.info('Start networkx to graphlab conversion')
        p = utils.networkx_to_graphlab(g)
        if verbose:
            LOGGER.info('Conversion done in: %s', time.time() - start2)

    if use_fast:
        good_nodes = node_signal
        good_nodes.rename({baseid_name: '__id'})
    else:
        good_nodes = p.vertices.join(node_signal, on={'__id': baseid_name}, how='inner')

    if excluded_ids:
        good_nodes = good_nodes.filter_by(excluded_ids, '__id', exclude=True)

    good_edges = p.get_edges(dst_ids=good_nodes['__id']).filter_by(good_nodes['__id'], '__src_id')

    # Remove self-edges
    if remove_self:
        good_edges = good_edges[(good_edges['__src_id'] - good_edges['__dst_id']) != 0]

    if verbose:
        LOGGER.info('Graph reduction done in: %s seconds', time.time() - start)

    return gl.SGraph(good_nodes, good_edges)


def create_spatio_temporal_graph(g, data, create_self_edges=True,
                                 baseid_name='baseID', layer_name='layer', verbose=True,
                                 force_python=False, excluded_ids=None):
    start = time.time()
    if verbose:
        LOGGER.info('Start spatio-temporal graph creation')

    signal = gl.SFrame(data)

    if HAS_FAST_MODULE and not force_python:
        node_signal = create_node_signal_fast(signal, baseid_name, layer_name, verbose=verbose)
    else:
        node_signal = create_node_signal(signal, baseid_name, layer_name, verbose=verbose)
    sg = merge_signal_on_graph(g, node_signal, baseid_name, layer_name, excluded_ids=excluded_ids, verbose=verbose)
    # Create graph
    if HAS_FAST_MODULE and not force_python:
        h = fast.build_sptgraph(sg, baseid_name, layer_name, create_self_edges, verbose=verbose)
    else:
        h = sptgraph_impl.build_sptgraph(sg, create_self_edges, baseid_name, layer_name)

    k = gl.SGraph(h.vertices.join(signal, ['page_id', 'layer']), h.edges)

    if verbose:
        LOGGER.info('Spatio-temporal graph created in: %s seconds', time.time() - start)

    return k


def get_max_id(spatial_graph):
    return spatial_graph.vertices['__id'].max()
