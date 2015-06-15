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

try:
    import ext.sptgraph_fast as fast
    HAS_FAST_MODULE = True
except ImportError:
    HAS_FAST_MODULE = False


def create_node_signal(data, baseid_name, layer_name, verbose=True):
    """Create signal on the node from pandas DataFrame or Graphlab SFrame"""

    def layers_to_long_str(x):
        """Convert layer number into bitstring then long str and add them as attributes on the nodes"""
        return str(layer_set(tuple(x['layers'])))

    start = time.time()
    if verbose:
        LOGGER.info('Create node signal')

    if isinstance(data, pd.DataFrame):
        signal = gl.SFrame(data[[baseid_name, layer_name]])
    else:
        signal = gl.SFrame(data)

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


def create_node_signal_fast(data, baseid_name, layer_name, verbose=True):
    start = time.time()
    if isinstance(data, pd.DataFrame):
        signal = gl.SFrame(data[[baseid_name, layer_name]])
    else:
        signal = gl.SFrame(data)

    nb_layers = signal[layer_name].max() + 1  # starts at 0
    res = fast.aggregate_layers(signal, baseid_name, layer_name, nb_layers)
    if verbose:
        LOGGER.info('Create node signal (fast) done in: %s seconds', time.time() - start)
    return res


def reduce_graph_to_signal(g, node_signal, baseid_name, layer_name, verbose=True):
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

    # Filter nodes and join from original graph and signal
    good_nodes = p.vertices.join(node_signal, on={'__id': baseid_name}, how='inner')
    # Filter edges
    good_edges = p.get_edges(src_ids=good_nodes['__id']).join(p.get_edges(dst_ids=good_nodes['__id']))

    if verbose:
        LOGGER.info('Graph reduction done in: %s seconds', time.time() - start)

    return gl.SGraph(good_nodes, good_edges)


def create_spatio_temporal_graph(g, data, create_self_edges=True,
                                 baseid_name='baseID', layer_name='layer', verbose=True, force_python=False):
    start = time.time()
    if verbose:
        LOGGER.info('Start spatio-temporal graph creation')

    if HAS_FAST_MODULE and not force_python:
        node_signal = create_node_signal_fast(data, baseid_name, layer_name, verbose=verbose)
    else:
        node_signal = create_node_signal(data, baseid_name, layer_name, verbose=verbose)
    sg = reduce_graph_to_signal(g, node_signal, baseid_name, layer_name, verbose=verbose)
    # Create graph
    if HAS_FAST_MODULE and not force_python:
        h = fast.build_sptgraph(sg, baseid_name, layer_name, create_self_edges, verbose)
    else:
        h = sptgraph_impl.build_sptgraph(sg, create_self_edges, baseid_name, layer_name)

    if verbose:
        LOGGER.info('Spatio-temporal graph created in: %s seconds', time.time() - start)

    return h


def get_max_id(spatial_graph):
    return spatial_graph.vertices['__id'].max()
