# -*- coding: utf-8 -*-

import time

import networkx as nx
import pandas as pd

import graphlab as gl
from bitsets import bitset

# Project's imports
import utils
import sptgraph_impl


def create_node_signal(data, baseid_name, layer_name, verbose=True):
    """Create signal on the node from pandas DataFrame or Graphlab SFrame"""
    start = time.time()
    if verbose:
        print 'Create node signal'

    if isinstance(data, pd.DataFrame):
        signal = gl.SFrame(data[[baseid_name, layer_name]])
    else:
        signal = gl.SFrame(data)

    nb_layers = signal[layer_name].max() + 1
    # Create the bitspace
    layer_set = bitset('Layers', tuple(range(nb_layers)))
    # Aggregate layers per node
    node_signal = signal.groupby(baseid_name, {'layers': gl.aggregate.CONCAT(layer_name)}).sort(baseid_name)

    def layers_to_long_str(x):
        """Convert layer number into bitstring then long str and add them as attributes on the nodes"""
        return str(layer_set(tuple(x['layers'].tolist())))

    layer_bitstring = node_signal.apply(layers_to_long_str)

    # Remove old layers column and replace it with the bitstring one
    node_signal = node_signal.remove_column('layers').add_column(layer_bitstring, 'layers')

    if verbose:
        print 'Create node signal done in:', time.time() - start, 'seconds'

    return node_signal, layer_set


def create_signal_graph(g, node_signal, baseid_name, layer_name, verbose=True):
    start = time.time()
    if verbose:
        print 'Create signal graph'

    p = g
    if isinstance(g, nx.Graph) or isinstance(g, nx.DiGraph):  # convert if needed
        start2 = time.time()
        if verbose:
            print '  Convert networkx graph to graphlab'

        p = utils.networkx_to_graphlab(g)

        if verbose:
            print '  Convert networkx graph to graphlab done in:', time.time() - start2, 'seconds'

    # Filter nodes and join from original graph and signal
    good_nodes = p.vertices.join(node_signal, on={'__id': baseid_name}, how='inner')
    # Filter edges
    good_edges = p.get_edges(src_ids=good_nodes['__id']).join(p.get_edges(dst_ids=good_nodes['__id']))

    if verbose:
        print 'Create signal graph done in:', time.time() - start, 'seconds'

    return gl.SGraph(good_nodes, good_edges)


def create_spatio_temporal_graph(g, data, create_self_edges=True,
                                 baseid_name='baseID', layer_name='layer', verbose=True):
    start = time.time()
    if verbose:
        print 'Create spatio-temporal graph'

    node_signal, layer_set = create_node_signal(data, baseid_name, layer_name, verbose=verbose)
    sg = create_signal_graph(g, node_signal, baseid_name, layer_name, verbose=verbose)
    # Create graph
    h = sptgraph_impl.build_sptgraph(sg, layer_set, create_self_edges, baseid_name, layer_name)

    if verbose:
        print 'Create spatio-temporal graph done in:', time.time() - start, 'seconds'

    return h

