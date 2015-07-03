# -*- coding: utf-8 -*-

import graphlab as gl
import numpy as np
import logging
from collections import defaultdict
import itertools

import utils

LOGGER = logging.getLogger(__name__)
HAS_GRAPHTOOL = False
HAS_NETWORKX = False

try:
    import graph_tool.all as gt
    import graph_tool.community as gtc
    HAS_GRAPHTOOL = True
except ImportError:
    LOGGER.warning('graph-tool package not found, some functions will be disabled')
    HAS_GRAPHTOOL = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    LOGGER.warning('Networkx package not found, some functions will be disabled')
    HAS_NETWORKX = False


def find_connected_components(g):
    cc = gl.graph_analytics.connected_components.create(g, False)
    g.vertices['component_id'] = cc['component_id']['component_id']
    nodes = g.vertices
    edges = g.edges.join(cc['component_id'], on={'__src_id': '__id'})
    return gl.SGraph(nodes, edges)


def get_component_sframe(g, baseid_name='baseID', layer_name='layer'):
    """Get component SFrame enriched with structural properties for each component"""

    baseids = baseid_name + 's'  # store array of base ids
    layers = layer_name + 's'
    comps = g.vertices.groupby('component_id', {
        'nodes': gl.aggregate.CONCAT('__id'),
        layers: gl.aggregate.CONCAT(layer_name),
        baseid_name + 's': gl.aggregate.CONCAT(baseid_name),
        'node_count': gl.aggregate.COUNT('__id')
    })

    comps['width'] = comps.apply(lambda x: len(np.unique(x[layers])))
    comps['height'] = comps.apply(lambda x: len(np.unique(x[baseids])))

    return comps.sort('node_count', False)


def extract_components(g, comp_sframe, min_node_count=2):
    comps = []
    nodes = g.vertices
    edges = g.edges
    cc = comp_sframe[comp_sframe['node_count'] > min_node_count]
    for cid in cc['component_id']:
        n = nodes[nodes['component_id'] == cid]
        e = edges[edges['component_id'] == cid]
        c = gl.SGraph(n, e)
        comps.append(c)
    return comps


def component_to_networkx(comp, h, baseid_name='baseID', layer_name='layer', layer_to_ts=None):
    if not HAS_NETWORKX:
        LOGGER.error('Networkx not installed, cannot use function')
        raise ImportError('Networkx not installed, cannot use function')

    g = nx.DiGraph()
    g.name = 'Component ' + str(comp['component_id'])

    layers = layer_name + 's'
    baseids = baseid_name + 's'

    if layer_to_ts is not None:
        for i in xrange(comp['node_count']):
            g.add_node(comp['nodes'][i],
                      {baseid_name: comp[baseids][i], layer_name: comp[layers][i],
                       'timestamp': layer_to_ts[comp[layers][i]].strftime("%Y-%m-%d %H:%M:%S"),
                       'label': comp[baseids][i]
                      })
    else:
        for i in xrange(comp['node_count']):
            g.add_node(comp['nodes'][i], {baseid_name: comp[baseids][i], layer_name: comp[layers][i]})

    edges = h.edges[h.edges['component_id'] == comp['component_id']][['__src_id', '__dst_id']]
    for k in edges:
        g.add_edge(k['__src_id'], k['__dst_id'])

    return g


def component_to_graphtool(comp, h, baseid_name='baseID', layer_name='layer', layer_to_ts=None):
    if not HAS_GRAPHTOOL:
        LOGGER.error('graph-tool not installed, cannot use function')
        raise ImportError('graph-tool not installed, cannot use function')

    layers = layer_name + 's'
    baseids = baseid_name + 's'

    # Create graph
    g = gt.Graph(directed=True)
    g.gp.name = g.new_graph_property('string')
    g.gp.name = 'Component ' + str(comp['component_id'])

    # Vertex properties
    g.vertex_properties[baseid_name] = g.new_vertex_property("int64_t")
    g.vertex_properties[layer_name] = g.new_vertex_property("int32_t")
    g.vp.nid = g.new_vertex_property("int64_t")
    if layer_to_ts is not None:
        g.vp.timestamp = g.new_vertex_property("string")

    # Add nodes
    vertex_map = dict()  # idx mapping
    vlist = g.add_vertex(comp['node_count'])
    for i, v in enumerate(vlist):
        g.vertex_properties[baseid_name][v] = comp[baseids][i]
        g.vertex_properties[layer_name][v] = comp[layers][i]
        g.vp.nid[v] = comp['nodes'][i]

        vertex_map[comp['nodes'][i]] = v

        if layer_to_ts is not None:
            g.vp.timestamp[v] = layer_to_ts[comp[layers][i]].strftime("%Y-%m-%d %H:%M:%S")

    # Add edges
    edges = h.edges[h.edges['component_id'] == comp['component_id']][['__src_id', '__dst_id']]
    for k in edges:
        src = vertex_map[k['__src_id']]
        tgt = vertex_map[k['__dst_id']]
        g.add_edge(src, tgt, False)

    return g

def get_weighted_static_component(dyn_g, baseid_name='baseID'):
    """Flatten dynamic component to a spatial static graph with some properties on the graph."""
    if HAS_NETWORKX and isinstance(dyn_g, nx.DiGraph):
        return _get_weighted_static_component_nx(dyn_g, baseid_name)

    if HAS_GRAPHTOOL and isinstance(dyn_g, gt.Graph):
        return _get_weighted_static_component_gt(dyn_g, baseid_name)

    LOGGER.error('Dynamic graph format not supported')
    return None


def _get_weighted_static_component_gt(dyn_g,  baseid_name='baseID'):
    if not HAS_GRAPHTOOL:
        LOGGER.error('Graph-tool not installed, cannot use function _get_weighted_static_component_gt')
        raise ImportError('Graph-tool not installed, cannot use function _get_weighted_static_component_gt')

    unique_baseids = np.unique(dyn_g.vertex_properties[baseid_name].get_array())

    g = gt.Graph(directed=True)
    vlist = g.add_vertex(len(unique_baseids))
    baseid_map = dict(zip(unique_baseids, vlist))

    g.vp.in_deg = g.new_vertex_property('int', 0)
    g.vp.out_deg = g.new_vertex_property('int', 0)
    g.vertex_properties[baseid_name] = g.new_vertex_property('int64_t')

    g.ep.count = g.new_edge_property('int', 1)
    g.ep.out_score = g.new_edge_property('double', 0)
    g.ep.in_score = g.new_edge_property('double', 0)
    g.ep.score = g.new_edge_property('double', 0)

    for arc in dyn_g.edges():
        # static g
        u = dyn_g.vertex_properties[baseid_name][arc.source()]
        v = dyn_g.vertex_properties[baseid_name][arc.target()]

        src = baseid_map[u]
        tgt = baseid_map[v]

        # Add baseid prop
        g.vertex_properties[baseid_name][src] = u
        g.vertex_properties[baseid_name][tgt] = v

        e = g.edge(src, tgt)
        if e:
            g.ep.count[e] += 1
        else:
            g.add_edge(src, tgt)

        # inc source and target degree
        g.vp.out_deg[src] += 1
        g.vp.in_deg[tgt] += 1

    # Normalize weights
    for e in g.edges():
        out_score = g.ep.count[e] / float(g.vp.out_deg[e.source()])
        in_score = g.ep.count[e] / float(g.vp.in_deg[e.target()])
        g.ep.out_score[e] = out_score
        g.ep.in_score[e] = in_score
        g.ep.score[e] = (out_score + in_score) / 2

    return g


def _get_weighted_static_component_nx(dyn_g,  baseid_name='baseID'):
    if not HAS_NETWORKX:
        LOGGER.error('Networkx not installed, cannot use function _get_weighted_static_component_nx')
        raise ImportError('Networkx not installed, cannot use function _get_weighted_static_component_nx')

    def inc_prop(g, nid, key):
        deg = g.node[nid].get(key, None)
        if deg:
            g.node[nid][key] += 1
        else:
            g.node[nid][key] = 1

    g = nx.DiGraph()  # directed + self-edges
    # Add unique nodes
    g.add_nodes_from(set(nx.get_node_attributes(dyn_g, baseid_name).values()))

    for (u, v) in dyn_g.edges_iter():
        src = dyn_g.node[u][baseid_name]
        tgt = dyn_g.node[v][baseid_name]

        if g.has_edge(src, tgt):
            g[src][tgt]['count'] += 1
        else:
            g.add_edge(src, tgt, count=1)

        inc_prop(g, src, 'out_degree')
        inc_prop(g, tgt, 'in_degree')

    # Normalize counts
    for (u, v, d) in g.edges_iter(data=True):
        d['out_score'] = d['count'] / float(g.node[u]['out_degree'])
        d['in_score'] = d['count'] / float(g.node[v]['in_degree'])
        d['weight'] = (d['out_score'] + d['in_score']) / 2
        d['score'] = d['weight']  # mostly for tulip which cannot display the weight prop ...

    return g


def partition_dynamic_component(dyn_g, static_g=None, threshold=0.0, thres_key='score', baseid_name='baseID',
                                weighted=True, method='block', purge_singletons=True):
    """Find the block partition of an unspecified size which minimizes the description length of the
    network, according to the stochastic blockmodel ensemble which best describes it. optionally
    remove low probability edges from a dynamic activated component and static_component.
    """
    if not HAS_GRAPHTOOL:
        LOGGER.error('Graph-tool not installed, cannot use function partition_dynamic_component')
        raise ImportError('Graph-tool not installed, cannot use function partition_dynamic_component')

    def _purge_singletons(g):
        g_deg = g.new_vertex_property('bool', False)
        deg = g.degree_property_map('total')
        g_deg.a[:] = deg.a > 0
        g.set_vertex_filter(g_deg)

    if static_g is None:
        static_g = get_weighted_static_component(dyn_g, baseid_name)

    if threshold > 0.0:
        filt_static = gt.GraphView(static_g, efilt=lambda e: static_g.edge_properties[thres_key][e] > threshold)

        to_keep_static = defaultdict(list)
        for e in filt_static.edges():
            u = filt_static.vertex_properties[baseid_name][e.source()]
            v = filt_static.vertex_properties[baseid_name][e.target()]
            to_keep_static[u].append(v)

        to_keep_dyn = dyn_g.new_edge_property('bool', False)
        for e in dyn_g.edges():
            u = dyn_g.vertex_properties[baseid_name][e.source()]
            v = dyn_g.vertex_properties[baseid_name][e.target()]
            if u in to_keep_static:
                if v in to_keep_static[u]:
                    to_keep_dyn[e] = True

        # keep all edges set to False (no deletion)
        dyn_g.set_edge_filter(to_keep_dyn)
        static_g = filt_static
        
    if purge_singletons:
        _purge_singletons(static_g)
        _purge_singletons(dyn_g)

    # Cluster static graph
    clusters = None
    if method == 'block':
        clusters = gtc.minimize_blockmodel_dl(static_g).b
    else:
        clusters = gtc.community_structure(static_g, 1000, 10, t_range=(5, 0.1),
                                           weight=static_g.edge_properties[thres_key] if weighted else None)

    # Add communities
    static_g.vp.cluster_id = clusters

    # Backport communities to dynamic component
    id2cluster = dict(itertools.izip(static_g.vertex_properties[baseid_name].a, clusters.a))
    dyn_g.vp.cluster_id = dyn_g.new_vertex_property('int')

    for n in dyn_g.vertices():
        cluster_id = id2cluster[dyn_g.vertex_properties[baseid_name][n]]
        dyn_g.vp.cluster_id[n] = cluster_id

    return dyn_g, static_g


