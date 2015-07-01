# -*- coding: utf-8 -*-

import graphlab as gl
import numpy as np
import networkx as nx
import logging

import utils

LOGGER = logging.getLogger(__name__)
HAS_GRAPHTOOL = False

try:
    import graph_tool.all as gt
    HAS_GRAPHTOOL = True
except ImportError:
    LOGGER.warning('graph-tool package not found, some functions will be disabled')
    HAS_GRAPHTOOL = False

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


def partition_dynamic_component(dyn_g, static_g=None, threshold=0.0, thres_key='score', baseid_name='baseID'):
    """Partition using Louvain modularity and optionally remove low
    probability edges from a dynamic activated component and static_component.
    """
    if static_g is None:
        static_g = get_weighted_static_component(dyn_g, baseid_name)

    if threshold > 0.0:
        edges = nx.get_edge_attributes(static_g, thres_key)
        to_remove_static = [k for k, v in edges.iteritems() if v <= threshold]
        static_g.remove_edges_from(to_remove_static)

        md = utils.to_multi_dict(to_remove_static)
        to_remove_dyn = []
        for u, v in dyn_g.edges_iter():
            src = dyn_g.node[u][baseid_name]
            tgt = dyn_g.node[v][baseid_name]
            if src in md:
                if tgt in md[src]:
                    to_remove_dyn.append((u, v))

        dyn_g.remove_edges_from(to_remove_dyn)

    parts = community.best_partition(dyn_g)
    print parts

    return dyn_g