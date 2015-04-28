# -*- coding: utf-8 -*-

import graphlab as gl
import numpy as np
import networkx as nx


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


def component_to_networkx(comp, h, layer_to_ts=None):
    g = nx.DiGraph()
    g.name = 'Component ' + str(comp['component_id'])

    if layer_to_ts is not None:
        for i in xrange(comp['node_count']):
            g.add_node(comp['nodes'][i],
                      {'base_id': comp['base_ids'][i], 'layer': comp['layers'][i],
                       'timestamp': layer_to_ts[comp['layers'][i]].strftime("%Y-%m-%d %H:%M:%S"),
                       'label': comp['base_ids'][i]
                      })
    else:
        for i in xrange(comp['node_count']):
            g.add_node(comp['nodes'][i], {'base_id': comp['base_ids'][i], 'layer': comp['layers'][i]})

    edges = h.edges[h.edges['component_id'] == comp['component_id']][['__src_id', '__dst_id']]

    for k in edges:
        g.add_edge(k['__src_id'], k['__dst_id'])
    return g

