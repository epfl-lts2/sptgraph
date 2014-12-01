# -*- coding: utf-8 -*-

import graphlab as gl
import numpy as np


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
        'size': gl.aggregate.COUNT('__id')
    })

    comps['width'] = comps.apply(lambda x: len(np.unique(x[layers])))
    comps['height'] = comps.apply(lambda x: len(np.unique(x[baseids])))

    return comps.sort('size', False)


def extract_components(g, comp_sframe, min_size=2):
    comps = []
    nodes = g.vertices
    edges = g.edges
    cc = comp_sframe[comp_sframe['size'] > min_size]
    for cid in cc['component_id']:
        n = nodes[nodes['component_id'] == cid]
        e = edges[edges['component_id'] == cid]
        c = gl.SGraph(n, e)
        comps.append(c)
    return comps

