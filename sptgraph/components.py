# -*- coding: utf-8 -*-

import graphlab as gl


def find_connected_components(g):
    cc = gl.graph_analytics.connected_components.create(g, False)
    g.vertices['component_id'] = cc['component_id']['component_id']
    nodes = g.vertices
    edges = g.edges.join(cc['component_id'], on={'__src_id': '__id'})
    return gl.SGraph(nodes, edges), cc['component_size']


def extract_components(g, cc, min_size=2):
    comps = []
    nodes = g.vertices
    edges = g.edges
    cc = cc[cc['Count'] > min_size]
    for cid in cc['component_id']:
        n = nodes[nodes['component_id'] == cid]
        e = edges[edges['component_id'] == cid]
        c = gl.SGraph(n, e)
        comps.append(c)
    return comps

