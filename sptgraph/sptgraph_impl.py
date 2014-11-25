# -*- coding: utf-8 -*-

import graphlab as gl
from ast import literal_eval as make_list


def create_edges(item, key, layer_set, base_src, base_tgt, max_id):
    layers = layer_set.frombits(item[key]).members()
    edges = list()
    for l in layers:
        src = base_src + (l * max_id)
        tgt = base_tgt + ((l + 1) * max_id)  # on next layer
        edges.append([src, tgt])
    return str(edges)


def generate_sp_edge_sframe(x):
    return make_list(x['sp_edges'])


def build_sptgraph(sg, layer_set, create_self_edges, baseid_name, layer_name):

    # It is used to generate node ids in the spatio-temporal graph
    max_id = sg['__id'].max()

    def spatio_edge_creation(src, edge, dst):
        """Closure (captures layer_set) to create all the
        spatio-temporal edges from a given spatial edge.
        """
        src_layers = src['layers']
        dst_layers = dst['layers']
        # Left shift on a string .. (bitset library does not support shifting)
        dst_layers = dst_layers[1:] + '0'
        res = layer_set.fromint(layer_set.frombits(src_layers) & layer_set.frombits(dst_layers))
        edge['sp_edges'] = res.bits()
        return src, edge, dst

    def expand_edge_bitstring(x):
        """Closure, capture layer_set and max_id to generate all edges for the spt graph"""
        base_src = x['__src_id']
        base_tgt = x['__dst_id']
        return create_edges(x, 'sp_edges', layer_set, base_src, base_tgt, max_id)

    def expand_vertex_bitstring(x):
        """Closure, capture layer_set and max_id to generate all self-edges for the spt graph"""
        base_src = x['__id']
        base_tgt = x['__id']
        return create_edges(x, 'layers', layer_set, base_src, base_tgt, max_id)

    # Create empty field which will hold the bitstring for the edge creation
    sg.edges['sp_edges'] = ''
    # Create edge bitstring
    sg = sg.triple_apply(spatio_edge_creation, mutated_fields=['sp_edges'])
    # Expand to actual source and destination as a string
    sg.edges['sp_edges'] = sg.edges.apply(expand_edge_bitstring, dtype=str)
    # Create new sframe with actual ids
    sp_edges = sg.edges.flat_map(['source', 'dest'], generate_sp_edge_sframe, column_types=[int, int])
    # Create the graph from edges
    h = gl.SGraph().add_edges(sp_edges, src_field='source', dst_field='dest')
    del sg.edges['sp_edges']

    if create_self_edges:
        sg.vertices['sp_edges'] = ''
        # Expand to actual source and destination as a string
        sg.vertices['sp_edges'] = sg.vertices.apply(expand_vertex_bitstring, dtype=str)
        # Create new sframe with actual ids
        sp_edges = sg.vertices.flat_map(['source', 'dest'], generate_sp_edge_sframe, column_types=[int, int])
        h = h.add_edges(sp_edges, src_field='source', dst_field='dest')
        del sg.vertices['sp_edges']

    # Add baseid and layer to spt graph
    h.vertices[baseid_name] = h.vertices.apply(lambda x: x['__id'] % max_id, dtype=int)
    h.vertices[layer_name] = h.vertices.apply(lambda x: x['__id'] / max_id, dtype=int)

    return h
