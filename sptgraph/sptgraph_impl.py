# -*- coding: utf-8 -*-

import graphlab as gl
from ast import literal_eval as make_list


def spatio_edge_creation(src, edge, dst):
    edge['sp_edges'] = str(build_edge_bitstring(src['layers'], dst['layers']))
    return src, edge, dst


def create_edges_from_layers(layers, base_src, base_tgt, max_id):
    edges = list()
    for l in layers:
        src = base_src + (l * max_id)
        tgt = base_tgt + ((l + 1) * max_id)  # on next layer
        edges.append([src, tgt])
    return str(edges)


def create_edges_from_item(item, key, layer_set, base_src, base_tgt, max_id):
    layers = layer_set.fromint(item[key]).members()
    return create_edges_from_layers(layers, base_src, base_tgt, max_id)


def generate_sp_edge_sframe(x):
    return make_list(x['sp_edges'])


def build_edge_bitstring(src_bitstring, tgt_bitstring):
    src_layers = int(src_bitstring)
    dst_layers = int(tgt_bitstring)
    # Right shift
    dst_layers >>= 1
    return src_layers & dst_layers


def build_sptgraph(sg, layer_set, create_self_edges, baseid_name, layer_name):
    # It is used to generate node ids in the spatio-temporal graph
    # IMPORTANT baseID starts at 1 and not 0
    max_id = sg.vertices['__id'].max()

    def expand_edge_layers(x):
        """Closure, capture layer_set and max_id to generate all edges for the spt graph"""
        base_src = x['__src_id']
        base_tgt = x['__dst_id']
        return create_edges_from_item(x, 'sp_edges', layer_set, base_src, base_tgt, max_id)

    def expand_vertex_layers(x):
        """Closure, capture layer_set and max_id to generate all self-edges for the spt graph"""
        base_src = x['__id']
        base_tgt = x['__id']
        val = build_edge_bitstring(x['layers'], x['layers'])
        layers = layer_set.fromint(val).members()
        return create_edges_from_layers(layers, base_src, base_tgt, max_id)

    # Create empty field which will hold the bitstring for the edge creation
    sg.edges['sp_edges'] = ''
    # Create edge bitstring
    sg = sg.triple_apply(spatio_edge_creation, mutated_fields=['sp_edges'])
    # Expand to actual source and destination as a string
    sg.edges['sp_edges'] = sg.edges.apply(expand_edge_layers, dtype=str)
    # Create new sframe with actual ids
    sp_edges = sg.edges.flat_map(['source', 'dest'], generate_sp_edge_sframe, column_types=[int, int])
    # Create the graph from edges
    h = gl.SGraph().add_edges(sp_edges, src_field='source', dst_field='dest')
    del sg.edges['sp_edges']

    if create_self_edges:
        sg.vertices['sp_edges'] = ''
        # Expand to actual source and destination as a string
        sg.vertices['sp_edges'] = sg.vertices.apply(expand_vertex_layers, dtype=str)
        # Create new sframe with actual ids
        sp_edges = sg.vertices.flat_map(['source', 'dest'], generate_sp_edge_sframe, column_types=[int, int])
        h = h.add_edges(sp_edges, src_field='source', dst_field='dest')
        del sg.vertices['sp_edges']

    # Add baseid and layer to spt graph
    h.vertices[layer_name] = h.vertices.apply(lambda x: (x['__id'] - 1) // max_id, dtype=int)
    # base_src = src - (l * max_id)
    h.vertices[baseid_name] = h.vertices.apply(lambda x: x['__id'] - (x[layer_name] * max_id), dtype=int)

    return h, max_id
