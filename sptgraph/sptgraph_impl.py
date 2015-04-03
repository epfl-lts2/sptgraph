# -*- coding: utf-8 -*-

import graphlab as gl
from ast import literal_eval as make_list


def create_causal_edges_bitstring(src, edge, dst):
    edge['sp_edges'] = str(shift_and_bitstrings(src['layers'], dst['layers']))
    return src, edge, dst


def shift_and_bitstrings(src_bitstring, tgt_bitstring):
    src_layers = int(src_bitstring)
    dst_layers = int(tgt_bitstring)
    # Right shift
    dst_layers >>= 1
    return src_layers & dst_layers


def expand_causal_edges_from_bitfield(bitfield, base_src, base_tgt, max_id):
    edges = list()
    # Find all activated causal edges
    while bitfield:
        # extract lsb on 2s complement machine
        index = bitfield & -bitfield
        bitfield ^= index

        # Get activated layer number (log2)
        layer = -1
        while index:
            index >>= 1
            layer += 1
        src = base_src + (layer * max_id)
        tgt = base_tgt + ((layer + 1) * max_id)  # on next layer
        edges.append([src, tgt])

    return str(edges)


def build_sptgraph(sg, create_self_edges, baseid_name, layer_name):
    # It is used to generate node ids in the causal multilayer graph
    # IMPORTANT baseID starts at 1 and not 0
    max_id = sg.vertices['__id'].max()

    def expand_edge_layers(x):
        """Closure, capture max_id to generate all edges for the graph"""
        base_src = x['__src_id']
        base_tgt = x['__dst_id']
        bitfield = int(x['sp_edges'])
        return expand_causal_edges_from_bitfield(bitfield, base_src, base_tgt, max_id)

    def expand_vertex_layers(x):
        """Closure, capture max_id to generate all edges for the graph"""
        base_src = x['__id']
        base_tgt = x['__id']
        bitfield = shift_and_bitstrings(x['layers'], x['layers'])
        return expand_causal_edges_from_bitfield(bitfield, base_src, base_tgt, max_id)

    def generate_sp_edge_sframe(x):
        return make_list(x['sp_edges'])

    # Create empty field which will hold the bitstring for the edge creation
    sg.edges['sp_edges'] = ''
    # Create edge bitstring
    sg = sg.triple_apply(create_causal_edges_bitstring, mutated_fields=['sp_edges'])
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
