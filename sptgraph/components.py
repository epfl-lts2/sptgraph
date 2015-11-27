# -*- coding: utf-8 -*-

import sys
import os
import graphlab as gl
import numpy as np
import logging
from collections import defaultdict, Counter
import itertools

import utils

LOGGER = logging.getLogger(__name__)
HAS_GRAPHTOOL = False
HAS_NETWORKX = False

STATIC_COMP_TYPE = 1
DYN_COMP_TYPE = 2
ALL_COMP_TYPE = STATIC_COMP_TYPE + DYN_COMP_TYPE

try:
    import graph_tool.all as gt
    import graph_tool.community as gtc
    import graph_tool.stats as gts
    import graph_tool.util as gtu
    import graph_tool.topology as gtt
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
    g.component = comp['component_id']

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
    g.gp.component = g.new_graph_property('int')
    g.gp.component = comp['component_id']
    g.gp.type = g.new_graph_property('int')
    g.gp.type = DYN_COMP_TYPE

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
            g.vp.timestamp[v] = layer_to_ts[comp[layers][i]]

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

    node_hist = Counter(dyn_g.vertex_properties[baseid_name].get_array())
    unique_baseids = node_hist.keys()

    N = len(unique_baseids)
    assert(N > 0)

    g = gt.Graph(directed=True)
    g.gp.component = g.new_graph_property('int')
    g.gp.component = dyn_g.gp.component
    g.gp.type = g.new_graph_property('int')
    g.gp.type = STATIC_COMP_TYPE

    vlist = g.add_vertex(N)
    if N == 1:
        vlist = [vlist]
    baseid_map = dict(zip(unique_baseids, vlist))

    # Node importance
    g.vp.count = g.new_vertex_property('int', 0)
    for k, v in node_hist.iteritems():
        g.vp.count[baseid_map[k]] = v

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
    g.component = dyn_g.component
    g.type = STATIC_COMP_TYPE
    # Add unique nodes
    node_hist = Counter(nx.get_node_attributes(dyn_g, baseid_name).values())
    g.add_nodes_from([(k, {'count': v}) for k, v in node_hist.iteritems()])

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


def filter_singletons(g, purge=False):
    deg_filter = g.new_vertex_property('bool', False)
    g_deg = g.degree_property_map('total')
    deg_filter.a[:] = g_deg.a > 0
    subset = g_deg.a == 2  # set to True only nodes with total_deg = 2
    for e in g.edges():
        if e.source() == e.target():  # if self-loop
            if subset[e.source()]:  # in set of nodes with tot_deg = 2
                deg_filter[e.source()] = False  # filter out node
    g.set_vertex_filter(deg_filter)

    if purge:
        g.purge_vertices()
        # g.purge_edges()
    return g


def partition_static_component(g, threshold, baseid_name='baseID', weighted=None,
                               filter_single=True, remove_self_edges=False, slack=0.25):
    if not HAS_GRAPHTOOL:
        LOGGER.error('Graph-tool not installed, cannot use function partition_static_component')
        raise ImportError('Graph-tool not installed, cannot use function partition_static_component')

    if threshold > 0.0:
        filts = list()
        filts.append(g.edge_properties['in_score'].a > threshold)
        filts.append(g.edge_properties['out_score'].a > threshold)
        if remove_self_edges:
            # Keep non self-edges
            self_edges = ~gts.label_self_loops(g, mark_only=True).a.astype(bool)
            filts.append(self_edges)
        # Logical and of all filters
        edge_filt = np.logical_and.reduce(filts)
        g = gt.GraphView(g, efilt=edge_filt)

    if filter_single:
        g = filter_singletons(g)

    if g.num_vertices() == 0:
        return None

    if g.num_vertices() < 4:  # do not partition below 4 nodes
        g.vp.cluster_id = g.new_vertex_property('int', 0)
        return g

    # Find number of connected components as a minimum of communities
    res, _ = gtt.label_components(g, directed=False)

    min_clusters = len(np.unique(res.fa))
    max_clusters = int(min_clusters * (2 + slack))  # add slack
    if min_clusters == max_clusters:
        max_clusters += 1

    if min_clusters == 1:  # no prior on the graph
        max_clusters = None

    # monkey-path graph_tool in blockmodel.py
    clusters = gtc.minimize_blockmodel_dl(g, min_B=min_clusters, max_B=max_clusters, verbose=False).b
    # Add communities
    g.vp.cluster_id = clusters
    return g


def partition_dynamic_component(dyn_g, static_g, baseid_name='baseID', filter_single=True):
    """Find the block partition of an unspecified size which minimizes the description length of the
    network, according to the stochastic blockmodel ensemble which best describes it. Optionally
    remove low likelihood in and out edges from a dynamic activated component and static component.
    """
    if not HAS_GRAPHTOOL:
        LOGGER.error('Graph-tool not installed, cannot use function partition_dynamic_component')
        raise ImportError('Graph-tool not installed, cannot use function partition_dynamic_component')

    to_keep_static = defaultdict(list)
    for e in static_g.edges():
        u = static_g.vertex_properties[baseid_name][e.source()]
        v = static_g.vertex_properties[baseid_name][e.target()]
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

    if filter_single:
        dyn_g = filter_singletons(dyn_g)

    # Backport communities to dynamic component
    id2cluster = dict(itertools.izip(static_g.vertex_properties[baseid_name].a, static_g.vp.cluster_id.a))
    dyn_g.vp.cluster_id = dyn_g.new_vertex_property('int')

    for n in dyn_g.vertices():
        cluster_id = id2cluster[dyn_g.vertex_properties[baseid_name][n]]
        dyn_g.vp.cluster_id[n] = cluster_id

    return dyn_g


def extract_community_subgraphs(comp):
    """Extract communities from a component. Associated properties are copied in each community and the property
    `cluster_id` is stored as a graph property for each community

    """
    def mirror_property_maps(src_g, tgt_g):
        for k, v in src_g.properties.iteritems():
            if k != ('v', 'cluster_id'):
                tgt_g.properties[k] = tgt_g.new_property(k[0], v.value_type())

    def create_graph(gid):
        g = gt.Graph(directed=True)
        mirror_property_maps(comp, g)
        g.gp.component = g.new_graph_property('int')
        g.gp.component = comp.gp.component  # comp id
        g.gp.cluster_id = g.new_graph_property('int')
        g.gp.cluster_id = gid
        g.gp.type = g.new_graph_property('int')
        g.gp.type = comp.gp.type
        return g

    def copy_props(key, src, src_g, tgt, tgt_g):
        src_props = src_g.vertex_properties if key == 'v' else src_g.edge_properties
        tgt_props = tgt_g.vertex_properties if key == 'v' else tgt_g.edge_properties
        for k, v in src_props.iteritems():
            if k in tgt_props:
                tgt_props[k][tgt] = v[src]

    # Create graphs
    graphs = map(create_graph, np.unique(comp.vp.cluster_id.a))

    # Map comp vertex to subgraphs vertex
    vertex_map = dict()
    # Create nodes and fill property maps
    for n in comp.vertices():
        cid = comp.vp.cluster_id[n]
        v = graphs[cid].add_vertex()
        vertex_map[n] = v
        copy_props('v', n, comp, v, graphs[cid])

    # Create edges
    for e in comp.edges():
        src_cid = comp.vp.cluster_id[e.source()]
        tgt_cid = comp.vp.cluster_id[e.target()]
        # Edges should belong to the same community
        if src_cid != tgt_cid:
            continue

        arc = graphs[src_cid].add_edge(vertex_map[e.source()], vertex_map[e.target()], False)
        copy_props('e', e, comp, arc, graphs[src_cid])

    return graphs


def save_gt_component(c, out_dir, i=0, verbose=False):
    name = ''
    if 'type' in c.graph_properties:
        name += 'st_' if c.gp.type == STATIC_COMP_TYPE else 'dyn_'
    name += 'comp_'
    if 'component' in c.graph_properties:
        name += str(c.gp.component)
    else:
        name += str(i)

    if 'cluster_id' in c.graph_properties:
        name += '_' + str(c.gp.cluster_id)

    path = str(os.path.join(out_dir, name + '.gt'))
    c.save(path)
    if verbose:
        LOGGER.info('Wrote {0}'.format(path))


def save_gt_components(comps, out_dir, verbose=False):
    """Save graph-tool components to disk"""
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, c in enumerate(comps):
        save_gt_component(c, out_dir, i, verbose)


def load_gt_components(input_dir, comp_type=ALL_COMP_TYPE):
    """Load graph-tool components from disk, if comp type is 2 return all types of components"""
    comps = defaultdict(list)
    for f in utils.list_dir(input_dir, fullpath=True):
        path = str(f)
        name = os.path.basename(path)
        if name.startswith('st') and comp_type in [STATIC_COMP_TYPE, ALL_COMP_TYPE]:
            comps['static'].append(gt.load_graph(path))

        if name.startswith('dyn') and comp_type in [DYN_COMP_TYPE, ALL_COMP_TYPE]:
            comps['dynamic'].append(gt.load_graph(path))
    return comps


def extract_molecular_components(comp, h, out_dir=None, score_threshold=0.05, baseid_name='page_id',
                                 layer_name='layer', layer_to_ts=None, with_dynamic=True, verbose=False):
    """Extract molecular components from graphlab dynamic activated component. The score threshold allows
    to prune out low-likelihood in and out edges when the dynamic component is contracted to a static component.
    The parameter `with_dynamic` returns the associated dynamic molecular components associated to normal output: the
    static molecular components. The parameter layer_ts maps a layer id to a timestamp for better readability.

    """
    dyn_comp = component_to_graphtool(comp, h, baseid_name, layer_name, layer_to_ts)
    static_comp = get_weighted_static_component(dyn_comp, baseid_name)
    static_comp = partition_static_component(static_comp, score_threshold, baseid_name)
    if static_comp is None:  # filter all nodes
        # print 'Filter out', dyn_comp.gp.component
        return None

    static_coms = extract_community_subgraphs(static_comp)

    if out_dir:
        save_gt_components(static_coms, out_dir, verbose)

    if with_dynamic:
        dyn_comp = partition_dynamic_component(dyn_comp, static_comp, baseid_name)
        dyn_coms = extract_community_subgraphs(dyn_comp)
        if out_dir:
            save_gt_components(dyn_coms, out_dir, verbose)

        return zip(static_coms, dyn_coms)
    else:
        return static_coms


def extract_all_molecular_components_par(gl_components, h, out_dir, score_threshold=0.05, baseid_name='page_id',
                                         layer_name='layer', layer_to_ts=None, with_dynamic=True):
    """Extract all molecular components in parallel by dumping them on disk and reading them back sequentially.
    Returns static molecular components (optionally dynamic ones) and the number of molecules per components.

    """
    def go(x):
        _ = extract_molecular_components(x, h, out_dir, score_threshold, baseid_name,
                                         layer_name, layer_to_ts, with_dynamic)
        return 0
    gl_components.apply(go, dtype=int)
    return load_gt_components(out_dir, STATIC_COMP_TYPE + (DYN_COMP_TYPE if with_dynamic else 0))


def extract_all_molecular_components_seq(gl_components, h, out_dir, score_threshold=0.05, baseid_name='page_id',
                                         layer_name='layer', layer_to_ts=None, with_dynamic=True):
    """Extract all molecular components sequentially by dumping them on disk and reading them back sequentially.
    Returns static molecular components (optionally dynamic ones) and the number of molecules per components.
    """
    for row in gl_components:
        _ = extract_molecular_components(row, h, out_dir, score_threshold, baseid_name,
                                         layer_name, layer_to_ts, with_dynamic)
    return load_gt_components(out_dir, STATIC_COMP_TYPE + (DYN_COMP_TYPE if with_dynamic else 0))
