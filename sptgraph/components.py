# -*- coding: utf-8 -*-

import itertools
import logging
from collections import defaultdict, Counter

import graphlab as gl
import numpy as np
import pandas as pd

from dump import save_molecules_db,  STATIC_COMP_TYPE, DYN_COMP_TYPE

LOGGER = logging.getLogger('sptgraph')
LOGGER.setLevel(logging.INFO)

HAS_GRAPHTOOL = False
HAS_NETWORKX = False

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


def create_component_sframe(g, baseid_name='page_id', layer_name='layer'):
    """Get component SFrame enriched with structural properties for each component"""

    columns = g.vertices.column_names()
    columns.remove('__id')
    columns.remove('component_id')

    # Append s to have unique column names (required by graphlab)
    gb_dict = {c + 's': gl.aggregate.CONCAT(c) for c in columns}
    gb_dict['nids'] = gl.aggregate.CONCAT('__id')
    gb_dict['node_count'] = gl.aggregate.COUNT('__id')
    comps = g.vertices.groupby('component_id', gb_dict)

    comps['width'] = comps.apply(lambda x: len(np.unique(x[layer_name + 's'])))
    comps['height'] = comps.apply(lambda x: len(np.unique(x[baseid_name + 's'])))

    edges = g.edges.groupby('component_id', {'src': gl.aggregate.CONCAT('__src_id'),
                                             'tgt': gl.aggregate.CONCAT('__dst_id')})
    comps = comps.join(edges, 'component_id')
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


def component_to_networkx(comp, h, baseid_name='page_id', layer_name='layer', layer_to_ts=None):
    if not HAS_NETWORKX:
        LOGGER.error('Networkx not installed, cannot use function')
        raise ImportError('Networkx not installed, cannot use function')

    g = nx.DiGraph()
    g.component = comp['component_id']

    layers = layer_name + 's'
    baseids = baseid_name + 's'

    if layer_to_ts is not None:
        for i in xrange(comp['node_count']):
            g.add_node(comp['nids'][i],
                      {baseid_name: comp[baseids][i], layer_name: comp[layers][i],
                       'timestamp': layer_to_ts[comp[layers][i]].strftime("%Y-%m-%d %H:%M:%S"),
                       'label': comp[baseids][i]
                      })
    else:
        for i in xrange(comp['node_count']):
            g.add_node(comp['nids'][i], {baseid_name: comp[baseids][i], layer_name: comp[layers][i]})

    edges = h.edges[h.edges['component_id'] == comp['component_id']][['__src_id', '__dst_id']]
    for k in edges:
        g.add_edge(k['__src_id'], k['__dst_id'])

    return g


def component_to_graphtool(comp, baseid_name='page_id', layer_name='layer', layer_to_ts=None, extra_props=None):
    if not HAS_GRAPHTOOL:
        LOGGER.error('graph-tool not installed, cannot use function')
        raise ImportError('graph-tool not installed, cannot use function')

    # Create graph
    g = gt.Graph(directed=True)
    g.gp.component = g.new_graph_property('int', comp['component_id'])
    g.gp.type = g.new_graph_property('int', DYN_COMP_TYPE)

    # Create Vertex properties
    prop_map = {baseid_name: 'int64_t', layer_name: 'int32_t', 'nid': 'int64_t'}
    if extra_props is not None:  # extra prop are doubles
        if isinstance(extra_props, (basestring, unicode)):
            extra_props = (extra_props, )
        for p in extra_props:
            prop_map[p] = 'double'

    for k, v in prop_map.iteritems():
        g.vertex_properties[k] = g.new_vertex_property(v)

    # Special timestamp prop
    if layer_to_ts is not None:
        g.vertex_properties['timestamp'] = g.new_vertex_property('string')

    # Add nodes
    vertex_map = dict()  # idx mapping
    vlist = g.add_vertex(comp['node_count'])
    for i, v in enumerate(vlist):
        for prop_name in prop_map.keys():
            g.vertex_properties[prop_name][v] = comp[prop_name + 's'][i]
        vertex_map[comp['nids'][i]] = v

        if layer_to_ts is not None:
            g.vertex_properties['timestamp'][v] = layer_to_ts[comp[layer_name + 's'][i]]

    # Add edges
    for u, v in itertools.izip(comp['src'], comp['tgt']):
        src = vertex_map[u]
        tgt = vertex_map[v]
        g.add_edge(src, tgt, False)

    return g


def get_weighted_static_component(dyn_g, baseid_name='page_id', extra_props=('count_views', )):
    """Flatten dynamic component to a spatial static graph with some properties on the graph."""
    if HAS_NETWORKX and isinstance(dyn_g, nx.DiGraph):
        return _get_weighted_static_component_nx(dyn_g, baseid_name)

    if HAS_GRAPHTOOL and isinstance(dyn_g, gt.Graph):
        return _get_weighted_static_component_gt(dyn_g, baseid_name, extra_props)

    LOGGER.error('Dynamic graph format not supported')
    return None


def _get_weighted_static_component_gt(dyn_g,  baseid_name='page_id', extra_props=('count_views', )):
    if not HAS_GRAPHTOOL:
        LOGGER.error('Graph-tool not installed, cannot use function _get_weighted_static_component_gt')
        raise ImportError('Graph-tool not installed, cannot use function _get_weighted_static_component_gt')

    unique_baseids = np.unique(dyn_g.vertex_properties[baseid_name].get_array())
    N = len(unique_baseids)
    assert(N > 0)

    g = gt.Graph(directed=True)
    g.gp.component = g.new_graph_property('int', dyn_g.gp.component)
    g.gp.type = g.new_graph_property('int', STATIC_COMP_TYPE)

    # Vertex props
    g.vp.dyn_count = g.new_vertex_property('int', 0)  # node importance
    g.vp.dyn_in_deg = g.new_vertex_property('int', 0)
    g.vp.dyn_out_deg = g.new_vertex_property('int', 0)
    g.vertex_properties[baseid_name] = g.new_vertex_property('int64_t')

    # Extra props
    if extra_props is not None:  # extra prop are doubles
        if isinstance(extra_props, (basestring, unicode)):
            extra_props = (extra_props, )
        for p in extra_props:
            g.vertex_properties[p] = g.new_vertex_property('double')

    # Edge props
    g.ep.count = g.new_edge_property('int', 1)
    g.ep.out_score = g.new_edge_property('double', 0)
    g.ep.in_score = g.new_edge_property('double', 0)
    g.ep.score = g.new_edge_property('double', 0)

    vlist = g.add_vertex(N)
    if N == 1:
        vlist = [vlist]
    baseid_map = dict(zip(unique_baseids, vlist))

    # Compress values on nodes
    for n in dyn_g.vertices():
        base_id = dyn_g.vertex_properties[baseid_name][n]
        g.vp.dyn_count[baseid_map[base_id]] += 1

        if extra_props is not None:  # agglomerate extra props
            for p in extra_props:
                k = dyn_g.vertex_properties[p][n]  # get prop
                g.vertex_properties[p][baseid_map[base_id]] += k  # agg

    # Compress edges
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
        g.vp.dyn_out_deg[src] += 1
        g.vp.dyn_in_deg[tgt] += 1

    # Normalize weights
    for e in g.edges():
        out_score = g.ep.count[e] / float(g.vp.dyn_out_deg[e.source()])
        in_score = g.ep.count[e] / float(g.vp.dyn_in_deg[e.target()])
        g.ep.out_score[e] = out_score
        g.ep.in_score[e] = in_score
        g.ep.score[e] = (out_score + in_score) / 2

    return g


def _get_weighted_static_component_nx(dyn_g,  baseid_name='page_id'):
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


def partition_static_component(g, threshold, baseid_name='page_id', weighted=None,
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


def partition_dynamic_component(dyn_g, static_g, baseid_name='page_id', filter_single=True):
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
    graphs = map(create_graph, np.sort(np.unique(comp.vp.cluster_id.a)))

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


def extract_molecular_components(comp, score_threshold=0.05, baseid_name='page_id',
                                 layer_name='layer', layer_to_ts=None,
                                 extra_props=('count_views', )):
    """Extract molecular components from graphlab dynamic activated component. The score threshold allows
    to prune out low-likelihood in and out edges when the dynamic component is contracted to a static component.
    The parameter layer_ts maps a layer id to a timestamp for better readability.

    """
    dyn_comp = component_to_graphtool(comp, baseid_name, layer_name, layer_to_ts, extra_props)
    static_comp = _get_weighted_static_component_gt(dyn_comp, baseid_name, extra_props)
    static_comp = partition_static_component(static_comp, score_threshold, baseid_name)
    if static_comp is None:  # filter all nodes
        return None

    static_coms = extract_community_subgraphs(static_comp)
    dyn_comp = partition_dynamic_component(dyn_comp, static_comp, baseid_name)
    dyn_coms = extract_community_subgraphs(dyn_comp)

    mols = {'static': static_coms, 'dynamic': dyn_coms}
    return mols


def extract_all_molecular_components(gl_components, score_threshold=0.05, baseid_name='page_id',
                                     layer_name='layer', layer_to_ts=None,
                                     extra_props=('count_views', )):
    """Extract all molecular components sequentially by dumping them on disk and reading them back sequentially.
    Returns static molecular components (optionally dynamic ones) and the number of molecules per components.
    """
    results = {'static': [], 'dynamic': []}
    for i, row in enumerate(gl_components):
        mols = extract_molecular_components(row, score_threshold, baseid_name,
                                            layer_name, layer_to_ts, extra_props)
        if i % 300 == 0:
            LOGGER.INFO('Extracted from {} components'.format(i))
        if mols:
            results['static'].extend(mols['static'])
            results['dynamic'].extend(mols['dynamic'])

    return molecules_to_df(results)


def temporal_shape(g, size=None, normalize=True):
    """Get temporal shape: number of nodes per layer.
    nb_feats: resize the feature vector to a given length.
    If length is bigger than number of layer, values are interpolated linearly
    normalize: the node count per layer as a percentage of the whole distribution.
    """
    layers = g.vp.layer.a
    nb_feats = len(np.unique(layers))
    if not size:
        size = nb_feats

    if size >= nb_feats:
        # Do histogram
        f = pd.Series(np.histogram(layers, size)[0], dtype='float')
        # If values are set to 0 the nb_bins is bigger than the number of layers
        f[f == 0] = None  # set to None for interpolation
        f.interpolate(inplace=True)
        f = f.values
    else:  # desired size if lower than nb_feats
        f = pd.Series(np.histogram(layers, nb_feats)[0], dtype='float')
        # expand to size * nb_feat array
        f.index *= size
        f = f.reindex(pd.Index(np.arange(f.index.max() + 1)))
        # interpolate values for missing indices and average to the selected number of bins
        f.interpolate(inplace=True)
        f = f.values[:-1]  # remove last value to reshape array
        window = len(f) / (nb_feats - 1)
        a = f.reshape(window, -1)
        f = np.mean(a, 1)

    if normalize:
        f /= f.sum()
    return f


def filter_molecules(molecules, indexes):
    idx = indexes
    if isinstance(indexes, (pd.DataFrame, pd.Series)):
        idx = indexes.index.values

    res = defaultdict(list)
    for i in idx:
        res['static'].append(molecules['static'][i])
        res['dynamic'].append(molecules['dynamic'][i])

    return dict(res)


def to_undirected(g):
    h = gt.Graph(directed=False)
    h.add_vertex(g.num_vertices())

    for e in g.edges():
        u = e.source()
        v = e.target()
        if u == v:
            continue

        f = h.edge(u, v)
        if not f:
            h.add_edge(u, v)
    return h


def find_cliques(G):
    """Adaptation of NetworkX find_clique
    https://networkx.github.io/documentation/latest/reference/generated/networkx.algorithms.clique.find_cliques.html
    """
    if G.num_vertices() == 0:
        return

    if G.is_directed():
        G = to_undirected(G)

    adj = {G.vertex_index[u]: {G.vertex_index[v] for v in u.all_neighbours() if v != u} for u in G.vertices()}
    Q = [None]
    all_idx = G.vertex_index.copy().a
    subg = set(all_idx)
    cand = set(all_idx)
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass


def molecules_to_df(molecules, signal_name='', layer_unit=''):
    df = pd.DataFrame(molecules)
    df['signal_name'] = signal_name
    df['layer_unit'] = layer_unit
    return df
