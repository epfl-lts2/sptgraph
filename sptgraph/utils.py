# -*- coding: utf-8 -*-

import os
import datetime
import graphlab as gl
import pandas as pd
import ctypes

# Use the built-in version of scandir/walk if possible, otherwise
# use the scandir module version
try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk


def networkx_to_graphlab(g):
    p = gl.SGraph()
    # Add nodes
    p = p.add_vertices(map(lambda x: gl.Vertex(x[0], attr=x[1]), g.nodes(data=True)))
    # Add edges
    p = p.add_edges(map(lambda x: gl.Edge(x[0], x[1], attr=x[2]), g.edges(data=True)))
    if not g.is_directed():
        p = p.add_edges(map(lambda x: gl.Edge(x[1], x[0], attr=x[2]), g.edges(data=True)))
    return p


def to_sarray_dt(series):
    """series is a pandas Series object, with dtype coercable to datetime.datetime"""
    dt_list = pd.to_datetime(series).astype(datetime.datetime).tolist()
    dt_list = [x if isinstance(x, datetime.datetime) else None for x in dt_list]
    sa = gl.SArray(dt_list, dtype=datetime.datetime, ignore_cast_failure=False)
    return sa


def from_pandas(df):
    """Issue importing datetime64[ns] Series into graphlab"""
    msk = df.dtypes != '<M8[ns]'
    sf = gl.SFrame(df.loc[:, msk])  # no issues (normally)
    msk = df.dtypes == '<M8[ns]'
    for i in df.columns[msk]:
        v = to_sarray_dt(df.loc[:, i])
        sf.add_column(v, i)
    return sf


def reform_layer_int_from_blocks(blocks):
    res = ''
    for b in blocks[::-1]:  # reverse the order
        res += bin(ctypes.c_ulong(b).value)[2:]
    res = '0b' + res
    return int(res, 2)


def to_multi_dict(items):
    def insert(d, kv):
        k, v = kv
        d.setdefault(k, []).append(v)
        return d
    return reduce(insert, [{}] + items)


def list_dir(dirpath, fullpath=False):
    """List files by name..."""
    paths = []
    for fn in scandir(dirpath):
        name = fn.name
        if fullpath:
            name = os.path.join(dirpath, name)
        paths.append(name)
    return paths

