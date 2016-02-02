
from io import BytesIO
from collections import defaultdict
from graph_tool import all as gt
from sptgraph import utils

import operator
import traceback
import os
import sys
import logging
import pandas as pd
LOGGER = logging.getLogger('sptgraph')
LOGGER.setLevel(logging.INFO)


STATIC_COMP_TYPE = 1
DYN_COMP_TYPE = 2
ALL_COMP_TYPE = STATIC_COMP_TYPE + DYN_COMP_TYPE


HAS_SQL = False
try:
    from sqlalchemy import create_engine, Table, Column, String, LargeBinary, MetaData, and_
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    Base = declarative_base()
    HAS_SQL = True
except ImportError:
    LOGGER.warning('sqlalchemy not found, cannot use DB to store components')

if HAS_SQL:
    class ComponentDAO(object):
        def __init__(self, db_dir, force_create=False):
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
            self.db_path = os.path.join(db_dir, 'components.db')

            if force_create and os.path.exists(self.db_path):
                os.remove(self.db_path)

            self.engine = create_engine('sqlite:///' + self.db_path)
            Base.metadata.create_all(self.engine)
            self.session = sessionmaker(bind=self.engine)()

        @staticmethod
        def _build_comp(sta_g, dyn_g, signal_name='', layer_unit=''):
            mol_id = get_molecule_id(sta_g, postfix=signal_name)
            return Component(mol_id=mol_id, signal_name=signal_name,
                             layer_unit=layer_unit,
                             static=ComponentDAO.serialize_graph(sta_g),
                             dynamic=ComponentDAO.serialize_graph(dyn_g))

        def add(self, sta_g, dyn_g, signal_name='', layer_unit=''):
            comp = self._build_comp(sta_g, dyn_g, signal_name, layer_unit)
            try:
                comp = self.session.merge(comp)
                self.session.add(comp)
                self.session.commit()
            except Exception as e:
                LOGGER.error("Exception: {}".format(e.message))
                self.session.rollback()

        def add_all(self, molecules, signal_name='', layer_unit='', batch_size=1000):
            def commit(res):
                try:
                    self.session.add_all(res)
                    self.session.commit()
                except Exception as e:
                    LOGGER.error("Exception: {}".format(e.message))
                    self.session.rollback()
                    return False
                return True

            mols = molecules
            if isinstance(molecules, pd.DataFrame):
                mols = molecules[['static', 'dynamic']].to_dict()

            res = []
            for i in xrange(len(mols['static'])):
                c = self._build_comp(mols['static'][i], mols['dynamic'][i], signal_name, layer_unit)
                c = self.session.merge(c)
                res.append(c)
                if len(res) > batch_size:
                    commit(res)
                    del res
                    res = []

            if len(res) > 0:
                commit(res)

        @staticmethod
        def serialize_graph(g):
            buf = BytesIO()
            g.save(buf)
            return buf.getvalue()

        @staticmethod
        def deserialize_graph(bytestring):
            buf = BytesIO()
            buf.write(bytestring)
            buf.seek(0)
            return gt.load_graph(buf)

        def query_all(self, signal_name='', layer_unit=''):
            query = self.session.query(Component)
            if layer_unit:
                query = query.filter(Component.layer_unit == layer_unit)
            if signal_name:
                query = query.filter(Component.signal_name == signal_name)
            query = query.order_by(Component.mol_id)
            return query

        def load_all(self, signal_name='', layer_unit=''):
            query = self.query_all(signal_name, layer_unit)
            data = []
            for comp in query:
                d = {'mol_id': comp.mol_id,
                     'signal_name': comp.signal_name,
                     'layer_unit': comp.layer_unit,
                     'static': ComponentDAO.deserialize_graph(comp.static),
                     'dynamic': ComponentDAO.deserialize_graph(comp.dynamic)}
                data.append(d)
            if not data:
                return None
            return pd.DataFrame(data)

        def count(self, signal_name='', layer_unit=''):
            return self.query_all(signal_name, layer_unit).count()


    class Component(Base):
        __tablename__ = 'component'
        mol_id = Column(String, primary_key=True)
        signal_name = Column(String)
        layer_unit = Column(String)
        static = Column(LargeBinary)
        dynamic = Column(LargeBinary)

        def __repr__(self):
           return "<Component(mol_id='%s', layer_unit='%s', signal_name='%s')>" \
                  % (self.mol_id, self.layer_unit, self.signal_name)


    def save_molecules_db(out_dir, mols):
        dao = ComponentDAO(out_dir)
        dao.add_all(mols)
        LOGGER.info('Saved {} molecule pairs'.format(len(mols['static'])))


def get_molecule_id(g, i=0, postfix=''):
    name = u'M#'
    if 'type' in g.graph_properties:
        name += 'st_' if g.gp.type == STATIC_COMP_TYPE else 'dyn_'

    if 'component' in g.graph_properties:
        name += str(g.gp.component)
    else:
        name += str(i)
    if 'cluster_id' in g.graph_properties:
        name += '_' + str(g.gp.cluster_id)

    if 'com_id' in g.graph_properties:
        name += '_' + str(g.gp.com_id)

    if 'layer_unit' in g.graph_properties:
        name += '_' + str(g.gp.layer_unit)

    if postfix:
        name += '_' + postfix
    return name


def save_gt_component(c, out_dir, i=0, verbose=False):
    name = get_molecule_id(c, i)[2:]
    path = str(os.path.join(out_dir, name + '.gt'))
    c.save(path)
    if verbose:
        LOGGER.info('Wrote {}'.format(path))


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
        if name.startswith('st') and comp_type in (STATIC_COMP_TYPE, ALL_COMP_TYPE):
            comps['static'].append((name, gt.load_graph(path)))

        if name.startswith('dyn') and comp_type in (DYN_COMP_TYPE, ALL_COMP_TYPE):
            comps['dynamic'].append((name, gt.load_graph(path)))

    res = {'static': None, 'dynamic': None}
    # Sort components by name
    for k in res:
        if k in comps:
            res[k] = map(lambda x: x[1], sorted(comps[k], key=operator.itemgetter(0)))

    return res


