/**
 * Copyright (C) 2015 EPFL
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the GPLv2 license. See the LICENSE file for details.
 */

#include <fstream>
#include <string>
#include <vector>
#include <stdint.h>

#include <boost/dynamic_bitset.hpp>

#include <graphlab/flexible_type/flexible_type.hpp>
#include <graphlab/sdk/toolkit_function_macros.hpp>
#include <graphlab/sdk/toolkit_class_macros.hpp>
#include <graphlab/sframe/group_aggregate_value.hpp>
#include <graphlab/sdk/gl_sarray.hpp>
#include <graphlab/sdk/gl_sframe.hpp>
#include <graphlab/sdk/gl_sgraph.hpp>


namespace io = boost::iostreams;

using namespace graphlab;
static size_t NB_LAYERS = 0;  // Pass bitset size as a global variable

typedef boost::dynamic_bitset<> Bitset;

namespace graphlab {

namespace archive_detail {

template <typename OutArcType, typename Block, typename Allocator>
struct serialize_impl<OutArcType, boost::dynamic_bitset<Block, Allocator>, false>
{
    static void exec(OutArcType& arc, const boost::dynamic_bitset<Block, Allocator>& t)
    {
        // Serialize bitset size
        std::size_t size = t.size();
        arc << size;
        // Convert bitset into a vector
        std::vector< Block > v( t.num_blocks() );
        boost::to_block_range( t, v.begin() );
        // Serialize vector
        arc << v;
    }
};

template <typename InArcType, typename Block, typename Allocator>
struct deserialize_impl<InArcType, boost::dynamic_bitset<Block, Allocator>, false>
{
    static void exec(InArcType& iarc,  boost::dynamic_bitset<Block, Allocator>& t)
    {
        std::size_t size;
        iarc >> size;
        t.resize( size );
        // Load vector
        std::vector< Block > v;
        iarc >> v;
        // Convert vector into a bitset
        boost::from_block_range( v.begin() , v.end() , t );
    }
};

} // end namespace archive_detail

flexible_type bitset_to_flexible(const Bitset& b)
{
    // block is unsigned long
    flex_list v(b.num_blocks());
    boost::to_block_range(b, v.begin());
    return v;
}

Bitset bitset_from_flexible(const flexible_type& f)
{
    // Ref cast
    const flex_list& raw = f.to<flex_list>();
    Bitset t;
    for (auto& v: raw) {
        unsigned long b = v.to<unsigned long>();
        t.append(b);
    }
    return t;
}

flexible_type bitset_to_flex_string( const Bitset& b )
{
    std::string t;
    boost::to_string(b, t);
    return t;
}

flexible_type flex_bitset_to_flex_string(const flexible_type& f)
{
    Bitset b(bitset_from_flexible(f));
    std::string ret(bitset_to_flex_string(b));
    return ret;
}

class layer_aggregator: public group_aggregate_value
{
public:

    layer_aggregator()
        : group_aggregate_value()
        , m_bitset(NB_LAYERS)
    {}

    virtual ~layer_aggregator() = default;

    virtual std::string name() const { return "layer_aggregator"; }
    virtual group_aggregate_value* new_instance() const override { return new layer_aggregator(); }

    virtual void add_element_simple(const flexible_type& flex) override
    {
        int layer = flex.to<flex_int>();
        m_bitset.set(layer);
    }

    virtual void combine(const group_aggregate_value& other) override
    {
        logprogress_stream  << "Combine: " << ((const layer_aggregator&)(other)).m_bitset << std::endl;
        m_bitset |= ((const layer_aggregator&)(other)).m_bitset;
    }

    virtual flexible_type emit() const override
    {
        return bitset_to_flexible(m_bitset);
    }

    virtual bool support_type(flex_type_enum type) const override
    {
        return type == flex_type_enum::INTEGER;
    }

    virtual flex_type_enum set_input_types(const std::vector<flex_type_enum>& types) override
    {
         return flex_type_enum::LIST;
    }

    virtual void save(oarchive& oarc) const override
    {
        oarc << m_bitset;
    }

    virtual void load(iarchive& iarc) override
    {
        iarc >> m_bitset;
    }

private:
    Bitset m_bitset;
};


Bitset shift_add_bitsets(Bitset& src, Bitset& tgt)
{
    tgt >>= 1;
    return src & tgt;
}

flex_list expand_causal_edges(const Bitset& bitfield, const flexible_type& base_src,
                              const flexible_type& base_tgt, uint64_t max_id)
{
    Bitset::size_type count = bitfield.count();
    flex_list edges;
    edges.reserve(count * 2);

    // All active layers
    Bitset::size_type cur_layer = bitfield.find_first();

    while (count) {
        uint64_t src = base_src.to<uint64_t>() + (cur_layer * max_id);
        uint64_t tgt = base_tgt.to<uint64_t>() + ((cur_layer + 1) * max_id);
        edges.push_back(src);
        edges.push_back(tgt);
        cur_layer = bitfield.find_next(cur_layer);
        count--;
    }
    return edges;
}

flex_list create_causal_edges(uint64_t max_id,
                              flexible_type src_layers,
                              flexible_type tgt_layers,
                              flexible_type src_id,
                              flexible_type tgt_id)
{
    auto src_data = bitset_from_flexible(src_layers);
    auto tgt_data = bitset_from_flexible(tgt_layers);

    auto active_layers = shift_add_bitsets(src_data, tgt_data);
    auto res = expand_causal_edges(active_layers, src_id, tgt_id, max_id);
    return res;
}

flex_list create_causal_edges(uint64_t max_id,
                              const std::map<std::string, flexible_type>& source,
                              const std::map<std::string, flexible_type>& target)
{
    return create_causal_edges(max_id, source.at("layers"), target.at("layers"),
                               source.at("__id"), target.at("__id"));
}


gl_sframe aggregate_layers(const gl_sframe& sf, const std::string& key_column,
                           const std::string& value_column, size_t nb_layers)
{
    NB_LAYERS = nb_layers;      // Set static global variable ...
    gl_sframe result_sframe = sf.groupby( {key_column},
        {{"layers", aggregate::make_aggregator<layer_aggregator>({value_column})}});
    return result_sframe;
}

flex_list flatten_half_edges(const flexible_type& values, bool source)
{
    flex_list res;  // results
    size_t i = source ? 0: 1;
    flex_list elems = values.to<flex_list>();
    for (; i < elems.size() - 1; i += 2) {
       res.push_back(elems[i]);
    }
    return res;
}

gl_sframe flatten_edges_1pass(const gl_sarray& sa)
{
    flex_list src;
    flex_list tgt;

    for (const auto& v: sa.range_iterator()) {
        flex_list elems = v.to<flex_list>();
        for (size_t i = 0; i < elems.size() - 1; i += 2) {
            src.push_back(elems[i]);
            tgt.push_back(elems[i+1]);
        }
    }
    return gl_sframe({{"src", src}, {"tgt", tgt}});
}

std::map<std::string, size_t> get_column_mapping(const gl_sgraph& g)
{
    // Map column names to integer ... because apply on sframe use vector instead of map
    std::map<std::string, size_t> col_map;
    const gl_sframe& nodes = g.get_vertices();
    for (size_t i = 0; i < nodes.column_names().size(); ++i ) {
        col_map[nodes.column_names()[i]] = i;
    }
    return col_map;
}

gl_sgraph build_sptgraph(gl_sgraph& g, const std::string& base_id_key,
                         const std::string& layer_key, bool with_self_edges)
{
    gl_sgraph h;  // to return

    // Get column mapping for apply on sframe
    auto col_map = get_column_mapping(g);

    // Create SArray of flex_list to store causal edges
    auto edge_count = g.num_edges();
    flex_list l(edge_count, flex_list());
    gl_sarray a(l, flex_type_enum::LIST);
    g.edges().add_column(a, "sp_edges");

    // Find max base id
    auto max_id = g.vertices()["__id"].max();
    // Create causal edges triple apply
    g = g.triple_apply([max_id](edge_triple& triple) -> void {
            triple.edge["sp_edges"] = create_causal_edges(max_id, triple.source, triple.target);
        }, {"sp_edges"}
    );

    auto edges = flatten_edges_1pass(g.edges()["sp_edges"]);

    // Add triple apply edges
    h = h.add_edges(edges, "src", "tgt");

    if (with_self_edges) {
        gl_sarray vedges = g.vertices().apply([max_id, &col_map](const std::vector<flexible_type>& x) {
            return create_causal_edges(max_id, x.at(col_map.at("layers")),
                                       x.at(col_map.at("layers")),
                                       x.at(col_map.at("__id")),
                                       x.at(col_map.at("__id")));
        }, flex_type_enum::LIST);

        edges = flatten_edges_1pass(vedges);
        // Add self edges
        h = h.add_edges(edges, "src", "tgt");
    }

    // Add new fields
    h.vertices()[layer_key] = h.vertices()["__id"].apply([max_id](const flexible_type& x) {
            return (x - 1) / max_id;  // int not float
    }, flex_type_enum::INTEGER);

    // Need new mapping for graph H
    col_map = get_column_mapping(h);
    h.vertices()[base_id_key] = h.vertices().apply([max_id, &col_map, &layer_key](const std::vector<flexible_type>& x) -> uint64_t {
            return x.at(col_map.at("__id")) - (x.at(col_map.at(layer_key)) * max_id);
    }, flex_type_enum::INTEGER);

    return h;
}

} // end namespace graphlab

/* Function registration, export to python */
BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(aggregate_layers, "data", "key_column", "value_column", "nb_layers");
REGISTER_FUNCTION(build_sptgraph, "g", "base_id_key", "layer_key", "with_self_edges");
REGISTER_FUNCTION(flex_bitset_to_flex_string, "f");
END_FUNCTION_REGISTRATION

