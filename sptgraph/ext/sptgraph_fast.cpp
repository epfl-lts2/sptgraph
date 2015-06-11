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
    std::vector<flexible_type> v(b.num_blocks());
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


gl_sframe aggregate_layers(const gl_sframe& sf, const std::string& key_column,
                           const std::string& value_column, size_t nb_layers)
{
    NB_LAYERS = nb_layers;      // Set static global variable ...
    gl_sframe result_sframe = sf.groupby( {key_column},
        {{"layers", aggregate::make_aggregator<layer_aggregator>({value_column})}});
    return result_sframe;
}


Bitset shift_add_bitsets(Bitset& src, Bitset& tgt)
{
    tgt >>= 1;
    return src & tgt;
}

// Triple apply function to create all causal edges
void create_causal_edges(edge_triple& triple)
{
    auto src_data = bitset_from_flexible(triple.source["layers"]);
    auto tgt_data = bitset_from_flexible(triple.target["layers"]);

    auto res = bitset_to_flexible(shift_add_bitsets(src_data, tgt_data));

    // Image seems to pass as a valid type for storage ...
    //    triple.edge["sp_edges"] = res;
//    logprogress_stream <<  g.edges()["sp_edges"];
    // TODO unpack
}

gl_sgraph build_sptgraph(gl_sgraph& g, const std::string& base_id_key,
                         const std::string& layer_key, bool with_self_edges)
{
//    g.vertices().add_column(flex_image(), "sp_edges");
    g.edges().add_column(flex_image(), "sp_edges");
    g = g.triple_apply(create_causal_edges, {"sp_edges"});
    return g;
}

} // end namespace graphlab



/* Function registration, export to python */
BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(aggregate_layers, "data", "key_column", "value_column", "nb_layers");
REGISTER_FUNCTION(build_sptgraph, "g", "base_id_key", "layer_key", "with_self_edges");
REGISTER_FUNCTION(flex_bitset_to_flex_string, "f");
END_FUNCTION_REGISTRATION

