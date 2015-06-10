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


namespace graphlab {

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
        // block is unsigned long
        std::vector<flexible_type> v(m_bitset.num_blocks());
        boost::to_block_range(m_bitset, v.begin());

        return v;
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
    boost::dynamic_bitset<> m_bitset;
};


namespace archive_detail {

template <typename OutArcType, typename Block, typename Allocator>
struct serialize_impl<OutArcType, boost::dynamic_bitset<Block, Allocator>, false>
{
    static void exec(OutArcType& arc, const boost::dynamic_bitset<Block, Allocator> & t)
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

} // end namspace archive_detail

gl_sframe aggregate_layers(const gl_sframe& sf, const std::string& key_column,
                           const std::string& value_column, size_t nb_layers)
{
    NB_LAYERS = nb_layers;      // Set static global variable ...
    gl_sframe result_sframe = sf.groupby( {key_column},
        {{"layers", aggregate::make_aggregator<layer_aggregator>({value_column})}});
    return result_sframe;
}

} // end namespace graphlab

/* Function registration, export to python */
BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(aggregate_layers, "data", "key_column", "value_column", "nb_layers");
END_FUNCTION_REGISTRATION

