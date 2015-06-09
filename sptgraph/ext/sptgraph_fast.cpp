/**
 * Copyright (C) 2015 EPFL
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the GPLv2 license. See the LICENSE file for details.
 */

#include <string>
#include <vector>
#include <graphlab/flexible_type/flexible_type.hpp>
#include <graphlab/sdk/toolkit_function_macros.hpp>
#include <graphlab/sdk/toolkit_class_macros.hpp>
#include <graphlab/sframe/group_aggregate_value.hpp>
#include <graphlab/sdk/gl_sarray.hpp>
#include <graphlab/sdk/gl_sframe.hpp>
#include <graphlab/sdk/gl_sgraph.hpp>

#include <boost/dynamic_bitset.hpp>

using namespace graphlab;


class layer_aggregator: public group_aggregate_value
{
public:

    layer_aggregator()
        : group_aggregate_value()
    {
    }

    layer_aggregator(const size_t* nb_layers)  // only supports const pointers
        : group_aggregate_value()
        , m_bitset(*nb_layers)
    {
    }

    virtual ~layer_aggregator() = default;

    virtual std::string name() const { return "layer_aggregator"; }
    virtual group_aggregate_value* new_instance() const override { return new layer_aggregator(); }

    virtual void add_element_simple(const flexible_type& flex) override
    {
        int layer = flex.to<flex_int>();
        m_bitset[layer] = 1;
    }

    virtual void combine(const group_aggregate_value& other) override
    {
        m_bitset |= ((const layer_aggregator&)(other)).m_bitset;
    }

    virtual flexible_type emit() const override
    {
        std::string ret;
        boost::to_string(m_bitset, ret);
        return ret;
    }

    virtual bool support_type(flex_type_enum type) const override
    {
        return type == flex_type_enum::INTEGER;
    }

    virtual flex_type_enum set_input_types(const std::vector<flex_type_enum>& types) override
    {
         return flex_type_enum::STRING;
    }

    virtual void save(oarchive& oarc) const override
    {
        std::string ret;
        boost::to_string(m_bitset, ret);
        oarc << ret;
    }

    virtual void load(iarchive& iarc) override
    {
        std::string tmp;
        iarc >> tmp;
        boost::dynamic_bitset<> x(tmp);
        m_bitset = x;
    }

private:
    boost::dynamic_bitset<> m_bitset;
};



gl_sframe aggregate_layers(const gl_sframe& sf, const std::string& key_column, const std::string& value_column, size_t nb_layers)
{
    auto agg = aggregate::make_aggregator<layer_aggregator>({value_column}, nb_layers);
    gl_sframe result_sframe = sf.groupby( {key_column}, {{"layers", agg}});
    return result_sframe;
}

BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(aggregate_layers, "data", "key_column", "value_column", "nb_layers");
END_FUNCTION_REGISTRATION
