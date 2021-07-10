#ifndef GBP_FUNCTIONS_HPP_
#define GBP_FUNCTIONS_HPP_

#include <bp_gbp/la_tools.hpp>
#include <bp_gbp/io_tools.hpp>
#include <bp_gbp/decoder_gbp.hpp>

#include "xtl/xany.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnpy.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xset_operation.hpp"
#include "xtensor-blas/xlinalg.hpp"

#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xframe/xvariable_view.hpp"

#include <cppitertools/product.hpp>

#include <math.h>
#include <valarray>
#include <string>
#include <iostream>
#include <stdio.h>
#include <lemon/list_graph.h>
#include <lemon/concepts/bpgraph.h>
#include <lemon/lgf_reader.h>
#include <lemon/adaptors.h>
#include <lemon/dfs.h>
// #include <lemon/dijkstra.h>

#include <experimental/tuple>



xt::xarray<int> calculate_syndrome(const xt::xarray<int> y, const xt::xarray<int> H);


template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::valarray<T> to_valarray(Tuple &&tuple)
{
  return std::experimental::apply([](auto &&...elems) {
    return std::valarray<T>{std::forward<decltype(elems)>(elems)...};
  },
                    std::forward<Tuple>(tuple));
} // https://stackoverflow.com/questions/42494715/c-transform-a-stdtuplea-a-a-to-a-stdvector-or-stddeque

// template <class E1, class E2>
// inline auto intersect1d(const xt::xexpression<E1>& ar1, const xt::xexpression<E2>& ar2)
// {
//     using value_type = typename E1::value_type;

//     auto unique1 = unique(ar1);
//     auto unique2 = unique(ar2);

//     auto tmp = xt::xtensor<value_type, 1>::from_shape({unique1.size()});

//     auto end = std::set_intersection(
//         unique1.begin(), unique1.end(),
//         unique2.begin(), unique2.end(),
//         tmp.begin()
//     );

//     std::size_t sz = static_cast<std::size_t>(std::distance(tmp.begin(), end));

//     auto result = xt::xtensor<value_type, 1>::from_shape({sz});

//     std::copy(tmp.begin(), end, result.begin());

//     return result;
// };

// struct xarrayCmp
// {
//     bool operator()(const xt::xarray<int> & lhs, const xt::xarray<int> & rhs)
//     {
//         return xt::less(lhs,rhs)();
//     };
// };

class RegionGraph2
{
    using coordinate_type = xf::xcoordinate<xf::fstring>;
    using dimension_type = xf::xdimension<xf::fstring>;
    using variable_type = xf::xvariable<long double, coordinate_type>;
    using data_type = variable_type::data_type;

    friend class GbpDecoder2;

private:
    lemon::ListDigraph rg;
    lemon::ListDigraph::NodeMap<int> layer;
    lemon::ListDigraph::NodeMap<xt::xarray<int>> checks;
    lemon::ListDigraph::NodeMap<xt::xarray<int>> variables;
    // lemon::ListDigraph::NodeMap<xt::xarray<int>> parents;
    // lemon::ListDigraph::NodeMap<xt::xarray<int>> children;
    lemon::ListDigraph::NodeMap<xt::xarray<int>> descendants;
    lemon::ListDigraph::NodeMap<xt::xarray<int>> E;
    // lemon::ListDigraph::NodeMap<xt::xarray<int>> ancestors;
    // lemon::ListDigraph::NodeMap<xt::xarray<int>> shadow;

    lemon::ListDigraph::ArcMap<xt::xarray<int>> N;
    lemon::ListDigraph::ArcMap<xt::xarray<int>> D;

    lemon::ListDigraph::ArcMap<std::vector<variable_type>> message;
    lemon::ListDigraph::ArcMap<xt::xarray<int>> vars_to_marginalize;
    lemon::ListDigraph::ArcMap<std::vector<size_t>> dim_of_vars_to_marg;
    lemon::ListDigraph::ArcMap<variable_type> local_factors_message;
    // lemon::ListDigraph::NodeMap<std::vector<variable_type>> message_before_marg;
    lemon::ListDigraph::ArcMap<variable_type> message_before_marg;
    lemon::ListDigraph::NodeMap<std::vector<variable_type>> belief;
    
    int n_edges;
    int n_v;
    int n_c;
    xt::xarray<int> H;
    xt::xarray<int> qubits;
    xt::xarray<int> checks_list;

    int max_level;

    void construct_rg();

    void make_edges();
    std::set< xt::xarray<int> ,xarrayCmp> find_intersections(int level);

    void print_regiongraph();

    void get_descendants_N_D();

public:
    // RegionGraph2(std::string pathToRG);
    RegionGraph2(xt::xarray<int> H);

    void initializeRG(std::string pathToRG);
};

class GbpDecoder2
{   
    using coordinate_type = xf::xcoordinate<xf::fstring>;
    using dimension_type = xf::xdimension<xf::fstring>;
    using variable_type = xf::xvariable<long double, coordinate_type>;
    using data_type = variable_type::data_type;
    friend class RegionGraph2;

private:
    int max_iter;
    int n_v;
    int n_c;
    int rank_H;
    xt::xarray<int> H;


    std::vector<variable_type> v_factors;
    std::vector<std::vector<variable_type>> c_factors;

    std::vector<std::vector< xt::xarray<long double> > > marginals;
    std::vector< xt::xarray<int> > hard_dec;

    RegionGraph2 RG;

public:
    // GbpDecoder2(std::string pathToPCM, std::string pathToRG, int max_iterations, bool print);
    GbpDecoder2(xt::xarray<int> H, int max_iterations, bool print);

    void initializeGBP(std::string pathToPCM, std::string pathToRG, int max_iterations, bool print);
    void fill_c_factors(std::vector<std::vector<variable_type>> *c_factors, RegionGraph2& RG);
    void fill_v_factors(std::vector<variable_type> *v_factors, xt::xarray<long double> p_initial);
    void init_messages(RegionGraph2& RG,int max_iterations);
    void init_beliefs(RegionGraph2& RG, int max_iterations);

    std::vector<int> make_schedule(const xt::xarray<int> *s_0);

    void update_messages(const xt::xarray<int> *s_0,int iteration, long double w);
    void update_messages_ft(const xt::xarray<int> *s_0,int iteration, long double w);
    void update_beliefs(const xt::xarray<int> *s_0,int iteration);
    void get_marginals_and_hard_dec(int iteration);

    xt::xarray<long double> get_marginals();
    xt::xarray<long double> get_messages();
    xt::xarray<int> get_hard_decisions();

    void prepare(xt::xarray<long double> p_initial, int max_iter);
    xt::xarray<int> decode(xt::xarray<int> s_0,xt::xarray<long double> p_initial,int max_iter, int type, long double w);

    int get_n_v() { return n_v; }
    int get_n_c() { return n_c; }
    int get_rank_H() { return rank_H; }
    xt::xarray<int> get_H() { return H; }

    int took_iterations;
    int n_edges_in_RG;

};

#endif
