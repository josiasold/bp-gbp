#ifndef CONSTRUCTOR_GBP_HPP_
#define CONSTRUCTOR_GBP_HPP_

#include <xtensor/xexpression.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xset_operation.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmath.hpp>

#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xframe/xvariable_view.hpp"

#include <cppitertools/product.hpp>

#include <lemon/list_graph.h>
#include <lemon/graph_to_eps.h>
#include <lemon/dfs.h>
#include <lemon/adaptors.h>

#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <fstream>

#include <bp_gbp/decoder_bp.hpp>
#include <bp_gbp/decoder_bp_kuolai.hpp>

using namespace xt::placeholders;

template <class E1, class E2>
inline auto intersect1d(const xt::xexpression<E1>& ar1, const xt::xexpression<E2>& ar2)
{
    using value_type = typename E1::value_type;

    auto unique1 = unique(ar1);
    auto unique2 = unique(ar2);

    auto tmp = xt::xtensor<value_type, 1>::from_shape({unique1.size()});

    auto end = std::set_intersection(
        unique1.begin(), unique1.end(),
        unique2.begin(), unique2.end(),
        tmp.begin()
    );

    std::size_t sz = static_cast<std::size_t>(std::distance(tmp.begin(), end));

    auto result = xt::xtensor<value_type, 1>::from_shape({sz});

    std::copy(tmp.begin(), end, result.begin());

    return result;
};

template <class E1, class E2>
inline auto union1d(const xt::xexpression<E1>& ar1, const xt::xexpression<E2>& ar2)
{
    using value_type = typename E1::value_type;

    auto unique1 = unique(ar1);
    auto unique2 = unique(ar2);

    auto tmp = xt::xtensor<value_type, 1>::from_shape({unique1.size()+unique2.size()});

    auto end = std::set_union(
        unique1.begin(), unique1.end(),
        unique2.begin(), unique2.end(),
        tmp.begin()
    );

    std::size_t sz = static_cast<std::size_t>(std::distance(tmp.begin(), end));

    auto result = xt::xtensor<value_type, 1>::from_shape({sz});

    std::copy(tmp.begin(), end, result.begin());

    return result;
};

// struct xarrayCmp
// {
//     public:
//         bool operator()(const xt::xarray<int> & lhs, const xt::xarray<int> & rhs)
//         {
//             // xt::xarray<int> setdiff = xt::setdiff1d(lhs,rhs);
//             if (xt::all(xt::equal(lhs,rhs)))
//             {
//                 return false;
//             }
//             else
//             {
//                 return true;
//             }
//             // return xt::equal(lhs,rhs)();
//         };
// };

struct xarrayHash 
{
    public:
        size_t operator()(const xt::xarray<int> & xarray) const {
            std::stringstream ststr;
            ststr << xarray;
            return std::hash<std::string>()(ststr.str());
        }
};

class RegionGraph
{
    using coordinate_type = xf::xcoordinate<xf::fstring>;
    using dimension_type = xf::xdimension<xf::fstring>;
    using variable_type = xf::xvariable<long double, coordinate_type>;
    using data_type = variable_type::data_type;

    friend class GbpDecoder;
    private:
        xt::xarray<int> H;
        int n_c;
        int n_q;
        xt::xarray<int> checks;
        xt::xarray<int> qubits;

        int n_edges;
        int n_regions;

        int max_level;

        lemon::ListDigraph rg;
        lemon::ListDigraph::NodeMap< xt::xarray<int> > region_qubits;
        lemon::ListDigraph::NodeMap< xt::xarray<int> > region_checks;
        lemon::ListDigraph::NodeMap<int> region_level;

        lemon::ListDigraph::NodeMap< std::vector<variable_type> > belief;
        lemon::ListDigraph::NodeMap<variable_type> belief_base;

        lemon::ListDigraph::ArcMap< xt::xarray<int> > vars_to_marginalize;
        lemon::ListDigraph::ArcMap< std::vector<size_t> > dim_of_vars_to_marg;
        
        lemon::ListDigraph::ArcMap< std::vector<variable_type> > message;
        lemon::ListDigraph::ArcMap< bool > edge_converged;
        lemon::ListDigraph::NodeMap< bool > region_converged;
        lemon::ListDigraph::ArcMap<variable_type> message_base;
        
        lemon::ListDigraph::NodeMap<int> counting_number;
        lemon::ListDigraph::NodeMap< xt::xarray<int> > ancestors;
        lemon::ListDigraph::NodeMap< xt::xarray<int> > descendants;
        lemon::ListDigraph::ArcMap< xt::xarray<int> > N;
        lemon::ListDigraph::ArcMap< xt::xarray<int> > D;



        void construct_rg(int n_checks_per_r0, int rg_type);
        void construct_rg(int n_checks_per_r0, xt::xarray<int> check_list, int rg_type);
        // std::set< xt::xarray<int> ,xarrayCmp>
        std::unordered_set< xt::xarray<int>, xarrayHash > find_intersections(int level);
        void make_edges();
        void get_descendants_N_D();
        void get_ancestors_counting_numbers();
        void print_regiongraph(std::string suffix);

    public:
        RegionGraph(xt::xarray<int> H, int n_checks_per_r0, int rg_type);
        RegionGraph(xt::xarray<int> H, int n_checks_per_r0, xt::xarray<int> check_list, int rg_type);

};

template <typename BPD>
xt::xarray<int> get_H_sub(BPD* bpDecoder, xt::xarray<int> H, xt::xarray<int> s, xt::xarray<int> * s_sub, xt::xarray<int> * c_indices, xt::xarray<int> * q_indices, int pauli, bool print)
{
    if (print)
    {
        std::cout << "get_H_sub, pauli = " << pauli << "\n";
    }

    unsigned int n_c = H.shape(0);
    unsigned int n_q = H.shape(1);

    xt::xarray<long double> marginals = bpDecoder->get_marginals();
    xt::xarray<long double> messages = bpDecoder->get_messages();
    xt::xarray<int> hard_decisions = bpDecoder->get_hard_decisions();
    xt::xarray<int> syndromes = bpDecoder->get_syndromes();
    xt::xarray<bool> converged_cq = bpDecoder->get_converged_cq();
    xt::xarray<bool> converged_qc = bpDecoder->get_converged_qc();

    std::vector<int> qc_not_converged;
    std::vector<int> cq_not_converged;

    for (int e = 0 ; e < converged_cq.shape(0); e++)
    {
        if (converged_cq(e) == false)
        {
            cq_not_converged.push_back(e);
        }
    }
    xt::xarray<int> cq_edges_not_converged = xt::adapt(cq_not_converged);


    for (int e = 0 ; e < converged_qc.shape(0); e++)
    {
        if (converged_qc(e) == false)
        {
            qc_not_converged.push_back(e);
        }
    }
    xt::xarray<int> qc_edges_not_converged = xt::adapt(qc_not_converged);
    

    xt::xarray<int> abs_diffs_s = xt::abs(xt::diff(syndromes,1,0));
    xt::xarray<int> diffs_s_sum = xt::sum(xt::view(abs_diffs_s,xt::range(-20,_),xt::all()),0);
    xt::xarray<int> syndromes_changing = xt::from_indices(xt::argwhere(diffs_s_sum > 0));

    // if (print == true)
    // {
        // xt::xarray<int> tmp = xt::flatten(syndromes_changing);
        // std::cout << "syndromes_changing = " << tmp << "\n  |sc| = " << xt::adapt(tmp.shape()) << std::endl;
    // }


    // qubits and checks involved
    std::set<int> checks_involved;
    std::set<int> qubits_involved;

    // if (cq_edges_not_converged.size() > 0)
    // {
    //     for (auto it = cq_edges_not_converged.begin(); it != cq_edges_not_converged.end(); ++it)
    //     // for (auto it = edges_changing_qc.begin(); it != edges_changing_qc.end(); ++it)
    //     {
    //         xt::xarray<int> cq = bpDecoder->get_check_and_qubit(*it);

    //         if (H(cq(0),cq(1)) == pauli)
    //         {
    //             checks_involved.insert(cq(0));
    //             qubits_involved.insert(cq(1));
    //         }
            
    //     }
    // }
    if (qc_edges_not_converged.size() > 0)
    {
        for (auto it = qc_edges_not_converged.begin(); it != qc_edges_not_converged.end(); ++it)
        // for (auto it = edges_changing_qc.begin(); it != edges_changing_qc.end(); ++it)
        {
            xt::xarray<int> cq = bpDecoder->get_check_and_qubit(*it);

            if (H(cq(0),cq(1)) == pauli)
            {
                checks_involved.insert(cq(0));
                qubits_involved.insert(cq(1));
            }
            
        }
    }
    if ((cq_edges_not_converged.size() == 0) && (qc_edges_not_converged.size() == 0))
    {
         for (auto it = syndromes_changing.begin(); it != syndromes_changing.end(); ++it)
        {
            checks_involved.insert(*it);
            xt::xarray<int>row_of_H = xt::row(H,*it);
            xt::xarray<int> qubits_in_support = xt::flatten(xt::from_indices(xt::argwhere(row_of_H > 0)));
            for (auto q = qubits_in_support.begin(); q != qubits_in_support.end(); ++q)
            {
                qubits_involved.insert(*q);
            }
        }
    }

    // std::cout << "checks_involved (" << checks_involved.size() << ") : \n";
    // for (auto it = checks_involved.begin(); it != checks_involved.end(); ++it)
    // {
    //     std::cout << *it << ",";
    // }
    // std::cout << std::endl;
    // std::cout << "qubits_involved (" << qubits_involved.size() << ") : \n";
    // for (auto it = qubits_involved.begin(); it != qubits_involved.end(); ++it)
    // {
    //     std::cout << *it << ",";
    // }
    // std::cout << std::endl;

    std::vector<int> ci_v(checks_involved.begin(), checks_involved.end()); 
    std::vector<int> qi_v(qubits_involved.begin(), qubits_involved.end()); 

    xt::xarray<int> ci = xt::adapt(ci_v);
    xt::xarray<int> qi = xt::adapt(qi_v);


    unsigned int n_qi = qi.size();
    unsigned int n_ci = ci.size();

    s_sub->resize({n_ci});

    c_indices->resize({n_ci});
    q_indices->resize({n_qi});

    *c_indices = ci;
    *q_indices = qi;


    // H_sub
    xt::xarray<int> H_sub_c;
    H_sub_c.resize({n_ci,n_q});

    H_sub_c = xt::view(H,xt::keep(ci),xt::all());

    xt::xarray<int> H_sub;
    H_sub.resize({n_ci,n_qi});
    H_sub = xt::view(H_sub_c,xt::all(),xt::keep(qi));

    for (int c = 0; c < n_ci; c++)
    {
        if (s(ci(c)) != 0)
        {
            s_sub->at(c) = 1;
        }
        else
        {
            s_sub->at(c) = 0;
        }
        for (int q = 0; q < n_qi; q++)
        {
            if (H_sub(c,q) == pauli)
            {
                H_sub(c,q) = 1;
            }
        }
    }

    return H_sub;
};

#endif