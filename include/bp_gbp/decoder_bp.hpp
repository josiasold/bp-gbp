#ifndef DECODER_BP_HPP_
#define DECODER_BP_HPP_

#include <lemon/list_graph.h>
#include <lemon/concepts/bpgraph.h>
#include <lemon/adaptors.h>

#include "xtl/xany.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xio.hpp"
#include "xtensor-blas/xlinalg.hpp"

#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xframe/xvariable_view.hpp"

#include <cppitertools/product.hpp>

#include <bp_gbp/la_tools.hpp>

#include <iostream>
#include <valarray>
#include <experimental/tuple>


class BpDecoder
{
    using coordinate_type = xf::xcoordinate<xf::fstring>;
    using dimension_type = xf::xdimension<xf::fstring>;
    using variable_type = xf::xvariable<long double, coordinate_type>;
    using data_type = variable_type::data_type;

    private:
        int max_iterations;

        xt::xarray<int> H;
        int n_q;
        int n_c;
        int n_edges;

        bool is_initialized;

        lemon::ListBpGraph g;

        lemon::ListBpGraph::BlueNodeMap<int> qubit_label;
        lemon::ListBpGraph::RedNodeMap<int> check_label;
        lemon::ListBpGraph::EdgeMap<int> edge_label;

        lemon::ListBpGraph::EdgeMap<int> edge_type;
        lemon::ListBpGraph::EdgeMap< xt::xarray<long double> > m_cq;
        lemon::ListBpGraph::EdgeMap< xt::xarray<long double> > m_qc;

        lemon::ListBpGraph::EdgeMap< bool > erased;
        bool erasure_channel;
        
        lemon::ListBpGraph::EdgeMap< bool > converged_qc;
        lemon::ListBpGraph::EdgeMap< bool > converged_cq;

        lemon::ListBpGraph::BlueNodeMap< xt::xarray<long double> > marginals;
        xt::xarray<int> hard_decision;
        xt::xarray<int> syndromes;

        xt::xarray<long double> free_energy;

        std::vector<std::vector<variable_type>> c_factors; //1st vector: 0/1, 2nd vecor: check c, variable_tpye: xframe
        void fill_c_factors();


        void initialize_graph();
       

        void check_to_bit(xt::xarray<int> * s_0, long double w, int iteration);
        void check_to_bit_fractional(xt::xarray<int> * s_0, long double w, long double alpha, int iteration);
        void bit_to_check(long double w, int iteration);
        void bit_to_check_memory(long double w, int iteration, long double alpha);
        void bit_to_check_urw(long double w, int iteration, long double alpha);

        void bit_serial_update(xt::xarray<int> * s_0, long double w, int iteration);
        
        void bit_serial_update_urw(xt::xarray<int> * s_0, long double w, long double alpha, int iteration);

        void check_serial_update(xt::xarray<int> * s_0, long double w, int iteration);

        void check_update(lemon::ListBpGraph::RedNode& check, int iteration, xt::xarray<int>* s_0, long double w);
        void bit_update(lemon::ListBpGraph::RedNode& check, int iteration, long double w);

        void marginals_and_hard_decision(int iteration);
        void marginals_and_hard_decision_serial(int iteration);
        void marginals_and_hard_decision_fractional(int iteration, long double alpha);
        void marginals_and_hard_decision_urw(int iteration, long double alpha);

        void calculate_free_energy(xt::xarray<int> * s_0,int iteration);


        template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
        std::valarray<T> to_valarray(Tuple &&tuple)
        {
        return std::experimental::apply([](auto &&...elems) {
            return std::valarray<T>{std::forward<decltype(elems)>(elems)...};
        },
                            std::forward<Tuple>(tuple));
        } // https://stackoverflow.com/questions/42494715/c-transform-a-stdtuplea-a-a-to-a-stdvector-or-stddeque

    public:
        BpDecoder(xt::xarray<int> H);

        xt::xarray<long double> p_initial;

        void initialize_bp(xt::xarray<long double> p_init, int max_iter); 
        void initialize_bp();
        void initialize_erasures(xt::xarray<int> * erasures);
        xt::xarray<int> decode_bp(xt::xarray<int> s_0, long double w, long double alpha, int type_bp, bool return_if_success, bool only_non_converged);

        xt::xarray<long double> get_marginals();
        xt::xarray<long double> get_messages();
        xt::xarray<int> get_hard_decisions();
        xt::xarray<int> get_syndromes();
        xt::xarray<bool> get_converged_cq();
        xt::xarray<bool> get_converged_qc();

        xt::xarray<long double> get_free_energy(){return free_energy;};

        xt::xarray<int> get_check_and_qubit(int edge);
        int took_iterations;
};

#endif