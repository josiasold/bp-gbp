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

namespace bp{

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

        lemon::ListBpGraph::EdgeMap< xt::xarray<long double> > m_cq_current;
        lemon::ListBpGraph::EdgeMap< xt::xarray<long double> > m_qc_current;

        lemon::ListBpGraph::EdgeMap< bool > erased;
        bool erasure_channel;
        
        lemon::ListBpGraph::EdgeMap< bool > converged_qc;
        lemon::ListBpGraph::EdgeMap< bool > converged_cq;

        lemon::ListBpGraph::BlueNodeMap< xt::xarray<long double> > marginals;
        xt::xarray<int> hard_decision;
        xt::xarray<int> syndromes;

        xt::xarray<int> s_0;

        xt::xarray<long double> free_energy;

        std::vector<std::vector<variable_type>> c_factors; //1st vector: 0/1, 2nd vecor: check c, variable_tpye: xframe
        void fill_c_factors();

        void initialize_graph();


        // Update single edges
        void update_edge(const lemon::ListBpGraph::Edge& edge, std::string direction, int iteration);
        void check_to_bit_single(const lemon::ListBpGraph::Edge& message_edge, int iteration);
        void bit_to_check_single(const lemon::ListBpGraph::Edge& message_edge, int iteration);

        // Update all outgoing edges of check/qubit
        void check_update(const lemon::ListBpGraph::RedNode& check, int iteration);
        void qubit_update(const lemon::ListBpGraph::BlueNode& qubit, int iteration);

        // Parallel
        void check_to_bit(int iteration);
        void bit_to_check(int iteration);

        // Serial
        void check_serial_update(int iteration);
        void bit_serial_update(int iteration);

        // Sequential
        void check_sequential_update(int iteration);
        void bit_sequential_update(int iteration);



        void marginals_and_hard_decision(int iteration);
        void marginals_and_hard_decision_serial(int iteration);

        void calculate_free_energy(xt::xarray<int> * s_0,int iteration);


        template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
        std::valarray<T> to_valarray(Tuple &&tuple)
        {
        return std::experimental::apply([](auto &&...elems) {
            return std::valarray<T>{std::forward<decltype(elems)>(elems)...};
        },
                            std::forward<Tuple>(tuple));
        } // https://stackoverflow.com/questions/42494715/c-transform-a-stdtuplea-a-a-to-a-stdvector-or-stddeque

        struct Properties
            {
                int m_max_iter;
                long double m_w;
                long double m_alpha;
                int m_type;
                bool m_return_if_success;
                bool m_only_nonconverged_edges;
            } m_properties;

        int m_schedule_bp;
        int m_reweight_bp;

        bool m_fractional=false;
        bool m_urw=false;
        bool m_memory=false;

    public:
        BpDecoder(xt::xarray<int> H);

        xt::xarray<long double> p_initial;

        void initialize_bp(xt::xarray<long double> p_init, int t_max_iter, long double t_w, long double t_alpha, int t_type, bool t_return_if_success, bool t_only_nonconverged_edges);
        void initialize_bp(xt::xarray<int> * t_s_0);
        void initialize_erasures(xt::xarray<int> * erasures);
        xt::xarray<int> decode_bp(xt::xarray<int> t_s_0);

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

} // end of namespace bp

#endif