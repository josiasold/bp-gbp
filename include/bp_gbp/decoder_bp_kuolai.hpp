#ifndef DECODER_BP_KUOLAI_HPP_
#define DECODER_BP_KUOLAI_HPP_

#include <math.h> // pow

#include <lemon/list_graph.h>
#include <lemon/concepts/bpgraph.h>
#include <lemon/adaptors.h>
#include <lemon/maps.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xoperation.hpp"
#include <bp_gbp/la_tools.hpp>

#include <iostream>

namespace bp{

    class BpDecoderKL
    {
        private:
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

            lemon::ListBpGraph::EdgeMap< long double > m_cq_current;
            lemon::ListBpGraph::EdgeMap< long double > m_qc_current;

            lemon::ListBpGraph::EdgeMap< xt::xarray<long double> > r;

            lemon::ListBpGraph::BlueNodeMap< bool > erased;
            bool erasure_channel;

            lemon::ListBpGraph::EdgeMap< bool > converged_qc;
            lemon::ListBpGraph::EdgeMap< bool > converged_cq;

            lemon::ListBpGraph::BlueNodeMap< xt::xarray<long double> > marginals;
            xt::xarray<int> hard_decision;
            xt::xarray<int> syndromes;

            xt::xarray<int> s_0;

            xt::xarray<long double> free_energy;


            void initialize_graph();
        
            void check_to_bit_single_edge(const lemon::ListBpGraph::Edge& message_edge, int iteration);
            void bit_to_check_single_edge(const lemon::ListBpGraph::Edge& message_edge, int iteration);

            void check_update(const lemon::ListBpGraph::RedNode& check, int iteration);
            void qubit_update(const lemon::ListBpGraph::BlueNode& qubit, int iteration);

            void calculate_r_single_edge(const lemon::ListBpGraph::Edge& edge);

            void check_to_bit(int iteration);
            void bit_to_check(int iteration);

            void check_serial_update(int iteration);
            void bit_serial_update(int iteration);

            void check_sequential_update(int iteration);
            void bit_sequential_update(int iteration);

            void marginals_and_hard_decision(int iteration);

            void calculate_free_energy(int iteration);
            
            struct Properties
            {
                int m_max_iter;
                long double m_w;
                long double m_alpha;
                int m_type;
                bool m_return_if_success;
                bool m_only_nonconverged_edges;
            } m_properties;

        public:
            BpDecoderKL(xt::xarray<int> H);

            xt::xarray<long double> p_initial;

            void initialize_bp(xt::xarray<int> * t_s_0);
            void initialize_bp(xt::xarray<long double> p_init, int t_max_iter, long double t_w, long double t_alpha, int t_type, bool t_return_if_success, bool t_only_nonconverged_edges);
            void initialize_erasures(xt::xarray<int> * erasures);
            xt::xarray<int> decode_bp(xt::xarray<int> t_s_0);

            xt::xarray<long double> get_marginals();
            xt::xarray<long double> get_messages();
            xt::xarray<int> get_hard_decisions();
            xt::xarray<int> get_syndromes();
            xt::xarray<bool> get_converged_cq();
            xt::xarray<bool> get_converged_qc();

            xt::xarray<int> get_check_and_qubit(int edge);

            xt::xarray<long double> get_free_energy();

            int took_iterations;

    };

} // end of namespace bp

#endif