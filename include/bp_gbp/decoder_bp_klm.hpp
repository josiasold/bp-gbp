#ifndef DECODER_BP_KLM_HPP_
#define DECODER_BP_KLM_HPP_

#include <math.h> // pow

#include <lemon/list_graph.h>
#include <lemon/concepts/bpgraph.h>
#include <lemon/adaptors.h>
#include <lemon/maps.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xindex_view.hpp"

#include <bp_gbp/la_tools.hpp>

#include <iostream>

class BpDecoderKLM
{
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

        lemon::ListBpGraph::EdgeMap< bool > converged_qc;
        lemon::ListBpGraph::EdgeMap< bool > converged_cq;

        lemon::ListBpGraph::BlueNodeMap< xt::xarray<long double> > marginals;
        xt::xarray<int> hard_decision;
        xt::xarray<int> syndromes;

        xt::xarray<long double> free_energy;


        void initialize_graph();
       

        void check_to_bit(xt::xarray<int> * s_0, int iteration,long double w, long double alpha);

        void bit_to_check_and_hd(xt::xarray<int> * s_0, int iteration,long double w, long double alpha);

        void bit_serial(xt::xarray<int> * s_0, int iteration,long double w, long double alpha);
        
        void marginals_and_hard_decision(int iteration, long double alpha);
        void calculate_free_energy(int iteration);

    public:
        BpDecoderKLM(xt::xarray<int> H);
        xt::xarray<long double> p_initial;

        void initialize_bp(xt::xarray<long double> p_init, int max_iter); 
        void initialize_bp();
        xt::xarray<int> decode_bp(xt::xarray<int> s_0,long double w, long double alpha, int type_bp, bool return_if_success);

        xt::xarray<long double> get_marginals();
        xt::xarray<long double> get_messages();
        xt::xarray<int> get_hard_decisions();
        xt::xarray<int> get_syndromes();
        xt::xarray<bool> get_converged_cq();
        xt::xarray<bool> get_converged_qc();

        xt::xarray<int> get_check_and_qubit(int edge);

        xt::xarray<long double> get_free_energy();

};

#endif