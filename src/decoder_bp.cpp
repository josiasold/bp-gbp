#include <bp_gbp/decoder_bp.hpp>


BpDecoder::BpDecoder(xt::xarray<int> H) : H(H), qubit_label(g), check_label(g),edge_label(g), edge_type(g), m_cq(g), m_qc(g),converged_cq(g), converged_qc(g), marginals(g)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    initialize_graph();
    fill_c_factors();
    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);
}

void BpDecoder::initialize_graph()
{
    for (size_t c = 0; c < n_c; c++)
    {
        lemon::ListBpGraph::RedNode new_check = g.addRedNode();
        check_label[new_check] = c;
    }

    for (size_t q = 0; q < n_q; q++)
    {
        lemon::ListBpGraph::BlueNode new_qubit = g.addBlueNode();
        qubit_label[new_qubit] = q;
    }

    n_edges = 0;

    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        int c = check_label[check];

        for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
        {
            int q = qubit_label[qubit];
             if (H(c,q) != 0)
            {
                lemon::ListBpGraph::Edge new_edge = g.addEdge(check,qubit);
                edge_type[new_edge] = H(c,q);
                edge_label[new_edge] = n_edges;
                // if ((q < 16) && (c<12)) std::cout << " edge " << n_edges << "\n";
                
                n_edges++;
            }
        }
    }


    std::cout << "initialized graph with n_c = " << n_c << " n_q = " << n_q << std::endl; 
}

void BpDecoder::fill_c_factors()
{
    c_factors.resize(2);
  for (int i = 0; i < 2; i++)
    c_factors[i].resize(n_c);

  for (int check = 0; check < n_c; check++)
  {

      xt::xarray<int>row = xt::row(H,check);
      xt::xarray<int> qubits_in_support = xt::col(xt::from_indices(xt::argwhere(row > 0)),0);
      int n_qbuits_in_region = qubits_in_support.size();

      variable_type::coordinate_map coord_map;
      dimension_type::label_list dim_list = {};


      for (int i = 0; i < n_qbuits_in_region; i++)
      {
        std::string v_name = std::to_string(qubits_in_support(i));
        const char *cstr = v_name.c_str();
        coord_map[cstr] = xf::axis({"0", "1"});
        dim_list.push_back(cstr);
      }

      std::vector<int> shape(n_qbuits_in_region, 2);
      xt::xarray<long double> all_zeros;
      all_zeros.resize(shape);
      all_zeros.fill(0);
      xt::xarray<long double> all_ones;
      all_ones.resize(shape);
      all_ones.fill(0);



      std::vector<int> v1{0, 1};
      const int nv = 7;
      for (auto &&t : iter::product<nv>(v1))
      {
        std::valarray<int> indices(n_qbuits_in_region);
        indices = to_valarray(t);
        std::valarray<int> bar = indices[std::slice(0, n_qbuits_in_region, 1)];
        std::vector<int> shape = {n_qbuits_in_region};
        std::vector<size_t> fo(bar.size());
        std::copy(begin(bar), end(bar), begin(fo));

        if (bar.sum() % 2 == 0)
        {
          all_zeros[fo] = 1;
        }
        else if (bar.sum() % 2 == 1)
        {
          all_ones[fo] = 1;
        }
      }
      variable_type c_factor_0(all_zeros, coord_map, dim_list);
      variable_type c_factor_1(all_ones, coord_map, dim_list);

      c_factors[0][check] = c_factor_0;
      c_factors[1][check] = c_factor_1;

    // }
  }
}

void BpDecoder::initialize_bp(xt::xarray<long double> p_init, int max_iter)
{
    p_initial = p_init;
    max_iterations = max_iter;
    hard_decision =  xt::zeros<int>({max_iter+1,n_q});
    syndromes = xt::zeros<int>({max_iter+1,n_c});
    free_energy = xt::zeros<long double>({max_iter});

    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({max_iter,4});
        m_qc[edge] = xt::zeros<long double>({max_iter,4});
        
       if ((qubit_label[g.blueNode(edge)] == 8) || (qubit_label[g.blueNode(edge)] == 11) || (qubit_label[g.blueNode(edge)] == 12))
        {
            xt::xarray<long double> tmp = {0.98,0.008,0.006,0.006};
            xt::row(m_qc[edge],0) = tmp;
        }
        else
        {
            xt::row(m_qc[edge],0) = p_initial;
        }
        // xt::row(m_qc[edge],0) = p_intial;




        // m_qc[edge](0) = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;
    }
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({max_iter,4});
    }
    is_initialized = true;
}

void BpDecoder::initialize_bp()
{
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        if ((qubit_label[g.blueNode(edge)] == 8) || (qubit_label[g.blueNode(edge)] == 11) || (qubit_label[g.blueNode(edge)] == 12))
        {
            xt::xarray<long double> tmp = {0.98,0.008,0.006,0.006};
            xt::row(m_qc[edge],0) = tmp;
        }
        else
        {
            xt::row(m_qc[edge],0) = p_initial;
        }
        // xt::row(m_qc[edge],0) = p_intial;
    }
}

xt::xarray<int> BpDecoder::decode_bp(xt::xarray<int> s_0, long double w)
{
    if (is_initialized == false)
    {
        std::cerr << "BP not initialized, please run initialize_bp(xt::xarray<long double> p_init)" << std::endl;
    }

    initialize_bp();

    int took_iter = max_iterations;
    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        check_to_bit(&s_0, w, iteration);

        bit_to_check(w, iteration);

        marginals_and_hard_decision(iteration);

        xt::xarray<int> s = xt::ones_like(s_0);
        xt::xarray<int> hd =  xt::row(hard_decision,iteration);

        gf4_syndrome(&s, &hd, &H);

        calculate_free_energy(&s_0, iteration);

        xt::row(syndromes,iteration) = s;
        
        if (s == s_0)
        {
            took_iter = iteration;
            return hd;
        }
    }
    
    xt::xarray<int> hd =  xt::row(hard_decision,took_iter);
    xt::xarray<int> hd_ret =  xt::row(hard_decision,took_iter);
    int min_weight_s = hamming_weight(s_0);
    if (xt::sum(hd)() == 0)
    {
        for (int i = max_iterations; i>=0; i--)
        {
            hd =  xt::row(hard_decision,i);
            if (xt::sum(hd)() != 0)
            {
                xt::xarray<int> s_min_res = s_0^xt::row(syndromes,i);
                if (hamming_weight(s_min_res) < min_weight_s)
                {
                    min_weight_s = hamming_weight(s_min_res);
                    hd_ret = hd;
                }
            }
        }
    }

    return hd_ret;
}

void BpDecoder::check_to_bit(xt::xarray<int> * s_0, long double w, int iteration)
{
    xt::xarray<long double> FT = xt::ones<long double>({4,4});
    FT(1,1) = -1;FT(1,3) = -1;FT(2,2) = -1;FT(2,3) = -1;FT(3,1) = -1;FT(3,2) = -1;

    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        xt::xarray<long double> message = {1.0,1.0,1.0,1.0};
        xt::xarray<long double> permuted_message = {1.0,1.0,1.0,1.0};
        xt::xarray<int> permutation = {0,1,2,3};

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            message.fill(1.0);
            permuted_message.fill(1.0);
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    for (int i = 0; i<4; i++)
                    {
                        int permutation_index = gf4_mul(edge_type[other_edge],gf4_conj(i));
                        permutation(i) = permutation_index;
                        permuted_message(i) = m_qc[other_edge](iteration-1,permutation_index);
                    }
                    xt::xarray<long double> tmp = xt::linalg::dot(FT,permuted_message);
                    message *= tmp;
                }
            }
            xt::xarray<long double> tmp = xt::linalg::dot(FT,message);
            message = tmp;
            // std::cout << "message = " << message << "\n";
            // message /= xt::sum(message);
            xt::xarray<long double> message_tmp = {1.0,1.0,1.0,1.0};

            for (int p = 0; p < 4; p++)
            {
                int i1 = (2*s_0->at(check_label[check])) ^ gf4_mul(edge_type[message_edge],gf4_conj(p));
                int i2 = i1^1;
                message_tmp(p) = 0.5 * (message(i1) + message(i2));
            }
            message_tmp /= xt::sum(message_tmp);

            xt::row(m_cq[message_edge],iteration) = (1-w) * xt::row(m_cq[message_edge],iteration-1)+ w * message_tmp;
            // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
            if ( xt::row(m_cq[message_edge],iteration) ==  xt::row(m_cq[message_edge],iteration-1))
            {
                converged_cq[message_edge] = true;
            }
        }
    }
    return;
}

void BpDecoder::bit_to_check(long double w, int iteration)
{
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = xt::ones<long double>({4});

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,qubit); message_edge != lemon::INVALID; ++message_edge)
        {
            q = p_initial;
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,qubit); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    q *= xt::row(m_cq[other_edge],iteration);
                }
            }
            q /= xt::sum(q);
            xt::row(m_qc[message_edge],iteration) =  (1-w) * xt::row(m_qc[message_edge],iteration-1)+ w * q;
            // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
            if (xt::row(m_qc[message_edge],iteration) == xt::row(m_qc[message_edge],iteration-1))
            {
                converged_qc[message_edge] = true;
            }
        }
    }
    return;
}

void BpDecoder::marginals_and_hard_decision(int iteration)
{
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = p_initial;
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            q *= xt::row(m_cq[incoming_edge],iteration);
        }
        q /= xt::sum(q);
        xt::row(marginals[qubit],iteration) = q;
        // m = q;
        int hd =  xt::argmax(q,0)();
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}

void BpDecoder::calculate_free_energy(xt::xarray<int> * s_0,int iteration)
{
    // long double average_energy = 0;
    // long double entropy = 0;

    // for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    // {
    //     xt::xarray<long double> belief_check = c_factors[s_0.at(check_label[check])][check_label[check]];
    //     for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,check); incoming_edge != lemon::INVALID; ++incoming_edge)
    //     {
            
    //         belief_check *= m_qc[incoming_edge](iteration);
    //         for (int p = 0; p < 4; p++)
    //         {
    //             average_energy += m_qc[incoming_edge](iteration,p) * log(m_cq[incoming_edge](iteration,p));
    //             entropy += m_qc[incoming_edge](iteration,p) * log(m_qc[incoming_edge](iteration,p));
    //         }
    //     }
    // }

    // free_energy(iteration) = - average_energy + entropy;
}


xt::xarray<long double> BpDecoder::get_messages()
{
    xt::xarray<long double> messages = xt::zeros<long double>({n_edges,max_iterations,2,4});
    
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        auto view_cq = xt::view(messages,edge_label[edge],xt::all(),0,xt::all());
        auto view_qc = xt::view(messages,edge_label[edge],xt::all(),1,xt::all());

        view_cq = m_cq[edge];
        view_qc = m_qc[edge];
    }

    return messages;
}


xt::xarray<long double> BpDecoder::get_marginals()
{
    xt::xarray<long double> marginals_to_return = xt::zeros<long double>({n_q,max_iterations,4});
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        auto view_q = xt::view(marginals_to_return,qubit_label[qubit],xt::all(),xt::all());
        view_q = marginals[qubit];
    }
    return marginals_to_return;
}


xt::xarray<int> BpDecoder::get_hard_decisions()
{
    xt::xarray<int> hard_decisions_to_return = xt::zeros<int>({max_iterations,n_q});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_hd = xt::view(hard_decisions_to_return,it,xt::all());
        view_hd = xt::row(hard_decision,it);
    }
    return hard_decisions_to_return;
}

xt::xarray<int> BpDecoder::get_syndromes()
{
    xt::xarray<int> syndromes_to_return = xt::zeros<int>({max_iterations,n_c});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_s = xt::view(syndromes_to_return,it,xt::all());
        view_s = xt::row(syndromes,it);
    }
    return syndromes_to_return;
}

xt::xarray<bool> BpDecoder::get_converged_cq()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_cq[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<bool> BpDecoder::get_converged_qc()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_qc[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<int> BpDecoder::get_check_and_qubit(int edge)
{
    lemon::ListBpGraph::Edge e = g.edgeFromId(edge);
    xt::xarray<int> cq = {check_label[g.redNode(e)],qubit_label[g.blueNode(e)]};
    return cq;
}

