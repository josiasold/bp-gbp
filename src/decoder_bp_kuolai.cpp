#include <bp_gbp/decoder_bp_kuolai.hpp>


BpDecoderKL::BpDecoderKL(xt::xarray<int> H) : H(H), qubit_label(g), check_label(g),edge_label(g), edge_type(g), m_cq(g), m_qc(g),converged_cq(g), converged_qc(g), marginals(g)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    initialize_graph();

    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);
}

void BpDecoderKL::initialize_graph()
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
                
                n_edges++;
            }
        }
    }


    std::cout << "initialized graph with n_c = " << n_c << " n_q = " << n_q << std::endl; 
}

void BpDecoderKL::initialize_bp(xt::xarray<long double> p_init, int max_iter)
{
    p_initial = p_init;
    max_iterations = max_iter;
    hard_decision =  xt::zeros<int>({max_iter+1,n_q});
    syndromes = xt::zeros<int>({max_iter+1,n_c});
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({max_iter});
        m_qc[edge] = xt::zeros<long double>({max_iter});

        m_qc[edge](0) = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;
    }
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({max_iter,4});
    }
    is_initialized = true;
}

xt::xarray<int> BpDecoderKL::decode_bp(xt::xarray<int> s_0, long double w, long double alpha)
{
    if (is_initialized == false)
    {
        std::cerr << "BP not initialized, please run initialize_bp(xt::xarray<long double> p_init)" << std::endl;
    }

    initialize_bp();

    int took_iter = max_iterations;
    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        check_to_bit(&s_0, iteration, w, alpha);

        bit_to_check(iteration, w, alpha);

        marginals_and_hard_decision(iteration);

        xt::xarray<int> s = xt::ones_like(s_0);
        xt::xarray<int> hd =  xt::row(hard_decision,iteration);

        gf4_syndrome(&s, &hd, &H);

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

void BpDecoderKL::initialize_bp()
{
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_qc[edge](0) = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;
    }
}

void BpDecoderKL::check_to_bit(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        long double message = 1;
        long double sign = std::pow(-1,s_0->at(check_label[check]));

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            message = 1;
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    message *= m_qc[other_edge](iteration-1);
                }
            }
            message *= sign;
            if (iteration == 1)
            {
                m_cq[message_edge](iteration-1) = message;
            }
            else
            {
                m_cq[message_edge](iteration-1) = (1-w) * m_cq[message_edge](iteration-2) + w * message;
            }
            
            // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
            long double diff = abs(m_cq[message_edge](iteration-1) - m_cq[message_edge](iteration));
            if (diff < 1E-20)
            {
                converged_cq[message_edge] = true;
            }
        }
    }
    return;
}

void BpDecoderKL::bit_to_check(int iteration,long double w, long double alpha)
{
    int comm_table[16] = {0, 0, 0, 0,
                          0, 0, 1, 1,
                          0, 1, 0, 1,
                          0, 1, 1, 0};

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
                    xt::xarray<long double> r = xt::ones<long double>({2});
                    r(0) = (1 + m_cq[other_edge](iteration-1))/2;
                    r(1) = (1 - m_cq[other_edge](iteration-1))/2;

                    q(0) *=  r(0);
                    
                    for (size_t p = 1; p < 4; p++)
                    {
                        int sp = comm_table[p * 4 + edge_type[other_edge]];
                        q(p) *= r(sp);
                    }
                }
            }

            long double q_0 = q(0) + q(edge_type[message_edge]);
            q_0 = pow(q_0,alpha);
            long double q_1 = 0;
            for (size_t p = 1; p < 4; p++)
            {
                if (p != edge_type[message_edge])
                {
                     q_1 += q(p);
                }
            }
            q_1 = pow(q_1,alpha);
            
            long double sum = q_0 + q_1;

            q_0 /= sum; q_1 /= sum;
            
            long double message = q_0 - q_1;

            m_qc[message_edge](iteration) = (1-w) * m_qc[message_edge](iteration-1) + w * message;
            // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
            if (abs(m_qc[message_edge](iteration-1) -  m_qc[message_edge](iteration))<1E-20)
            {
                converged_qc[message_edge] = true;
            }
        }
    }
    return;
}

void BpDecoderKL::marginals_and_hard_decision(int iteration)
{
    int comm_table[16] = {0, 0, 0, 0,
                          0, 0, 1, 1,
                          0, 1, 0, 1,
                          0, 1, 1, 0};

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = p_initial;
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            xt::xarray<long double> r = xt::ones<long double>({2});
            r(0) = (1 + m_cq[incoming_edge](iteration-1))/2;
            r(1) = (1 - m_cq[incoming_edge](iteration-1))/2;
            
            q(0) *= r(0);
            for (size_t p = 1; p < 4; p++)
            {
                int sp = comm_table[p * 4 + edge_type[incoming_edge]];
                q(p) *= r(sp);
            }
        }
        q /= xt::sum(q);
        xt::row(marginals[qubit],iteration) = q;
        // m = q;
        int hd =  xt::argmax(q,0)();
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}


xt::xarray<long double> BpDecoderKL::get_messages()
{
    xt::xarray<long double> messages = xt::zeros<long double>({n_edges,max_iterations,2});
    
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        auto view_cq = xt::view(messages,edge_label[edge],xt::all(),0);
        auto view_qc = xt::view(messages,edge_label[edge],xt::all(),1);

        view_cq = m_cq[edge];
        view_qc = m_qc[edge];
    }

    return messages;
}


xt::xarray<long double> BpDecoderKL::get_marginals()
{
    xt::xarray<long double> marginals_to_return = xt::zeros<long double>({n_q,max_iterations,4});
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        auto view_q = xt::view(marginals_to_return,qubit_label[qubit],xt::all(),xt::all());
        view_q = marginals[qubit];
    }
    return marginals_to_return;
}


xt::xarray<int> BpDecoderKL::get_hard_decisions()
{
    xt::xarray<int> hard_decisions_to_return = xt::zeros<int>({max_iterations,n_q});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_hd = xt::view(hard_decisions_to_return,it,xt::all());
        view_hd = xt::row(hard_decision,it);
    }
    return hard_decisions_to_return;
}

xt::xarray<int> BpDecoderKL::get_syndromes()
{
    xt::xarray<int> syndromes_to_return = xt::zeros<int>({max_iterations,n_c});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_s = xt::view(syndromes_to_return,it,xt::all());
        view_s = xt::row(syndromes,it);
    }
    return syndromes_to_return;
}

xt::xarray<bool> BpDecoderKL::get_converged_cq()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_cq[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<bool> BpDecoderKL::get_converged_qc()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_qc[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<int> BpDecoderKL::get_check_and_qubit(int edge)
{
    lemon::ListBpGraph::Edge e = g.edgeFromId(edge);
    xt::xarray<int> cq = {check_label[g.redNode(e)],qubit_label[g.blueNode(e)]};
    return cq;
}
