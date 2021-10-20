#include <bp_gbp/decoder_bp_klm.hpp>


BpDecoderKLM::BpDecoderKLM(xt::xarray<int> H) : H(H), qubit_label(g), check_label(g),edge_label(g), edge_type(g), m_cq(g), m_qc(g),converged_cq(g), converged_qc(g), marginals(g)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    initialize_graph();

    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);

}

void BpDecoderKLM::initialize_graph()
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
                // if ((q < 16) && (c <= 203) && (c >= 192))
                // {
                //     std::cout << "(q,c) = (" << q << "," << c << "), g.id(new_edge) = " << g.id(new_edge) << "\n";
                // }
                // if ((q < 16) && (c <= 203) && (c >= 192))
                // {
                //     std::cout << g.id(new_edge) << ",";
                // }
                
                n_edges++;
            }
        }
    }


    std::cout << "initialized graph with n_c = " << n_c << " n_q = " << n_q << std::endl; 
}

void BpDecoderKLM::initialize_bp(xt::xarray<long double> p_init, int max_iter)
{
    p_initial = xt::zeros<long double>({3});
    for (int i = 1; i < 4; i++)
    {
        p_initial(i-1) = log(p_init(0)/p_init(i));
    }
    std::cout << "p_initial = " << p_initial << "\n";
    
    max_iterations = max_iter;
    hard_decision =  xt::zeros<int>({max_iter+1,n_q});
    syndromes = xt::zeros<int>({max_iter+1,n_c});
    free_energy = xt::zeros<long double>({max_iter+1});
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({max_iter});
        m_qc[edge] = xt::zeros<long double>({max_iter,3});

        xt::row(m_qc[edge],0) = p_initial;
    }
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({max_iter,4});
        xt::row(marginals[qubit],0) = p_init;
    }
    is_initialized = true;
}

void BpDecoderKLM::initialize_bp()
{
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        xt::row(m_qc[edge],0) = p_initial;
    }
}

xt::xarray<int> BpDecoderKLM::decode_bp(xt::xarray<int> s_0, long double w, long double alpha, int type_bp, bool return_if_success)
{
    if (is_initialized == false)
    {
        std::cerr << "BP not initialized, please run initialize_bp(xt::xarray<long double> p_init)" << std::endl;
    }

    initialize_bp();

    int took_iter = max_iterations;
    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        
        if (type_bp == 0)
        {
            check_to_bit(&s_0, iteration, w, alpha);
            bit_to_check_and_hd(&s_0, iteration, w, alpha);
        }

        if (type_bp == 10)
        {
            bit_serial(&s_0, iteration, w, alpha);
        }

        // marginals_and_hard_decision(iteration,alpha);

        calculate_free_energy(iteration);

        xt::xarray<int> hd =  xt::row(hard_decision,iteration);


        xt::xarray<int> s =  gf4_syndrome(&hd, &H);

        xt::row(syndromes,iteration) = s;
        
        if (s == s_0)
        {
            took_iter = iteration;
            if (return_if_success)
            {
                return hd;
            }
            
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


void BpDecoderKLM::check_to_bit(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        long double message  = 1;

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            message = 1;
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    long double sum_acomm = 0;
                    for (int i = 0; i<3;i++)
                    {
                        if (i != edge_type[other_edge])
                            sum_acomm += exp(-m_qc[other_edge](iteration-1,i));
                    }
                    // if (iteration < 3) std::cout << "i" << iteration<< " sum0 = " << sum << "\n";
                    // if (iteration < 3) std::cout << "i" << iteration<< " sum1 = " << sum << "\n";
                    long double sum_comm =  1 + exp(-m_qc[other_edge](iteration-1,edge_type[other_edge]));
                    long double lambda = log(sum_comm / sum_acomm);
                    // if (iteration < 3) std::cout << "i" << iteration<< " lambda = " << lambda << "\n";
                    message *= tanh(lambda/2);
                }
            }

            message = 2*atanh(message);
            message *= std::pow(-1,s_0->at(check_label[check]));
            // if (iteration < 20) std::cout << "i" << iteration<< " message = " << message << "\n";
            if (iteration == 1)
            {
                m_cq[message_edge](iteration) = message;
            }
            else
            {
                m_cq[message_edge](iteration) = (1-w) * m_cq[message_edge](iteration-1) + w * message;
            }
            
            // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
            long double diff = abs(m_cq[message_edge](iteration-1) - m_cq[message_edge](iteration));
            if ((diff < 1E-100) && (iteration > 2))
            {
                converged_cq[message_edge] = true;
            }
        }
    }
    return;
}



void BpDecoderKLM::bit_to_check_and_hd(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    xt::xarray<int>comm_table = {{0, 0, 0, 0},
                                 {0, 0, 1, 1},
                                 {0, 1, 0, 1},
                                 {0, 1, 1, 0}};

    xt::xarray<int> hd = xt::zeros<int>({n_q});

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> llr = p_initial;

        for (lemon::ListBpGraph::IncEdgeIt edge_from_check(g,qubit); edge_from_check != lemon::INVALID; ++edge_from_check)
        {
            for (int i = 1; i<4; i++)
            {
                int commutation = comm_table(i,edge_type[edge_from_check]);
                if (commutation == 1)
                {
                    llr(i-1) += m_cq[edge_from_check](iteration)/alpha;
                }
            }
        }
        if (xt::all(llr>0) == 1)
        {
            hd(qubit_label[qubit]) = 0;
        }
        else
        {
            hd(qubit_label[qubit]) = xt::argmin(llr,0)()+1;
        }



        for (lemon::ListBpGraph::IncEdgeIt edge_to_check(g,qubit); edge_to_check != lemon::INVALID; ++edge_to_check)
        {
            xt::xarray<long double> message = llr;
            for (int i = 1; i<4; i++)
            {
                message(i-1) -= comm_table(i,edge_type[edge_to_check])*m_cq[edge_to_check](iteration);
            }
            xt::row(m_qc[edge_to_check],iteration) = (1-w) * xt::row(m_qc[edge_to_check],iteration-1) + w * message;
        }

    }
    xt::row(hard_decision,iteration) = hd;


           
            // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
            // long double diff = abs(m_qc[message_edge](iteration-1) -  m_qc[message_edge](iteration));
            // // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
            // if ((diff < 1E-100) && (iteration > 2))
            // {
            //     converged_qc[message_edge] = true;
            // }
    //     }
    // }
    return;
}


void BpDecoderKLM::bit_serial(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{

    xt::xarray<int>comm_table = {{0, 0, 0, 0},
                                 {0, 0, 1, 1},
                                 {0, 1, 0, 1},
                                 {0, 1, 1, 0}};

    xt::xarray<int> hd = xt::zeros<int>({n_q});

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    { // for each qubit do

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,qubit); message_edge != lemon::INVALID; ++message_edge)
        {
            // check = g.redNode(message_edge)
            // for each check compute c -> q messages
            long double message  = 1;

            for (lemon::ListBpGraph::IncEdgeIt edge_into_check(g,g.redNode(message_edge)); edge_into_check != lemon::INVALID; ++edge_into_check)
            {

                if (edge_into_check != message_edge)
                {
                    
                    long double sum_acomm = 0;
                    long double sum_comm =  1;
                    for (int i = 1; i<4;i++)
                    {
                        if (i != edge_type[edge_into_check])
                            sum_acomm += exp(-m_qc[edge_into_check](0,i-1));
                        else
                            sum_comm += exp(-m_qc[edge_into_check](0,i-1));
                    }
                    
                    long double lambda = log(sum_comm / sum_acomm);
                    
                    message *= tanh(lambda/2);
                }
                
            }
            message = 2*atanh(message);
            message *= std::pow(-1,s_0->at(check_label[g.redNode(message_edge)]));
           
            m_cq[message_edge](0) = message;
                
                // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
                // long double diff = abs(m_cq[edge_to_check](iteration-1) - m_cq[edge_to_check](iteration));
                // if ((diff < 1E-100) && (iteration > 2))
                // {
                //     converged_cq[edge_to_check] = true;
                // }
            // }
        }

        // for each check compute q -> c messages

        xt::xarray<long double> llr = xt::zeros<long double>({3});

        for (lemon::ListBpGraph::IncEdgeIt edge_from_check(g,qubit); edge_from_check != lemon::INVALID; ++edge_from_check)
        {
            for (int i = 1; i<4; i++)
            {
                int commutation = comm_table(i,edge_type[edge_from_check]);
                if (commutation == 1)
                {
                    llr(i-1) += m_cq[edge_from_check](0);
                }
            }
        }
        
        llr /= alpha;
        llr += p_initial;
        
        if (xt::all(llr>0) == 1)
        {
            hd(qubit_label[qubit]) = 0;
        }
        else
        {
            hd(qubit_label[qubit]) = xt::argmin(llr,0)()+1;
        }


        for (lemon::ListBpGraph::IncEdgeIt edge_to_check(g,qubit); edge_to_check != lemon::INVALID; ++edge_to_check)
        {
            xt::xarray<long double> message = llr;
            for (int i = 1; i<4; i++)
            {
                message(i-1) -= comm_table(i,edge_type[edge_to_check])*m_cq[edge_to_check](0);
            }
            xt::row(m_qc[edge_to_check],0) =  message;
        }

    }

    xt::row(hard_decision,iteration) = hd;


}


void BpDecoderKLM::marginals_and_hard_decision(int iteration, long double alpha)
{
    int comm_table[16] = {0, 0, 0, 0,
                          0, 0, 1, 1,
                          0, 1, 0, 1,
                          0, 1, 1, 0};

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = p_initial;
        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,qubit); message_edge != lemon::INVALID; ++message_edge)
        {
            for (int i = 0; i<3; i++)
            {
                int sp = comm_table[i+1 * 4 + edge_type[message_edge]];
                if (sp == 1)
                {
                    q(i) += m_cq[message_edge](iteration)/alpha;
                }
            }
        }
        int t = 0;
        for (int i = 0; i<3; i++)
        {
            marginals[qubit](iteration,i+1) = q(i);
            if (q(i) > 0) t++;
        }
        // std::cout << "q = " << q << ", ";
        // m = q;
        int hd;
        if (t == 3) hd = 0;
        else  hd =  xt::argmin(q,0)()+1;
        // std::cout << "hd = " << hd << "\n";
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}

void BpDecoderKLM::calculate_free_energy(int iteration)
{
    lemon::OutDegMap<lemon::ListBpGraph> outDeg(g);
    long double f = 0;
    // std::cout << "*** F: iteration " << iteration << " f (init) = " << f << "\n";
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q_belief = xt::row(marginals[qubit],iteration);
        // if (g.id(qubit) == 0) std::cout << "| q_belief b/f = " << q_belief << "\n";

        auto q_b_0 = xt::filter(q_belief, xt::equal(q_belief,0));
        auto q_b_n0 = xt::filter(q_belief, xt::not_equal(q_belief,0));
        q_b_0 = 0;
        q_b_n0 *= log(q_b_n0);


        q_belief *= outDeg[qubit] - 1;
        // if (g.id(qubit) == 0) std::cout << "| q_belief a/f = " << q_belief << "\n";
        // if (g.id(qubit) == 0) std::cout << "| f b/f = " << f << "\n";
        f += xt::sum(q_belief)();
        // if (g.id(qubit) == 0) std::cout << "| f a/f = " << f << "\n";
    }
    // std::cout << "*** f =  " << f << "\n";
    free_energy(iteration) = f;
}


xt::xarray<long double> BpDecoderKLM::get_messages()
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


xt::xarray<long double> BpDecoderKLM::get_marginals()
{
    xt::xarray<long double> marginals_to_return = xt::zeros<long double>({n_q,max_iterations,4});
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        auto view_q = xt::view(marginals_to_return,qubit_label[qubit],xt::all(),xt::all());
        view_q = marginals[qubit];
    }
    return marginals_to_return;
}


xt::xarray<int> BpDecoderKLM::get_hard_decisions()
{
    xt::xarray<int> hard_decisions_to_return = xt::zeros<int>({max_iterations,n_q});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_hd = xt::view(hard_decisions_to_return,it,xt::all());
        view_hd = xt::row(hard_decision,it);
    }
    return hard_decisions_to_return;
}

xt::xarray<int> BpDecoderKLM::get_syndromes()
{
    xt::xarray<int> syndromes_to_return = xt::zeros<int>({max_iterations,n_c});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_s = xt::view(syndromes_to_return,it,xt::all());
        view_s = xt::row(syndromes,it);
    }
    return syndromes_to_return;
}

xt::xarray<bool> BpDecoderKLM::get_converged_cq()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_cq[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<bool> BpDecoderKLM::get_converged_qc()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_qc[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<int> BpDecoderKLM::get_check_and_qubit(int edge)
{
    lemon::ListBpGraph::Edge e = g.edgeFromId(edge);
    xt::xarray<int> cq = {check_label[g.redNode(e)],qubit_label[g.blueNode(e)]};
    return cq;
}

xt::xarray<long double> BpDecoderKLM::get_free_energy()
{
    return free_energy;
}