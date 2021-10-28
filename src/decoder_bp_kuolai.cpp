#include <bp_gbp/decoder_bp_kuolai.hpp>

namespace bp{

BpDecoderKL::BpDecoderKL(xt::xarray<int> H) : H(H), qubit_label(g), check_label(g),edge_label(g), edge_type(g), m_cq(g), m_qc(g), m_cq_current(g), m_qc_current(g), r(g), erased(g), converged_cq(g), converged_qc(g), marginals(g)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    m_properties.m_only_nonconverged_edges = false;
    erasure_channel = false;
    initialize_graph();

    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);
    lemon::mapFill(g,erased,false);
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
    // std::cout << "initialized graph with n_c = " << n_c << " n_q = " << n_q << std::endl; 
}

void BpDecoderKL::initialize_bp(xt::xarray<long double> p_init, int t_max_iter, long double t_w, long double t_alpha, int t_type, bool t_return_if_success, bool t_only_nonconverged_edges)
{
    p_initial = p_init;
    
    m_properties.m_max_iter = t_max_iter;
    m_properties.m_w = t_w;
    m_properties.m_alpha = t_alpha;
    m_properties.m_type = t_type;
    m_properties.m_return_if_success = t_return_if_success;
    m_properties.m_only_nonconverged_edges = t_only_nonconverged_edges;


    hard_decision =  xt::zeros<int>({t_max_iter+1,n_q});
    syndromes = xt::zeros<int>({t_max_iter+1,n_c});
    free_energy = xt::zeros<long double>({t_max_iter+1});
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({t_max_iter});
        m_qc[edge] = xt::zeros<long double>({t_max_iter});

        m_qc[edge](0) = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;

        m_cq_current[edge] = 1.0;
        m_qc_current[edge] = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;

        r[edge] = xt::zeros<long double>({2});
    }
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({t_max_iter,4});
        xt::row(marginals[qubit],0) = p_init;
    }
    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);
    is_initialized = true;
    took_iterations = 0;
}

void BpDecoderKL::initialize_bp(xt::xarray<int> * t_s_0)
{
    s_0 = *t_s_0;
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({m_properties.m_max_iter});
        m_qc[edge] = xt::zeros<long double>({m_properties.m_max_iter});

        m_cq_current[edge] = 1.0;
        m_qc_current[edge] = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;

        r[edge] = xt::ones<long double>({2});

        if (erasure_channel == true)
        {
            if (erased[g.blueNode(edge)] == false)
            {
                m_qc[edge](0) = 1.0;
                converged_cq[edge] = true;
                converged_qc[edge] = true;
            }
            else
            {
                m_qc[edge](0) = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;
                converged_cq[edge] = false;
                converged_qc[edge] = false;
            }
        }
        else
        {
            m_qc[edge](0) = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;
            converged_cq[edge] = false;
            converged_qc[edge] = false;
        }

    }
    check_to_bit(0);

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({m_properties.m_max_iter,4});
        if (erasure_channel)
        {
            if (erased[qubit])
            {
                xt::row(marginals[qubit],0) = p_initial;
                marginals[qubit](0,0) = 0.0;
                xt::row(marginals[qubit],0) /= xt::sum(xt::row(marginals[qubit],0));
            }
            else
            {
                marginals[qubit](0,0) = 1.0; marginals[qubit](0,1) = 0.0; marginals[qubit](0,2) = 0.0; marginals[qubit](0,3) = 0.0; 
            }
        }
        else
        {
             xt::row(marginals[qubit],0) = p_initial;
        }
       
    }

    is_initialized = true;
    took_iterations = 0;
}

void BpDecoderKL::initialize_erasures(xt::xarray<int> * erasures)
{
    erasure_channel = true;
    p_initial(0) = 0.0;
    p_initial/= xt::sum(p_initial);
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        int qubit = qubit_label[g.blueNode(edge)];
        if (erasures->at(qubit) == 1)
        {
            erased[g.blueNode(edge)] = true;
            m_qc[edge](0) = 2 * (p_initial(0) + p_initial(edge_type[edge]))- 1;
        }
        else
        {
            erased[g.blueNode(edge)] = false;
            m_qc[edge](0) = 1.0;
        }
    }
}


xt::xarray<int> BpDecoderKL::decode_bp(xt::xarray<int> t_s_0)
{
    if (is_initialized == false)
    {
        std::cerr << "BP not initialized, please run initialize_bp(xt::xarray<long double> p_init)" << std::endl;
    }



    initialize_bp(&t_s_0);

    int took_iter = m_properties.m_max_iter;
    for (int iteration = 1; iteration < m_properties.m_max_iter; iteration++)
    {
        
        if (m_properties.m_type == 0) // parallel
        {
            check_to_bit(iteration);
            bit_to_check(iteration);
        }
        else if (m_properties.m_type == 1) // bit serial
        {
            bit_serial_update(iteration);
        }
        else if (m_properties.m_type == 2) // check serial
        {
            check_serial_update(iteration);
        }
        else if (m_properties.m_type == 3) // bit sequential
        {
            bit_sequential_update(iteration);
        }
        else if (m_properties.m_type == 4) // check sequential
        {
            check_sequential_update(iteration);
        }
        marginals_and_hard_decision(iteration);

        calculate_free_energy(iteration);

        xt::xarray<int> s = xt::ones_like(s_0);
        xt::xarray<int> hd =  xt::row(hard_decision,iteration);

        gf4_syndrome(&s, &hd, &H);

        xt::row(syndromes,iteration) = s;
        
        if (s == s_0)
        {
            took_iter = iteration;
            took_iterations = took_iter;
            if (m_properties.m_return_if_success)
            {
                return hd;
            }
            
        }
    }

    xt::xarray<int> hd =  xt::zeros<int>({n_q});
    took_iter = m_properties.m_max_iter;
    for (int i = m_properties.m_max_iter; i>0; i--)
    {
        xt::xarray<int> hd =  xt::row(hard_decision,i);
        if (xt::sum(hd)() != 0)
        {
            return hd;
        }
    }
    return hd;
}

void BpDecoderKL::check_to_bit_single_edge(const lemon::ListBpGraph::Edge& message_edge, int iteration)
{
    if (!((converged_cq[message_edge]) && (m_properties.m_only_nonconverged_edges)))
    {
        long double w = m_properties.m_w;
        long double alpha = m_properties.m_alpha;

        long double message_update = std::pow(-1.0,s_0.at(check_label[g.redNode(message_edge)]));

        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g, g.redNode(message_edge)); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            if (message_edge != incoming_edge)
            {
                message_update *= m_qc_current[incoming_edge];
            }
        } 

        if (iteration <= 1)
        {
            m_cq[message_edge](iteration) = message_update;
            m_cq_current[message_edge] = message_update;
        }
        else
        {
            m_cq[message_edge](iteration) = (1-w) * m_cq_current[message_edge] + w * message_update;
            m_cq_current[message_edge] = (1-w) * m_cq_current[message_edge] + w * message_update;
        }
        long double diff = fabs(m_cq[message_edge](iteration-1) - m_cq[message_edge](iteration));
        if ((diff < 1E-100) && (iteration > 5))
        {
            converged_cq[message_edge] = true;
        }
    }
    else
    {
        m_cq[message_edge](iteration) = m_cq[message_edge](iteration-1);
    }
    return;
}

void BpDecoderKL::calculate_r_single_edge(const lemon::ListBpGraph::Edge& edge)
{
    r[edge](0) = pow((1 + m_cq_current[edge])/2,1.0/m_properties.m_alpha);
    r[edge](1) = pow((1 - m_cq_current[edge])/2,1.0/m_properties.m_alpha);
}

void BpDecoderKL::bit_to_check_single_edge(const lemon::ListBpGraph::Edge& message_edge, int iteration)
{
    if (!((converged_qc[message_edge]) && (m_properties.m_only_nonconverged_edges)))
    {
        long double w = m_properties.m_w;
        long double alpha = m_properties.m_alpha;
        xt::xarray<int> comm_table = {{0, 0, 0, 0},
                                    {0, 0, 1, 1},
                                    {0, 1, 0, 1},
                                    {0, 1, 1, 0}};

        xt::xarray<long double> q = p_initial;

        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g, g.blueNode(message_edge)); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            if (message_edge != incoming_edge)
            {
                q(0) *= r[incoming_edge](0); // I
                q(1) *= r[incoming_edge](comm_table(1,edge_type[incoming_edge])); // X
                q(2) *= r[incoming_edge](comm_table(2,edge_type[incoming_edge])); // Y
                q(3) *= r[incoming_edge](comm_table(3,edge_type[incoming_edge])); // Z
            }
        }

        long double q_0 = q(0) + q(edge_type[message_edge]);
        long double q_1 = 0;
        for (size_t p = 1; p < 4; p++)
        {
            if (p != edge_type[message_edge])
            {
                q_1 += q(p);
            }
        }

        if (alpha != 1)
        {
            long double r_0 = pow(r[message_edge](0),1.0/alpha-1.0);
            long double r_1 = pow(r[message_edge](1),1.0/alpha-1.0);
            q_0 *= r_0;
            q_1 *= r_1;
        }
        
        
        long double sum = q_0 + q_1;
        q_0 /= sum; q_1 /= sum;

            
        
        long double message = q_0 - q_1;

        m_qc[message_edge](iteration) = (1-w) * m_qc_current[message_edge]  + w * message;
        m_qc_current[message_edge] = (1-w) *  m_qc_current[message_edge] + w * message;
        long double diff = fabs(m_qc[message_edge](iteration-1) -  m_qc[message_edge](iteration));

        if ((diff < 1E-100) && (iteration > 5))
        {
            converged_qc[message_edge] = true;
        }
    }
    else
    {
        m_qc[message_edge](iteration) = m_qc[message_edge](iteration-1);
    }
    return;
}

// full update of all outgoing messages of given check
void BpDecoderKL::check_update(const lemon::ListBpGraph::RedNode& check, int iteration)
{
    for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,check);outgoing_edge != lemon::INVALID; ++outgoing_edge)
    {
        check_to_bit_single_edge(outgoing_edge, iteration);
    }
    return;
}

// full update of all outgoing messages of given qubit
void BpDecoderKL::qubit_update(const lemon::ListBpGraph::BlueNode& qubit, int iteration)
{
    for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,qubit); outgoing_edge != lemon::INVALID; ++outgoing_edge)
    {
        calculate_r_single_edge(outgoing_edge);
    }
    for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,qubit);outgoing_edge != lemon::INVALID; ++outgoing_edge)
    {
        bit_to_check_single_edge(outgoing_edge, iteration);
    }
    return;
}

// (parallel) update of all check-to-bit messages
void BpDecoderKL::check_to_bit(int iteration)
{
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        check_to_bit_single_edge(edge, iteration);
    }
    return;
}

// (parallel) update of all bit-to-check messages
void BpDecoderKL::bit_to_check(int iteration)
{
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        calculate_r_single_edge(edge);
    }
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        bit_to_check_single_edge(edge, iteration);
    }
    return;
}

// serial update of, qubit wise
void BpDecoderKL::bit_serial_update(int iteration)
{
    // for every qubit
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        // update incoming messages
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            check_to_bit_single_edge(incoming_edge, iteration);
        }
        //update outgoing messages
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,qubit); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            calculate_r_single_edge(outgoing_edge);
        }
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,qubit); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            bit_to_check_single_edge(outgoing_edge, iteration);
        }
    }
    return;
}

// serial update of, check wise
void BpDecoderKL::check_serial_update(int iteration)
{
    // for every check
    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        // update incoming messages
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,check); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            calculate_r_single_edge(outgoing_edge);
        }
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,check); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            bit_to_check_single_edge(incoming_edge, iteration);
        }
        //update outgoing messages
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,check); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            check_to_bit_single_edge(outgoing_edge, iteration);
        }
    }
    return;
}

// Sequential Update Routines

void BpDecoderKL::bit_sequential_update(int iteration)
{
    // for every qubit
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        // fully update qubit
        qubit_update(qubit,iteration);

        // fully update adjacent checks
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,qubit); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            check_update(g.redNode(outgoing_edge),iteration);
        }
    }
    
    return;
}

void BpDecoderKL::check_sequential_update(int iteration)
{
    // for every check
    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        // fully update check
        check_update(check,iteration);

        // fully update adjacent qubits
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,check); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            qubit_update(g.blueNode(outgoing_edge),iteration);
        }
    }
    
    return;
}

void BpDecoderKL::marginals_and_hard_decision(int iteration)
{
    long double alpha = m_properties.m_alpha;
    xt::xarray<int> comm_table = {{0, 0, 0, 0},
                                  {0, 0, 1, 1},
                                  {0, 1, 0, 1},
                                  {0, 1, 1, 0}};

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = p_initial;
        if (erasure_channel)
        {
            if (erased[qubit])
            {
                q(0) = 0;
                q /= xt::sum(q);
            }
            else
            {
                q(0) = 1.0; q(1) = 0.0; q(2) = 0.0; q(3) = 0.0;
            }
        }

        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            q(0) *= r[incoming_edge](0);
            for (size_t p = 1; p < 4; p++)
            {
                int sp = comm_table(p,edge_type[incoming_edge]);
                q(p) *= r[incoming_edge](sp);
            }
        }
        q /= xt::sum(q);
        xt::row(marginals[qubit],iteration) = q;

        int hd = 0;
        int hd2 = 0;
        double q_max = q(0);
        for (int i = 1; i < q.size(); i++)
        {
            if (q(i) > q(hd))
            {
                hd = i;
            }
            else if (q(i) == q(hd))
            {
                hd2 == i;
            }
        }
        if (hd2 == hd)
        {
            xt::xarray<int> hds = {hd,hd2};
            hd = xt::random::choice(hds,1)(0);
        }
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}


void BpDecoderKL::calculate_free_energy(int iteration)
{
    lemon::OutDegMap<lemon::ListBpGraph> outDeg(g);
    long double f = 0;

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q_belief = xt::row(marginals[qubit],iteration);

        auto q_b_0 = xt::filter(q_belief, xt::equal(q_belief,0));
        auto q_b_n0 = xt::filter(q_belief, xt::not_equal(q_belief,0));
        q_b_0 = 0;
        q_b_n0 *= log(q_b_n0);
        q_belief *= outDeg[qubit] - 1;

        f += xt::sum(q_belief)();
    }
    free_energy(iteration) = -f;
}


xt::xarray<long double> BpDecoderKL::get_messages()
{
    xt::xarray<long double> messages = xt::zeros<long double>({n_edges,m_properties.m_max_iter,2});
    
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
    xt::xarray<long double> marginals_to_return = xt::zeros<long double>({n_q,m_properties.m_max_iter,4});
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        auto view_q = xt::view(marginals_to_return,qubit_label[qubit],xt::all(),xt::all());
        view_q = marginals[qubit];
    }
    return marginals_to_return;
}


xt::xarray<int> BpDecoderKL::get_hard_decisions()
{
    xt::xarray<int> hard_decisions_to_return = xt::zeros<int>({m_properties.m_max_iter,n_q});
    for (int it = 0; it < m_properties.m_max_iter; it++)
    {
        auto view_hd = xt::view(hard_decisions_to_return,it,xt::all());
        view_hd = xt::row(hard_decision,it);
    }
    return hard_decisions_to_return;
}

xt::xarray<int> BpDecoderKL::get_syndromes()
{
    xt::xarray<int> syndromes_to_return = xt::zeros<int>({m_properties.m_max_iter,n_c});
    for (int it = 0; it < m_properties.m_max_iter; it++)
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

xt::xarray<long double> BpDecoderKL::get_free_energy()
{
    return free_energy;
}

} // end of namespace bp