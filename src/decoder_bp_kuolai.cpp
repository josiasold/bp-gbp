#include <bp_gbp/decoder_bp_kuolai.hpp>

namespace bp{

BpDecoderKL::BpDecoderKL(xt::xarray<int> H) : H(H), qubit_label(g), check_label(g),edge_label(g), edge_type(g), m_cq(g), m_qc(g), erased(g), converged_cq(g), converged_qc(g), marginals(g)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    only_non_converged = false;
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
                // if ((check_label[check] == 25) || (check_label[check] == 26) || (check_label[check] == 29) || (check_label[check] == 30))
                //     std::cout << "(q->c) =  (" << q << "," << c << ") edge = " << edge_label[new_edge] << "\n";
            }
        }
    }


    // std::cout << "initialized graph with n_c = " << n_c << " n_q = " << n_q << std::endl; 
}

void BpDecoderKL::initialize_bp(xt::xarray<long double> p_init, int t_max_iter, long double t_w, long double t_alpha, int t_type, bool t_return_if_success, bool t_only_nonconverged_edges)
{
    p_initial = p_init;
    max_iterations = t_max_iter;
    
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

void BpDecoderKL::initialize_bp(xt::xarray<int> * s_0)
{
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({max_iterations});
        m_qc[edge] = xt::zeros<long double>({max_iterations});

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
    check_to_bit(s_0, 0);
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({max_iterations,4});
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


xt::xarray<int> BpDecoderKL::decode_bp(xt::xarray<int> s_0)
{
    if (is_initialized == false)
    {
        std::cerr << "BP not initialized, please run initialize_bp(xt::xarray<long double> p_init)" << std::endl;
    }

    only_non_converged = m_properties.m_only_nonconverged_edges;

    initialize_bp(&s_0);

    int took_iter = max_iterations;
    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        
        if (m_properties.m_type == 0) // parallel
        {
            check_to_bit(&s_0, iteration);
            bit_to_check(iteration);
            marginals_and_hard_decision(iteration);
        }
        else if (m_properties.m_type == 1) // bit serial
        {
            bit_serial_update(&s_0, iteration);
            marginals_and_hard_decision_serial(iteration);
        }
        else if (m_properties.m_type == 2) // check serial
        {
            check_serial_update(&s_0, iteration);
            marginals_and_hard_decision_serial(iteration);
        }
        else if (m_properties.m_type == 3) // bit sequential
        {
            bit_sequential_update(&s_0, iteration);
            marginals_and_hard_decision_serial(iteration);
        }
        else if (m_properties.m_type == 4) // check sequential
        {
            check_sequential_update(&s_0, iteration);
            marginals_and_hard_decision_serial(iteration);
        }
        

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
    took_iter = max_iterations;
    for (int i = max_iterations; i>0; i--)
    {
        xt::xarray<int> hd =  xt::row(hard_decision,i);
        if (xt::sum(hd)() != 0)
        {
            return hd;
        }
    }
    return hd;
}


void BpDecoderKL::check_to_bit(xt::xarray<int> * s_0, int iteration)
{
    long double w = m_properties.m_w;
    long double alpha = m_properties.m_alpha;
    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        long double message = 1;
        long double sign = std::pow(-1,s_0->at(check_label[check]));

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            if (!((converged_cq[message_edge]) && (only_non_converged)))
            {
                message = 1;
                for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
                {
                    if (other_edge != message_edge)
                    {
                        if (iteration > 0)
                            message *= m_qc[other_edge](iteration-1);
                        else
                            message *= m_qc[other_edge](iteration);
                    }
                }
                message *= sign;
                if (iteration <= 1)
                {
                    m_cq[message_edge](iteration) = message;
                }
                else
                {
                    m_cq[message_edge](iteration) = (1-w) * m_cq[message_edge](iteration-1) + w * message;
                }
                
                // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
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
        }
    }
    return;
}

void BpDecoderKL::bit_to_check(int iteration)
{
    long double w = m_properties.m_w;
    long double alpha = m_properties.m_alpha;
    xt::xarray<int> comm_table = {{0, 0, 0, 0},
                                  {0, 0, 1, 1},
                                  {0, 1, 0, 1},
                                  {0, 1, 1, 0}};

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = xt::ones<long double>({4});

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,qubit); message_edge != lemon::INVALID; ++message_edge)
        {
            if (!((converged_qc[message_edge]) && (only_non_converged)))
            {
                q = p_initial;
                for (lemon::ListBpGraph::IncEdgeIt other_edge(g,qubit); other_edge != lemon::INVALID; ++other_edge)
                {
                    if (other_edge != message_edge)
                    {
                        xt::xarray<long double> r = xt::ones<long double>({2});
                        if (alpha == 1)
                        {
                            r(0) = (1 + m_cq[other_edge](iteration))/2;
                            r(1) = (1 - m_cq[other_edge](iteration))/2;
                        }
                        else
                        {
                            r(0) = pow((1 + m_cq[other_edge](iteration))/2,1.0/alpha);
                            r(1) = pow((1 - m_cq[other_edge](iteration))/2,1.0/alpha);
                        }
                        

                        q(0) *=  r(0); // I
                        
                        q(1) *= r(comm_table(1,edge_type[other_edge])); // X
                        q(2) *= r(comm_table(2,edge_type[other_edge])); // Y
                        q(3) *= r(comm_table(3,edge_type[other_edge])); // Z

                        // for (size_t p = 1; p < 4; p++)
                        // {
                        //     int sp = comm_table(p,edge_type[other_edge]);
                        //     q(p) *= r(sp);
                        // }
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
                    long double r_0 = pow((1 + m_cq[message_edge](iteration))/2.0,1.0/alpha-1.0);
                    long double r_1 = pow((1 - m_cq[message_edge](iteration))/2.0,1.0/alpha-1.0);
                    q_0 *= r_0;
                    q_1 *= r_1;
                }
                
                
                long double sum = q_0 + q_1;
                q_0 /= sum; q_1 /= sum;

                    
                
                long double message = q_0 - q_1;

                m_qc[message_edge](iteration) = (1-w) * m_qc[message_edge](iteration-1) + w * message;
                // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
                long double diff = fabs(m_qc[message_edge](iteration-1) -  m_qc[message_edge](iteration));
                // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
                if ((diff < 1E-100) && (iteration > 5))
                {
                    converged_qc[message_edge] = true;
                }
            }
            else
            {
                m_qc[message_edge](iteration) = m_qc[message_edge](iteration-1);
            }
        }
    }
    return;
}

void BpDecoderKL::bit_serial_update(xt::xarray<int> * s_0, int iteration)
{
    long double w = m_properties.m_w;
    long double alpha = m_properties.m_alpha;
    xt::xarray<int> comm_table= {{0, 0, 0, 0},
                                 {0, 0, 1, 1},
                                 {0, 1, 0, 1},
                                 {0, 1, 1, 0}};


    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    { // for each qubit do

        // calculate all incoming (c -> q) messages
        for (lemon::ListBpGraph::IncEdgeIt incoming_message_edge(g,qubit); incoming_message_edge != lemon::INVALID; ++incoming_message_edge)
        {
            if (!((converged_cq[incoming_message_edge]) && (only_non_converged)))
            {
            //check = g.redNode(message_edge)
                long double message = 1;
                long double sign = std::pow(-1.0,s_0->at(check_label[g.redNode(incoming_message_edge)]));

                for (lemon::ListBpGraph::IncEdgeIt edge_to_check(g,g.redNode(incoming_message_edge)); edge_to_check != lemon::INVALID; ++edge_to_check)
                {
                    if (edge_to_check != incoming_message_edge)
                    {
                        message *= m_qc[edge_to_check](0);
                    }
                }
                message *= sign;
                long double old_message = m_cq[incoming_message_edge](0);
                m_cq[incoming_message_edge](0) = (1-w)*old_message + w*message;
                
                // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
                long double diff = fabs(m_cq[incoming_message_edge](0) - old_message);
                if ((diff < 1E-100) && (iteration > 5))
                {
                    converged_cq[incoming_message_edge] = true;
                }

            }
            
            // if (iteration < 4)
            //     std::cout << "it " << iteration << " q " << qubit_label[qubit] << " <- c " << check_label[g.redNode(message_edge)] << " m_cq = " << m_cq[message_edge](0) << "\n";
        }

        // calculate  all outgoing (q -> c) messages
        for (lemon::ListBpGraph::IncEdgeIt outgoing_message_edge(g,qubit); outgoing_message_edge != lemon::INVALID; ++outgoing_message_edge)
        { 
            if (!((converged_qc[outgoing_message_edge]) && (only_non_converged)))
            {
                xt::xarray<long double> q = p_initial;
                for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
                {
                    if (incoming_edge != outgoing_message_edge)
                    {
                        xt::xarray<long double> r = xt::ones<long double>({2});
                        if (alpha != 1.0)
                        {
                            r(0) = pow((1 + m_cq[incoming_edge](0))/2,1.0/alpha);
                            r(1) = pow((1 - m_cq[incoming_edge](0))/2,1.0/alpha);
                        }
                        else
                        {
                            r(0) = (1 + m_cq[incoming_edge](0))/2;
                            r(1) = (1 - m_cq[incoming_edge](0))/2;
                        }

                        q(0) *=  r(0);
                        
                        for (size_t p = 1; p < 4; p++)
                        {
                            int sp = comm_table(p,edge_type[incoming_edge]);
                            q(p) *= r(sp);
                        }
                    }
                }


                long double q_0 = q(0) + q(edge_type[outgoing_message_edge]);
                long double q_1 = 0;
                for (size_t p = 1; p < 4; p++)
                {
                    if (p != edge_type[outgoing_message_edge])
                    {
                        q_1 += q(p);
                    }
                }

                if (alpha != 1.0)
                {
                    long double r_0 = pow((1 + m_cq[outgoing_message_edge](0))/2.0,1.0/alpha-1.0);
                    long double r_1 = pow((1 - m_cq[outgoing_message_edge](0))/2.0,1.0/alpha-1.0);
                    q_0 *= r_0;
                    q_1 *= r_1;
                }
                
                long double sum = q_0 + q_1;

                q_0 /= sum; q_1 /= sum;
                
                long double message = q_0 - q_1;

                long double old_message = m_qc[outgoing_message_edge](0);

                m_qc[outgoing_message_edge](0) = (1-w)*old_message + w * message;
                
                // if (iteration < 4)
                    // std::cout << "it " << iteration << " q " << qubit_label[qubit] << " -> c " << check_label[g.redNode(message_edge)] << " m_qc = " << m_qc[message_edge](0) << "\n";

                // m_qc[message_edge](iteration) = (1-w) * m_qc[message_edge](iteration-1) + w * message;

                // // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
                long double diff = fabs(m_qc[outgoing_message_edge](0) -  old_message);
                // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
                if ((diff < 1E-100) && (iteration > 5))
                {
                    converged_qc[outgoing_message_edge] = true;
                }
            }
        }
    }
    return;
}

void BpDecoderKL::bit_sequential_update(xt::xarray<int> * s_0, int iteration)
{
    long double w = m_properties.m_w;
    long double alpha = m_properties.m_alpha;
    xt::xarray<int> comm_table= {{0, 0, 0, 0},
                                 {0, 0, 1, 1},
                                 {0, 1, 0, 1},
                                 {0, 1, 1, 0}};


    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    { // for each qubit do

        // calculate  all outgoing (q -> c) messages
        for (lemon::ListBpGraph::IncEdgeIt outgoing_message_edge(g,qubit); outgoing_message_edge != lemon::INVALID; ++outgoing_message_edge)
        { 
            if (!((converged_qc[outgoing_message_edge]) && (only_non_converged)))
            {
                xt::xarray<long double> q = p_initial;
                for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
                {
                    if (incoming_edge != outgoing_message_edge)
                    {
                        xt::xarray<long double> r = xt::ones<long double>({2});
                        if (alpha != 1.0)
                        {
                            r(0) = pow((1 + m_cq[incoming_edge](0))/2,1.0/alpha);
                            r(1) = pow((1 - m_cq[incoming_edge](0))/2,1.0/alpha);
                        }
                        else
                        {
                            r(0) = (1 + m_cq[incoming_edge](0))/2;
                            r(1) = (1 - m_cq[incoming_edge](0))/2;
                        }

                        q(0) *=  r(0);
                        
                        for (size_t p = 1; p < 4; p++)
                        {
                            int sp = comm_table(p,edge_type[incoming_edge]);
                            q(p) *= r(sp);
                        }
                    }
                }


                long double q_0 = q(0) + q(edge_type[outgoing_message_edge]);
                long double q_1 = 0;
                for (size_t p = 1; p < 4; p++)
                {
                    if (p != edge_type[outgoing_message_edge])
                    {
                        q_1 += q(p);
                    }
                }

                if (alpha != 1.0)
                {
                    long double r_0 = pow((1 + m_cq[outgoing_message_edge](0))/2.0,1.0/alpha-1.0);
                    long double r_1 = pow((1 - m_cq[outgoing_message_edge](0))/2.0,1.0/alpha-1.0);
                    q_0 *= r_0;
                    q_1 *= r_1;
                }
                
                long double sum = q_0 + q_1;

                q_0 /= sum; q_1 /= sum;
                
                long double message = q_0 - q_1;

                long double old_message = m_qc[outgoing_message_edge](0);

                m_qc[outgoing_message_edge](0) = (1-w)*old_message + w * message;

                long double diff = fabs(m_qc[outgoing_message_edge](0) -  old_message);

                if ((diff < 1E-100) && (iteration > 5))
                {
                    converged_qc[outgoing_message_edge] = true;
                }
            }
        }

        // update all neighboring checks (c -> q) messages
        for (lemon::ListBpGraph::IncEdgeIt edge_to_check(g,qubit); edge_to_check != lemon::INVALID; ++edge_to_check)
        {
            // update all outgoing messages
            for (lemon::ListBpGraph::IncEdgeIt message_edge(g,g.redNode(edge_to_check)); message_edge != lemon::INVALID; ++message_edge)
            {
                 if (!((converged_cq[message_edge]) && (only_non_converged)))
                {
                //check = g.redNode(message_edge)
                    long double message = 1;
                    long double sign = std::pow(-1.0,s_0->at(check_label[g.redNode(message_edge)]));

                    for (lemon::ListBpGraph::IncEdgeIt other_edge(g,g.redNode(message_edge)); other_edge != lemon::INVALID; ++other_edge)
                    {
                        if (other_edge != message_edge)
                        {
                            message *= m_qc[other_edge](0);
                        }
                    }
                    message *= sign;
                    long double old_message = m_cq[message_edge](0);
                    m_cq[message_edge](0) = (1-w)*old_message + w*message;
                    
                    long double diff = fabs(m_cq[message_edge](0) - old_message);
                    if ((diff < 1E-100) && (iteration > 5))
                    {
                        converged_cq[message_edge] = true;
                    }

                }
            }
        }
    }
    return;
}

void BpDecoderKL::check_serial_update(xt::xarray<int> * s_0, int iteration)
{
    xt::xarray<int> comm_table= {{0, 0, 0, 0},
                                 {0, 0, 1, 1},
                                 {0, 1, 0, 1},
                                 {0, 1, 1, 0}};

    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    { // for each check do

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        { // calculate  all incoming (q -> c) messages
            if (!((converged_qc[message_edge]) && (only_non_converged)))
            {
                xt::xarray<long double> q = p_initial;
                for (lemon::ListBpGraph::IncEdgeIt other_edge(g,g.blueNode(message_edge)); other_edge != lemon::INVALID; ++other_edge)
                {
                    if (other_edge != message_edge)
                    {   
                        xt::xarray<long double> r = xt::ones<long double>({2});

                        r(0) = pow((1 + m_cq[other_edge](0))/2,1.0/m_properties.m_alpha);
                        r(1) = pow((1 - m_cq[other_edge](0))/2,1.0/m_properties.m_alpha);

                        q(0) *= r(0);
                        for (size_t p = 1; p < 4; p++)
                        {
                            int sp = comm_table(p,edge_type[other_edge]);
                            q(p) *= r(sp);
                        }
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

                if (m_properties.m_alpha != 1.0)
                {
                    long double r_0 = pow((1 + m_cq[message_edge](0))/2.0,1.0/m_properties.m_alpha-1.0);
                    long double r_1 = pow((1 - m_cq[message_edge](0))/2.0,1.0/m_properties.m_alpha-1.0);
                    q_0 *= r_0;
                    q_1 *= r_1;
                }
                
                long double sum = q_0 + q_1;

                q_0 /= sum; q_1 /= sum;
                
                long double message = q_0 - q_1;

                long double old_message = m_qc[message_edge](0);

                m_qc[message_edge](0) = (1-m_properties.m_w) * old_message + m_properties.m_w*message;
                m_qc[message_edge](iteration) = (1-m_properties.m_w) * old_message + m_properties.m_w*message;

                long double diff = fabs(m_qc[message_edge](0) -  old_message);
                if ((diff < 1E-100) && (iteration > 5))
                {
                    converged_qc[message_edge] = true;
                }
            }
        }

        // calculate all outgoing (c -> q) messages
        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            //qubit = g.Node(bluemessage_edge)
            if (!((converged_cq[message_edge]) && (only_non_converged)))
            {
                long double message = std::pow(-1,s_0->at(check_label[check]));

                for (lemon::ListBpGraph::IncEdgeIt edge_to_qubit(g,check); edge_to_qubit != lemon::INVALID; ++edge_to_qubit)
                {
                    if (edge_to_qubit != message_edge)
                    {
                        message *= m_qc[edge_to_qubit](0);
                    }
                }

                
                long double old_message = m_cq[message_edge](0);
                m_cq[message_edge](0) = (1-m_properties.m_w)*old_message + m_properties.m_w*message;
                m_cq[message_edge](iteration) = (1-m_properties.m_w)*old_message + m_properties.m_w*message;

                long double diff = fabs(m_cq[message_edge](0) - old_message);
                if ((diff < 1E-100) && (iteration > 5))
                {
                    converged_cq[message_edge] = true;
                }
            }
        }
    }
    return;
}

void BpDecoderKL::check_sequential_update(xt::xarray<int> * s_0, int iteration)
{
    long double w = m_properties.m_w;
    long double alpha = m_properties.m_alpha;
    xt::xarray<int> comm_table= {{0, 0, 0, 0},
                                 {0, 0, 1, 1},
                                 {0, 1, 0, 1},
                                 {0, 1, 1, 0}};


    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    { // for each check do

        // calculate  all outgoing (c -> q) messages
        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            if (!((converged_cq[message_edge]) && (only_non_converged)))
            {
            //check = g.redNode(message_edge)
                long double message = 1;
                long double sign = std::pow(-1.0,s_0->at(check_label[check]));

                for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
                {
                    if (other_edge != message_edge)
                    {
                        message *= m_qc[other_edge](0);
                    }
                }
                message *= sign;
                long double old_message = m_cq[message_edge](0);
                m_cq[message_edge](0) = (1-w)*old_message + w*message;
                
                // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
                long double diff = fabs(m_cq[message_edge](0) - old_message);
                if ((diff < 1E-100) && (iteration > 5))
                {
                    converged_cq[message_edge] = true;
                }

            }
        }

        // update all neighboring qubits (q -> c) messages
        for (lemon::ListBpGraph::IncEdgeIt edge_to_qubit(g,check); edge_to_qubit != lemon::INVALID; ++edge_to_qubit)
        {
            for (lemon::ListBpGraph::IncEdgeIt outgoing_message_edge(g,g.blueNode(edge_to_qubit)); outgoing_message_edge != lemon::INVALID; ++outgoing_message_edge)
            { 
                if (!((converged_qc[outgoing_message_edge]) && (only_non_converged)))
                {
                    xt::xarray<long double> q = p_initial;
                    for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,g.blueNode(edge_to_qubit)); incoming_edge != lemon::INVALID; ++incoming_edge)
                    {
                        if (incoming_edge != outgoing_message_edge)
                        {
                            xt::xarray<long double> r = xt::ones<long double>({2});
                            if (alpha != 1.0)
                            {
                                r(0) = pow((1 + m_cq[incoming_edge](0))/2,1.0/alpha);
                                r(1) = pow((1 - m_cq[incoming_edge](0))/2,1.0/alpha);
                            }
                            else
                            {
                                r(0) = (1 + m_cq[incoming_edge](0))/2;
                                r(1) = (1 - m_cq[incoming_edge](0))/2;
                            }

                            q(0) *=  r(0);
                            
                            for (size_t p = 1; p < 4; p++)
                            {
                                int sp = comm_table(p,edge_type[incoming_edge]);
                                q(p) *= r(sp);
                            }
                        }
                    }


                    long double q_0 = q(0) + q(edge_type[outgoing_message_edge]);
                    long double q_1 = 0;
                    for (size_t p = 1; p < 4; p++)
                    {
                        if (p != edge_type[outgoing_message_edge])
                        {
                            q_1 += q(p);
                        }
                    }

                    if (alpha != 1.0)
                    {
                        long double r_0 = pow((1 + m_cq[outgoing_message_edge](0))/2.0,1.0/alpha-1.0);
                        long double r_1 = pow((1 - m_cq[outgoing_message_edge](0))/2.0,1.0/alpha-1.0);
                        q_0 *= r_0;
                        q_1 *= r_1;
                    }
                    
                    long double sum = q_0 + q_1;

                    q_0 /= sum; q_1 /= sum;
                    
                    long double message = q_0 - q_1;

                    long double old_message = m_qc[outgoing_message_edge](0);

                    m_qc[outgoing_message_edge](0) = (1-w)*old_message + w * message;
                    
                    long double diff = fabs(m_qc[outgoing_message_edge](0) -  old_message);
                    if ((diff < 1E-100) && (iteration > 5))
                    {
                        converged_qc[outgoing_message_edge] = true;
                    }
                }
            }
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
            xt::xarray<long double> r = xt::ones<long double>({2});
            if (alpha != 1.0)
            {
                r(0) = pow((1 + m_cq[incoming_edge](iteration))/2.0,1.0/alpha);
                r(1) = pow((1 - m_cq[incoming_edge](iteration))/2.0,1.0/alpha);
            }
            else
            {
                r(0) = (1 + m_cq[incoming_edge](iteration))/2.0;
                r(1) = (1 - m_cq[incoming_edge](iteration))/2.0;
            }
            
            
            q(0) *= r(0);
            for (size_t p = 1; p < 4; p++)
            {
                int sp = comm_table(p,edge_type[incoming_edge]);
                q(p) *= r(sp);
            }
        }
        q /= xt::sum(q);
        xt::row(marginals[qubit],iteration) = q;
        // m = q;        
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
        
        // xt::argmax(q,0)();
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}

void BpDecoderKL::marginals_and_hard_decision_serial(int iteration)
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
                q(0) = 1; q(1) = 0; q(2) = 0; q(3) = 0;
            }
        }
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            xt::xarray<long double> r = xt::ones<long double>({2});
            if (alpha != 1.0)
            {
                r(0) = pow((1 + m_cq[incoming_edge](0))/2.0,1.0/alpha);
                r(1) = pow((1 - m_cq[incoming_edge](0))/2.0,1.0/alpha);
            }
            else
            {
                r(0) = (1 + m_cq[incoming_edge](0))/2.0;
                r(1) = (1 - m_cq[incoming_edge](0))/2.0;
            }
            
            
            q(0) *= r(0);
            for (size_t p = 1; p < 4; p++)
            {
                int sp = comm_table(p,edge_type[incoming_edge]);
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



void BpDecoderKL::calculate_free_energy(int iteration)
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
    free_energy(iteration) = -f;
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

xt::xarray<long double> BpDecoderKL::get_free_energy()
{
    return free_energy;
}

} // end of namespace bp