#include <bp_gbp/decoder_bp2.hpp>

Bp2Decoder::Bp2Decoder(xt::xarray<int> H) : H(H),g(), qubit_label(g), check_label(g),edge_label(g), m_cq(g), m_qc(g),converged_cq(g), converged_qc(g), marginals(g)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    initialize_graph();

    only_non_converged = false;

    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);
}

void Bp2Decoder::initialize_graph()
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
                // edge_type[new_edge] = H(c,q);
                edge_label[new_edge] = n_edges;
                // if  (qubit_label[qubit] == 399)
                    // std::cout << "(q,c) = (" << q << "," << c << "), edge = " << edge_label[new_edge] << "\n";
                n_edges++;
            }
        }
    }

    std::cout << "initialized graph with n_c = " << n_c << " n_q = " << n_q << std::endl; 
}

void Bp2Decoder::initialize_bp(xt::xarray<long double> p_init, int max_iter)
{
    p_initial = p_init;
    max_iterations = max_iter;
    hard_decision =  xt::zeros<int>({max_iter+1,n_q});
    syndromes = xt::zeros<int>({max_iter+1,n_c});
    free_energy = xt::zeros<long double>({max_iter+1});

    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({max_iter,2});
        m_qc[edge] = xt::zeros<long double>({max_iter,2});

       xt::row(m_qc[edge],0) = p_initial;
    }

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({max_iter,2});
        xt::row(marginals[qubit],0) = p_initial;
    }
    is_initialized = true;
}

xt::xarray<int> Bp2Decoder::decode_bp(xt::xarray<int> s_0, long double w, long double alpha, int type_message, bool return_if_success, bool converged)
{
    only_non_converged = converged;
    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);
    if (is_initialized == false)
    {
        std::cerr << "BP not initialized, please run initialize_bp(xt::xarray<long double> p_init)" << std::endl;
    }
    // check_to_bit(&s_0, 0, 1.0, 0.0);

    int took_iter = max_iterations;
    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        if (type_message == 0)
        {
            check_to_bit(&s_0, iteration, w, 1);
            bit_to_check(iteration, w, 1);
            marginals_and_hard_decision(iteration, 1);
        }
        else if (type_message == 1)
        {
            check_to_bit_mfr(&s_0, iteration, w, alpha);
            bit_to_check(iteration, w, alpha);
            marginals_and_hard_decision(iteration, alpha);
        }
        else if (type_message == 10)
        {
            bit_serial_update(&s_0, iteration, w, 1);
            marginals_and_hard_decision_serial(iteration, 1);
        }
        else if (type_message == 3)
        {
            check_to_bit_to_check_mfr(&s_0, iteration, w, alpha);
            marginals_and_hard_decision(iteration, alpha);
        }
        else if (type_message == 4)
        {
            check_to_bit_fr(&s_0, iteration, w, alpha);
            bit_to_check(iteration, w, alpha);
            marginals_and_hard_decision(iteration, alpha);
        }
        else if (type_message == 5)
        {
            check_to_bit_to_check_fr(&s_0, iteration, w, alpha);
            marginals_and_hard_decision(iteration, alpha);
        }
        
        

        calculate_free_energy(iteration);

        xt::xarray<int> s = xt::ones_like(s_0);
        xt::xarray<int> hd =  xt::row(hard_decision,iteration);

        gf2_syndrome(&s, &hd, &H);

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

void Bp2Decoder::check_to_bit(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    xt::xarray<long double> FT = xt::ones<long double>({2,2});
    FT(1,1) = -1;

    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        xt::xarray<long double> message = xt::ones<long double>({2});

        int c = check_label[check];


        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            // std::cout << "**\nconverged_cq[message_edge] = " << converged_cq[message_edge] << "\nonly_non_converged = " << only_non_converged << "\n(converged_cq[message_edge]) && (only_non_converged) = " << ((converged_cq[message_edge]) && (only_non_converged)) << "\n!((converged_cq[message_edge]) && (only_non_converged)) = "<< !((converged_cq[message_edge]) && (only_non_converged)) <<"\n"; 
            if (!((converged_cq[message_edge]) && (only_non_converged)))
            {
                // std::cout << "y\n"; 
                message = {1.0,1.0};
                for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
                {
                    if (other_edge != message_edge)
                    {
                        if (iteration == 0)
                        {
                            message *= xt::linalg::dot(FT,xt::row(m_qc[other_edge],0));
                        }
                        else
                        {
                            message *= xt::linalg::dot(FT,xt::row(m_qc[other_edge],iteration-1));
                        }
                        
                    }
                }

                message = xt::linalg::dot(FT,message);


                if (s_0->at(c) == 1)
                {
                    message = xt::flip(message);
                }

                message /= xt::sum(message)();

                if (iteration == 1)
                {
                    xt::row(m_cq[message_edge],iteration) = message;
                }
                else
                {
                    xt::row(m_cq[message_edge],iteration) = (1-w) * xt::row(m_cq[message_edge],iteration-1) + w * message;
                }
                
                // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
                long double diff = fabs(m_cq[message_edge](iteration-1,0) - m_cq[message_edge](iteration,0));
                if ((diff < 1E-200) && (iteration > 5))
                {
                    converged_cq[message_edge] = true;
                }
            }
        }
    }
    return;
}

void Bp2Decoder::check_to_bit_mfr(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    xt::xarray<long double> FT = xt::ones<long double>({2,2});
    FT(1,1) = -1;

    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        xt::xarray<long double> message = xt::ones<long double>({2});

        int c = check_label[check];


        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            message = {1.0,1.0};
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    message *= xt::linalg::dot(FT,xt::row(m_qc[other_edge],iteration-1));
                }
            }

            message = xt::linalg::dot(FT,message);


            if (s_0->at(c) == 1)
            {
                message = xt::flip(message);
            }

            // here's the difference
            if (iteration > 1)
            {
                message *= xt::row(m_qc[message_edge],iteration-1) / p_initial;
            }
            

            // for (lemon::ListBpGraph::IncEdgeIt in_edge(g,g.blueNode(message_edge)); in_edge != lemon::INVALID; ++in_edge)
            // {
            //     // if (g.id(g.redNode(message_edge)) == 5)
            //     // {
            //     //     std::cout << "g.id(g.redNode(message_edge)) = " << g.id(g.redNode(message_edge)) << "\n";
            //     //     std::cout << "g.id(g.blueNode(message_edge)) = " << g.id(g.blueNode(message_edge)) << "\n";
            //     //     std::cout << "g.id(g.redNode(in_edge)) = " << g.id(g.redNode(in_edge)) << "\n\n";

            //     // }
            //     if ((g.id(g.redNode(in_edge)) != g.id(g.redNode(message_edge))) && (iteration > 1)) 
            //     {
            //         message *= xt::row(m_cq[in_edge],iteration-1);
            //     }
                
            // }

            message /= xt::sum(message)();

            if (iteration == 1)
            {
                xt::row(m_cq[message_edge],iteration) = message;
            }
            else
            {
                xt::row(m_cq[message_edge],iteration) = (1-w) * xt::row(m_cq[message_edge],iteration-1) + w * message;
            }
            
            // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
            // long double diff = abs(m_cq[message_edge](iteration-1) - m_cq[message_edge](iteration));
            // if ((diff < 1E-200) && (iteration > 2))
            // {
            //     converged_cq[message_edge] = true;
            // }
        }
    }
    return;
}


void Bp2Decoder::check_to_bit_fr(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    xt::xarray<long double> FT = xt::ones<long double>({2,2});
    FT(1,1) = -1;

    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        xt::xarray<long double> message = xt::ones<long double>({2});

        int c = check_label[check];


        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,check); message_edge != lemon::INVALID; ++message_edge)
        {
            message = {1.0,1.0};
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    message *= xt::linalg::dot(FT,xt::row(m_qc[other_edge],iteration-1));
                }
            }

            message = xt::linalg::dot(FT,message);


            if (s_0->at(c) == 1)
            {
                message = xt::flip(message);
            }


            // here's the difference
            if (iteration > 1)
            {
                message *= xt::pow(xt::row(m_cq[message_edge],iteration-1),1.0-1.0/alpha);

                // for (lemon::ListBpGraph::IncEdgeIt other_edge(g,check); other_edge != lemon::INVALID; ++other_edge)
                // {
                //     if (other_edge != message_edge)
                //     {
                //         message *= xt::pow(xt::row(m_cq[other_edge],iteration-1),1.0-1.0/alpha);
                //     }
                // }
            }
            

            // for (lemon::ListBpGraph::IncEdgeIt in_edge(g,g.blueNode(message_edge)); in_edge != lemon::INVALID; ++in_edge)
            // {
            //     // if (g.id(g.redNode(message_edge)) == 5)
            //     // {
            //     //     std::cout << "g.id(g.redNode(message_edge)) = " << g.id(g.redNode(message_edge)) << "\n";
            //     //     std::cout << "g.id(g.blueNode(message_edge)) = " << g.id(g.blueNode(message_edge)) << "\n";
            //     //     std::cout << "g.id(g.redNode(in_edge)) = " << g.id(g.redNode(in_edge)) << "\n\n";

            //     // }
            //     if ((g.id(g.redNode(in_edge)) != g.id(g.redNode(message_edge))) && (iteration > 1)) 
            //     {
            //         message *= xt::row(m_cq[in_edge],iteration-1);
            //     }
                
            // }

            message /= xt::sum(message)();

            if (iteration == 1)
            {
                xt::row(m_cq[message_edge],iteration) = message;
            }
            else
            {
                xt::row(m_cq[message_edge],iteration) = (1-w) * xt::row(m_cq[message_edge],iteration-1) + w * message;
            }
            
            // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
            long double diff = fabs(m_cq[message_edge](iteration-1,0) - m_cq[message_edge](iteration,0));
            if ((diff < 1E-200) && (iteration > 2))
            {
                converged_cq[message_edge] = true;
            }
        }
    }
    return;
}

void Bp2Decoder::bit_to_check(int iteration,long double w, long double alpha)
{
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> message = xt::ones<long double>({2});

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,qubit); message_edge != lemon::INVALID; ++message_edge)
        {
            // std::cout << "**\nconverged_qc[message_edge] = " << converged_qc[message_edge] << "\nonly_non_converged = " << only_non_converged << "\n(converged_qc[message_edge]) && (only_non_converged) = " << ((converged_qc[message_edge]) && (only_non_converged)) << "\n!((converged_qc[message_edge]) && (only_non_converged)) = "<< !((converged_qc[message_edge]) && (only_non_converged)) <<"\n"; 
            if (!((converged_qc[message_edge]) && (only_non_converged)))
            {
                // std::cout << "y\n";
                message = p_initial;
                message *= xt::pow(xt::row(m_cq[message_edge],iteration),alpha-1);
                for (lemon::ListBpGraph::IncEdgeIt other_edge(g,qubit); other_edge != lemon::INVALID; ++other_edge)
                {
                    if (other_edge != message_edge)
                    {
                        message *=  xt::pow(xt::row(m_cq[other_edge],iteration),alpha);
                        // message *=  xt::row(m_cq[other_edge],iteration);
                    }
                }

                message /= xt::sum(message)();

                xt::row(m_qc[message_edge],iteration) = (1-w) * xt::row(m_qc[message_edge],iteration-1) + w * message;
                // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
                long double diff = fabs(m_qc[message_edge](iteration-1,0) -  m_qc[message_edge](iteration,0));
                // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
                if ((diff < 1E-200) && (iteration > 5))
                {
                    converged_qc[message_edge] = true;
                }
            }
        }
    }
    return;
}

void Bp2Decoder::bit_serial_update(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    xt::xarray<long double> FT = xt::ones<long double>({2,2});
    FT(1,1) = -1;

    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    { // for each check do
        if ((iteration < 2) && (check_label[check] == 4))
        {
            std::cout << "** g.id(check) = " << g.id(check) << "\n";
            std::cout << "| check_label[check] = " << check_label[check] << "\n";
            for (lemon::ListBpGraph::IncEdgeIt incEdge(g,check); incEdge != lemon::INVALID; ++incEdge)
            {

                std::cout << "  -- g.id(incEdge) = " << g.id(incEdge) << "\n";
                std::cout << "  |  g.id(check) = " << g.id(check) << "\n";
                std::cout << "  |  g.id(g.baseNode(incEdge)) = " << g.id(g.baseNode(incEdge)) << "\n";
                std::cout << "  |  g.id(g.asRedNode(g.baseNode(incEdge))) = " << g.id(g.asRedNode(g.baseNode(incEdge))) << "\n";
                std::cout << "  |  g.id(g.runningNode(incEdge)) = " << g.id(g.runningNode(incEdge)) << "\n";
                std::cout << "  |  g.id(g.redNode(incEdge)) = " << g.id(g.redNode(incEdge))  << "\n";
                std::cout << "  |  g.id(g.blueNode(incEdge)) = " << g.id(g.blueNode(incEdge))  << "\n";
            }
        }
    }

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    { // for each qubit do
        if ((iteration < 2) && (qubit_label[qubit] == 399))
        {
            std::cout << "** g.id(qubit) = " << g.id(qubit) << "\n";
            std::cout << "| qubit_label[qubit] = " << qubit_label[qubit] << "\n";
            for (lemon::ListBpGraph::IncEdgeIt incEdge(g,qubit); incEdge != lemon::INVALID; ++incEdge)
            {

                std::cout << "  -- g.id(incEdge) = " << g.id(incEdge) << "\n";
                std::cout << "  |  g.id(qubit) = " << g.id(qubit) << "\n";
                std::cout << "  |  g.id(g.baseNode(incEdge)) = " << g.id(g.baseNode(incEdge)) << "\n";
                std::cout << "  |  g.id(g.asBlueNode(g.baseNode(incEdge))) = " << g.id(g.asBlueNode(g.baseNode(incEdge))) << "\n";
                std::cout << "  |  g.id(g.runningNode(incEdge)) = " << g.id(g.runningNode(incEdge)) << "\n";
                std::cout << "  |  g.id(g.redNode(incEdge)) = " << g.id(g.redNode(incEdge))  << "\n";
                std::cout << "  |  g.id(g.blueNode(incEdge)) = " << g.id(g.blueNode(incEdge))  << "\n";
            }
        }
        
    }



    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    { // for each qubit do
        // update incoming (c->q) messages
        for (lemon::ListBpGraph::IncEdgeIt incoming_message_edge(g,qubit); incoming_message_edge != lemon::INVALID; ++incoming_message_edge)
        {
            if ((iteration < 3) && (qubit_label[qubit] == 399))
                std::cout << "it " << iteration << " g.id(incoming_message_edge) =  " << g.id(incoming_message_edge) << "\n";
            if (converged_cq[incoming_message_edge] == false)
            {
                xt::xarray<long double> update_message = xt::ones<long double>({2});

                int c = check_label[g.redNode(incoming_message_edge)];


                // for (lemon::ListBpGraph::IncEdgeIt message_edge(g,g.redNode(incoming_message_edge)); message_edge != lemon::INVALID; ++message_edge)
                // if ((iteration < 3) && (qubit_label[qubit] == 399))
                //     {
                //         std::cout << "g.id(incoming_message_edge) =  " << g.id(incoming_message_edge) << "\n";
                //         std::cout << "c =  " << check_label[g.redNode(incoming_message_edge)] << "\n";
                //     }
                // {
                //     message = {1.0,1.0};
                for (lemon::ListBpGraph::IncEdgeIt check_incoming_edge(g,g.redNode(incoming_message_edge)); check_incoming_edge != lemon::INVALID; ++check_incoming_edge)
                {
                    // if ((iteration < 3) && (qubit_label[qubit] == 399))
                    // {
                    //     std::cout << "g.id(check_incoming_edge) =  " << g.id(check_incoming_edge) << "\n";
                    //     std::cout << "check_label[g.redNode(check_incoming_edge)] =  " << check_label[g.redNode(check_incoming_edge)] << "\n";
                    // }
                    // if (other_edge != message_edge)
                    if (check_incoming_edge != incoming_message_edge)
                    {
                        // message *= xt::linalg::dot(FT,xt::row(m_qc[check_incoming_edge],iteration-1));
                        update_message *= xt::linalg::dot(FT,xt::row(m_qc[check_incoming_edge],0));
                        if ((iteration < 3) && (qubit_label[qubit] == 399))
                        {
                            std::cout << "edge: " << g.id(check_incoming_edge) << "\n";
                            std::cout << "check: " << check_label[g.redNode(incoming_message_edge)] << "\n";
                            std::cout << "xt::row(m_qc[check_incoming_edge],0) = " << xt::row(m_qc[check_incoming_edge],0) << "\n";
                        }
                            
                    }
                }
                update_message = xt::linalg::dot(FT,update_message);
                // if ((iteration < 3) && (qubit_label[qubit] == 399))
                //     std::cout << "update_message = " << update_message << "\n";

                if (s_0->at(c) == 1)
                {
                    update_message = xt::flip(update_message);
                }

                update_message /= xt::sum(update_message)();
                
                // if ((iteration < 3) && (qubit_label[qubit] == 399))
                //     std::cout << "update_message = " << update_message << "\n";

                if (iteration == 1)
                {
                    xt::row(m_cq[incoming_message_edge],0) = update_message;
                }
                else
                {
                    xt::xarray<long double> old_message = xt::row(m_cq[incoming_message_edge],0);
                    
                    xt::row(m_cq[incoming_message_edge],0) = (1-w) * old_message + w * update_message;
                    long double diff = fabs(m_cq[incoming_message_edge](0,0) -  old_message(0)) + fabs(m_cq[incoming_message_edge](0,1) -  old_message(1));
                    // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
                    if ((diff < 1E-15) && (iteration > 5))
                    {
                        converged_cq[incoming_message_edge] = true;
                    }

                }


                // if (iteration == 1)
                // {
                //     xt::row(m_cq[message_edge],iteration) = message;
                // }
                // else
                // {
                //     xt::row(m_cq[message_edge],iteration) = (1-w) * xt::row(m_cq[message_edge],iteration-1) + w * message;
                // }
                
                
                if ((iteration < 3) && (qubit_label[qubit] == 399))
                std::cout << "it " << iteration << " q " << qubit_label[qubit] << " <- c " << check_label[g.redNode(incoming_message_edge)] << " m_cq = " << xt::row(m_cq[incoming_message_edge],0) << std::endl;
                // long double diff = abs(m_cq[message_edge](iteration-1) - m_cq[message_edge](iteration));
                // if ((diff < 1E-200) && (iteration > 2))
                // {
                //     converged_cq[message_edge] = true;
                // }
            // }
            }

        }

        // calculate outgoing (q->c) messages
        
        
        xt::xarray<long double> update_message = xt::ones<long double>({2});

        for (lemon::ListBpGraph::IncEdgeIt outgoing_message_edge(g,qubit); outgoing_message_edge != lemon::INVALID; ++outgoing_message_edge)
        {
            if (converged_qc[outgoing_message_edge] == false)
            {
                update_message = p_initial;
            // update_message *= xt::pow(xt::row(m_cq[outgoing_message_edge],iteration),alpha-1);
                // update_message *= xt::pow(xt::row(m_cq[outgoing_message_edge],0),alpha-1);

                for (lemon::ListBpGraph::IncEdgeIt other_edge(g,qubit); other_edge != lemon::INVALID; ++other_edge)
                {
                    if (other_edge != outgoing_message_edge)
                    {
                        // update_message *=  xt::pow(xt::row(m_cq[other_edge],iteration),alpha);
                        // update_message *=  xt::pow(xt::row(m_cq[other_edge],0),alpha);
                        update_message *=  xt::row(m_cq[other_edge],0);
                    }
                }

                // update_message /= xt::sum(update_message)();

                xt::xarray<long double> old_message = xt::row(m_qc[outgoing_message_edge],0);
                xt::row(m_qc[outgoing_message_edge],0) = (1-w) * old_message + w * update_message;

                // xt::row(m_qc[outgoing_message_edge],iteration) = (1-w) * xt::row(m_qc[outgoing_message_edge],iteration-1) + w * update_message;
                // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
                // long double diff = abs(m_qc[message_edge](iteration-1) -  m_qc[message_edge](iteration));
                long double diff = fabs(m_qc[outgoing_message_edge](0,0) -  old_message(0)) + fabs(m_qc[outgoing_message_edge](0,1) -  old_message(1));
                
                // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
                if ((diff < 1E-15) && (iteration > 5))
                {
                    converged_qc[outgoing_message_edge] = true;
                }
                if ((iteration < 3) && (qubit_label[qubit] == 399))
                std::cout << "it " << iteration << " q " << qubit_label[qubit] << " -> c " << check_label[g.redNode(outgoing_message_edge)] << " m_cq = " << xt::row(m_qc[outgoing_message_edge],0) << std::endl;
            }
        }
        
    }

    return;
}

void Bp2Decoder::check_to_bit_to_check_mfr(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    xt::xarray<long double> FT = xt::ones<long double>({2,2});
    FT(1,1) = -1;

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        
        // calculate check to bit messages from each incoming check
        for (lemon::ListBpGraph::IncEdgeIt inc_edge(g,qubit); inc_edge != lemon::INVALID; ++inc_edge)
        {
            xt::xarray<long double> message = xt::ones<long double>({2});

        int c = check_label[g.redNode(inc_edge)];


        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,g.redNode(inc_edge)); message_edge != lemon::INVALID; ++message_edge)
        {
            message = {1.0,1.0};
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,g.redNode(inc_edge)); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    message *= xt::linalg::dot(FT,xt::row(m_qc[other_edge],iteration-1));
                }
            }

            message = xt::linalg::dot(FT,message);


            if (s_0->at(c) == 1)
            {
                message = xt::flip(message);
            }

            //  here's the difference
            if (iteration > 1)
            {
                message *= xt::row(m_qc[message_edge],iteration-1) / p_initial;
            }

            message /= xt::sum(message)();

            if (iteration == 1)
            {
                xt::row(m_cq[message_edge],iteration) = message;
            }
            else
            {
                xt::row(m_cq[message_edge],iteration) = (1-w) * xt::row(m_cq[message_edge],iteration-1) + w * message;
            }
            
            // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
            // long double diff = abs(m_cq[message_edge](iteration-1) - m_cq[message_edge](iteration));
            // if ((diff < 1E-200) && (iteration > 2))
            // {
            //     converged_cq[message_edge] = true;
            // }
        }

        }


        xt::xarray<long double> message = xt::ones<long double>({2});

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,qubit); message_edge != lemon::INVALID; ++message_edge)
        {
            message = p_initial;
            message *= xt::pow(xt::row(m_cq[message_edge],iteration),alpha-1);
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,qubit); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    message *=  xt::pow(xt::row(m_cq[other_edge],iteration),alpha);
                    // message *=  xt::row(m_cq[other_edge],iteration);
                }
            }

            message /= xt::sum(message)();

            xt::row(m_qc[message_edge],iteration) = (1-w) * xt::row(m_qc[message_edge],iteration-1) + w * message;
            // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
            // long double diff = abs(m_qc[message_edge](iteration-1) -  m_qc[message_edge](iteration));
            // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
            // if ((diff < 1E-200) && (iteration > 2))
            // {
            //     converged_qc[message_edge] = true;
            // }
        }
    }

    return;
}

void Bp2Decoder::check_to_bit_to_check_fr(xt::xarray<int> * s_0, int iteration,long double w, long double alpha)
{
    xt::xarray<long double> FT = xt::ones<long double>({2,2});
    FT(1,1) = -1;

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        
        // calculate check to bit messages from each incoming check
        for (lemon::ListBpGraph::IncEdgeIt inc_edge(g,qubit); inc_edge != lemon::INVALID; ++inc_edge)
        {
            xt::xarray<long double> message = xt::ones<long double>({2});

        int c = check_label[g.redNode(inc_edge)];


        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,g.redNode(inc_edge)); message_edge != lemon::INVALID; ++message_edge)
        {
            message = {1.0,1.0};
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,g.redNode(inc_edge)); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    message *= xt::linalg::dot(FT,xt::row(m_qc[other_edge],iteration-1));
                }
            }

            message = xt::linalg::dot(FT,message);


            if (s_0->at(c) == 1)
            {
                message = xt::flip(message);
            }

            //  here's the difference
            if (iteration > 1)
            {
                message *= xt::pow(xt::row(m_cq[message_edge],iteration-1),1.0-1.0/alpha);
            }

            message /= xt::sum(message)();

            if (iteration == 1)
            {
                xt::row(m_cq[message_edge],iteration) = message;
            }
            else
            {
                xt::row(m_cq[message_edge],iteration) = (1-w) * xt::row(m_cq[message_edge],iteration-1) + w * message;
            }
            
            // std::cout <<  "m_cq[message_edge] = " << m_cq[message_edge] << std::endl;
            // long double diff = abs(m_cq[message_edge](iteration-1) - m_cq[message_edge](iteration));
            // if ((diff < 1E-200) && (iteration > 2))
            // {
            //     converged_cq[message_edge] = true;
            // }
        }

        }


        xt::xarray<long double> message = xt::ones<long double>({2});

        for (lemon::ListBpGraph::IncEdgeIt message_edge(g,qubit); message_edge != lemon::INVALID; ++message_edge)
        {
            message = p_initial;
            message *= xt::pow(xt::row(m_cq[message_edge],iteration),alpha-1);
            for (lemon::ListBpGraph::IncEdgeIt other_edge(g,qubit); other_edge != lemon::INVALID; ++other_edge)
            {
                if (other_edge != message_edge)
                {
                    message *=  xt::pow(xt::row(m_cq[other_edge],iteration),alpha);
                    // message *=  xt::row(m_cq[other_edge],iteration);
                }
            }

            message /= xt::sum(message)();

            xt::row(m_qc[message_edge],iteration) = (1-w) * xt::row(m_qc[message_edge],iteration-1) + w * message;
            // std::cout <<  "m_qc[message_edge] = " << m_qc[message_edge] << std::endl;
            // long double diff = abs(m_qc[message_edge](iteration-1) -  m_qc[message_edge](iteration));
            // std::cout << "iteration = " << iteration << " diff = " << diff << "\n";
            // if ((diff < 1E-200) && (iteration > 2))
            // {
            //     converged_qc[message_edge] = true;
            // }
        }
    }

    return;
}

void Bp2Decoder::marginals_and_hard_decision(int iteration, long double alpha)
{

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        // if (qubit_label[qubit] == 4)
        // {
        //     std::cout << "qubit 4: iteration " << iteration <<"\n";
        // }
        xt::xarray<long double> marginal = p_initial;
       
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            marginal *=  xt::pow(xt::row(m_cq[incoming_edge],iteration),alpha);
            // marginal *=  xt::row(m_cq[incoming_edge],iteration);
            //  if (qubit_label[qubit] == 4)
            // {
            //     std::cout << " edge_label[incoming_edge] =  " << edge_label[incoming_edge] << "\n";
            //     std::cout << " xt::row(m_cq[incoming_edge],iteration) =  " << xt::row(m_cq[incoming_edge],iteration) << "\n";
            // }
        }
        marginal /= xt::sum(marginal);
        xt::row(marginals[qubit],iteration) = marginal;
        // m = q;
        int hd =  xt::argmax(marginal,0)();
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}

void Bp2Decoder::marginals_and_hard_decision_serial(int iteration, long double alpha)
{

    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        // if (qubit_label[qubit] == 4)
        // {
        //     std::cout << "qubit 4: iteration " << iteration <<"\n";
        // }
        xt::xarray<long double> marginal = p_initial;
       
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            marginal *=  xt::pow(xt::row(m_cq[incoming_edge],0),alpha);
            // marginal *=  xt::row(m_cq[incoming_edge],iteration);
            //  if (qubit_label[qubit] == 4)
            // {
            //     std::cout << " edge_label[incoming_edge] =  " << edge_label[incoming_edge] << "\n";
            //     std::cout << " xt::row(m_cq[incoming_edge],iteration) =  " << xt::row(m_cq[incoming_edge],iteration) << "\n";
            // }
        }
        marginal /= xt::sum(marginal);
        xt::row(marginals[qubit],iteration) = marginal;
        // m = q;
        int hd =  xt::argmax(marginal,0)();
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}

void Bp2Decoder::calculate_free_energy(int iteration)
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


xt::xarray<long double> Bp2Decoder::get_m_qc()
{
    xt::xarray<long double> messages = xt::zeros<long double>({n_edges,max_iterations,2});
    
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        auto view = xt::view(messages,edge_label[edge],xt::all(),xt::all());

        view = m_qc[edge];
    }

    return messages;
}

xt::xarray<long double> Bp2Decoder::get_m_cq()
{
xt::xarray<long double> messages = xt::zeros<long double>({n_edges,max_iterations,2});
    
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        auto view = xt::view(messages,edge_label[edge],xt::all(),xt::all());
        view = m_cq[edge];
    }

    return messages;
}


xt::xarray<long double> Bp2Decoder::get_marginals()
{
    xt::xarray<long double> marginals_to_return = xt::zeros<long double>({n_q,max_iterations,2});
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        auto view_q = xt::view(marginals_to_return,qubit_label[qubit],xt::all(),xt::all());
        view_q = marginals[qubit];
    }
    return marginals_to_return;
}


xt::xarray<int> Bp2Decoder::get_hard_decisions()
{
    xt::xarray<int> hard_decisions_to_return = xt::zeros<int>({max_iterations,n_q});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_hd = xt::view(hard_decisions_to_return,it,xt::all());
        view_hd = xt::row(hard_decision,it);
    }
    return hard_decisions_to_return;
}

xt::xarray<int> Bp2Decoder::get_syndromes()
{
    xt::xarray<int> syndromes_to_return = xt::zeros<int>({max_iterations,n_c});
    for (int it = 0; it < max_iterations; it++)
    {
        auto view_s = xt::view(syndromes_to_return,it,xt::all());
        view_s = xt::row(syndromes,it);
    }
    return syndromes_to_return;
}

xt::xarray<bool> Bp2Decoder::get_converged_cq()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_cq[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<bool> Bp2Decoder::get_converged_qc()
{
    xt::xarray<bool> conv_ret = xt::zeros<bool>({n_edges});
    for (int edge = 0; edge < n_edges; edge++)
    {
        conv_ret(edge) = converged_qc[g.edgeFromId(edge)];
    }
    return conv_ret;
}

xt::xarray<int> Bp2Decoder::get_check_and_qubit(int edge)
{
    lemon::ListBpGraph::Edge e = g.edgeFromId(edge);
    xt::xarray<int> cq = {check_label[g.redNode(e)],qubit_label[g.blueNode(e)]};
    return cq;
}

xt::xarray<long double> Bp2Decoder::get_free_energy()
{
    return free_energy;
}