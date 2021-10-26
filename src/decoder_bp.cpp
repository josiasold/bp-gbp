#include <bp_gbp/decoder_bp.hpp>

namespace bp{

BpDecoder::BpDecoder(xt::xarray<int> H) : H(H), qubit_label(g), check_label(g),edge_label(g), edge_type(g), m_cq(g), m_qc(g), m_cq_current(g), m_qc_current(g),converged_cq(g), converged_qc(g), erased(g), marginals(g)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    initialize_graph();
    fill_c_factors();
    lemon::mapFill(g,converged_cq,false);
    lemon::mapFill(g,converged_qc,false);
    erasure_channel = false;
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


    std::cout << "initialized graph with n_c = " << n_c << " n_q = " << n_q << " n_edges = " << n_edges << std::endl; 
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

void BpDecoder::initialize_bp(xt::xarray<long double> p_init, int t_max_iter, long double t_w, long double t_alpha, int t_type, bool t_return_if_success, bool t_only_nonconverged_edges)
{
    took_iterations = 0;
    p_initial = p_init;
    max_iterations = t_max_iter;

    m_properties.m_max_iter = t_max_iter;
    m_properties.m_w = t_w;
    m_properties.m_alpha = t_alpha;
    m_properties.m_type = t_type;
    m_properties.m_return_if_success = t_return_if_success;
    m_properties.m_only_nonconverged_edges = t_only_nonconverged_edges;

    m_schedule_bp = t_type % 10;
    m_reweight_bp = t_type/10;

    hard_decision =  xt::zeros<int>({t_max_iter+1,n_q});
    syndromes = xt::zeros<int>({t_max_iter+1,n_c});
    free_energy = xt::zeros<long double>({t_max_iter});

    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        m_cq[edge] = xt::zeros<long double>({t_max_iter,4});
        m_qc[edge] = xt::zeros<long double>({t_max_iter,4});

        m_cq_current[edge] = p_initial;
        m_qc_current[edge] = p_initial;
        
        xt::row(m_qc[edge],0) = p_initial;

    }
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        marginals[qubit] = xt::ones<long double>({t_max_iter,4});
    }
    is_initialized = true;
}

void BpDecoder::initialize_bp(xt::xarray<int> * t_s_0)
{
    s_0 = *t_s_0;
    took_iterations = 0;
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        if (erasure_channel)
        {
            if (erased[edge] == false)
            {
                xt::xarray<long double> new_p_init = {1.0,0.0,0.0,0.0};
                xt::row(m_qc[edge],0) = new_p_init;
            }
            else
            {
                xt::xarray<long double> new_p_init = p_initial;
                new_p_init(0) = 0;
                new_p_init /= xt::sum(new_p_init);
                xt::row(m_qc[edge],0) = new_p_init;
            }
            
        }
        else
        {
             xt::row(m_qc[edge],0) = p_initial;
        }
        m_cq_current[edge] = p_initial;
        m_qc_current[edge] = p_initial;
    }
}


void BpDecoder::initialize_erasures(xt::xarray<int> * erasures)
{
    erasure_channel = true;
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        int qubit = qubit_label[g.blueNode(edge)];
        if (erasures->at(qubit) == 0)
        {
            xt::xarray<long double> new_p_init = {1.0,0.0,0.0,0.0};
            xt::row(m_qc[edge],0) = new_p_init;
            converged_qc[edge] = true;
            converged_cq[edge] = true;
            erased[edge] = false;
            marginals[g.blueNode(edge)](0,0) = 1.0;
            for (int i = 1; i < 4; i++) marginals[g.blueNode(edge)](0,i) = 0.0;
        }
        else
        {
            xt::xarray<long double> new_p_init = p_initial;
            new_p_init(0) = 0;
            new_p_init /= xt::sum(new_p_init);
            xt::row(m_qc[edge],0) = new_p_init;
            converged_qc[edge] = false;
            converged_cq[edge] = false;
            erased[edge] = true;
        }
    }
}

xt::xarray<int> BpDecoder::decode_bp(xt::xarray<int> t_s_0)
{
    if (is_initialized == false)
    {
        std::cerr << "BP not initialized, please run initialize_bp(xt::xarray<long double> p_init)" << std::endl;
    }
    int took_iter = max_iterations;
    // 0: parallel
    // 1: bit serial
    // 2: check serial
    // 3: bit sequential
    // 4: check sequential
    // 10,11,12,13,14: fractional
    // 20,21,22,23,24: urw
    // 30,31,32,33,34: memory

    if (m_reweight_bp == 1)
    {
        m_fractional = true;
    }
    else if (m_reweight_bp == 2)
    {
        m_urw = true;
    }
    else if (m_reweight_bp == 3)
    {
        m_memory = true;
    }

    initialize_bp(&t_s_0);
    check_to_bit(0);

    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        if (m_schedule_bp == 0) // parallel
        {
            check_to_bit(iteration);
            bit_to_check(iteration);
            marginals_and_hard_decision(iteration);
        }
        else if (m_schedule_bp == 1) // bit serial
        {
            bit_serial_update(iteration);
            marginals_and_hard_decision_serial(iteration);
        }
        else if (m_schedule_bp == 2) // check serial
        {
            check_serial_update(iteration);
            marginals_and_hard_decision_serial(iteration);
        }
        else if (m_schedule_bp == 3) // bit sequential
        {
            bit_sequential_update(iteration);
            marginals_and_hard_decision_serial(iteration);
        }
        else if (m_schedule_bp == 4) // check sequentiaö
        {
            check_sequential_update(iteration);
            marginals_and_hard_decision_serial(iteration);
        }

        xt::xarray<int> s = xt::ones_like(s_0);
        xt::xarray<int> hd =  xt::row(hard_decision,iteration);

        gf4_syndrome(&s, &hd, &H);

        calculate_free_energy(&s_0, iteration);

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


// Updates a sinlge edge with a given direction (check to qubit = "cq", qubit to check = "qc")
void BpDecoder::update_edge(const lemon::ListBpGraph::Edge& edge, std::string direction, int iteration)
{
    if (direction == "cq")
    {
        check_to_bit_single(edge, iteration);
    }
    else if (direction == "qc")
    {
        bit_to_check_single(edge, iteration);
    }
    return;
}


void BpDecoder::check_to_bit_single(const lemon::ListBpGraph::Edge& message_edge, int iteration)
{
    long double alpha = m_properties.m_alpha;
    long double w = m_properties.m_w;
    xt::xarray<long double> FT = xt::ones<long double>({4,4});
    FT(1,1) = -1;FT(1,3) = -1;FT(2,2) = -1;FT(2,3) = -1;FT(3,1) = -1;FT(3,2) = -1;

    xt::xarray<long double> message = {1.0,1.0,1.0,1.0};
    xt::xarray<long double> permuted_message = {1.0,1.0,1.0,1.0};
    xt::xarray<int> permutation = {0,1,2,3};


    message.fill(1.0);
    permuted_message.fill(1.0);
    for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,g.u(message_edge)); incoming_edge != lemon::INVALID; ++incoming_edge)
    {
        if (message_edge != incoming_edge)
        {
            for (int i = 0; i<4; i++)
            {
                int permutation_index = gf4_mul(edge_type[incoming_edge],gf4_conj(i));
                permutation(i) = permutation_index;
                long double frac_message = 1;
                if (m_fractional && iteration > 1)
                {
                    frac_message = pow(m_cq_current[message_edge](permutation_index),1.0-1.0/alpha);
                }
                if (iteration > 0)
                   { 
                       permuted_message(i) = m_qc_current[incoming_edge](permutation_index) * frac_message;
                    }
                else
                    {
                        permuted_message(i) = m_qc_current[incoming_edge](permutation_index);
                    }
            }
            xt::xarray<long double> tmp = xt::linalg::dot(FT,permuted_message);
            message *= tmp;
        }
    }
    xt::xarray<long double> tmp = xt::linalg::dot(FT,message);
    message = tmp;
    // std::cout << "message = " << message << "\n";
    // message /= xt::sum(message);
    xt::xarray<long double> update_message = {1.0,1.0,1.0,1.0};

    for (int p = 0; p < 4; p++)
    {
        int i1 = (2*s_0.at(check_label[g.redNode(message_edge)])) ^ gf4_mul(edge_type[message_edge],gf4_conj(p));
        int i2 = i1^1;
        update_message(p) = 0.5 * (message(i1) + message(i2));
    }
    update_message /= xt::sum(update_message);

    if (iteration > 0)
    {
        xt::xarray<long double> old_message = m_cq_current[message_edge];
        xt::row(m_cq[message_edge],iteration) = (1-w) * old_message + w * update_message;
        m_cq_current[message_edge] = (1-w) * old_message + w * update_message;
    }
    else
    {
        m_cq_current[message_edge] =  update_message;
    }
    if ( xt::row(m_cq[message_edge],iteration) ==  xt::row(m_cq[message_edge],iteration-1))
    {
        converged_cq[message_edge] = true;
    }
    return;
}

void BpDecoder::bit_to_check_single(const lemon::ListBpGraph::Edge& message_edge, int iteration)
{
    long double w = m_properties.m_w;

    xt::xarray<long double> update_message = p_initial;

    for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,g.v(message_edge)); incoming_edge != lemon::INVALID; ++incoming_edge)
    {
        if (message_edge != incoming_edge)
        {
            update_message *= m_cq_current[incoming_edge];
        }
    }

    if (m_urw)
    {
        update_message = xt::pow(m_cq_current[message_edge],m_properties.m_alpha-1.0) * xt::pow(update_message,m_properties.m_alpha);
    }
    if (m_memory)
    {
        update_message = xt::pow(m_cq_current[message_edge],1.0/m_properties.m_alpha-1.0) * xt::pow(update_message,1.0/m_properties.m_alpha);
    }

    update_message /= xt::sum(update_message);
    xt::xarray<long double> old_message =  m_qc_current[message_edge];

    xt::row(m_qc[message_edge],iteration) =  (1-w) * old_message + w * update_message;
    m_qc_current[message_edge]=  (1-w) * old_message + w * update_message;

    if (xt::row(m_qc[message_edge],iteration) == xt::row(m_qc[message_edge],iteration-1))
    {
        converged_qc[message_edge] = true;
    }
    return;
}

// full update of all outgoing messages of given check
void BpDecoder::check_update(const lemon::ListBpGraph::RedNode& check, int iteration)
{
    for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,check);outgoing_edge != lemon::INVALID; ++outgoing_edge)
    {
        update_edge(outgoing_edge,"cq", iteration);
    }
    return;
}

// full update of all outgoing messages of given qubit
void BpDecoder::qubit_update(const lemon::ListBpGraph::BlueNode& qubit, int iteration)
{
    for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,qubit);outgoing_edge != lemon::INVALID; ++outgoing_edge)
    {
        update_edge(outgoing_edge,"qc", iteration);
    }
    return;
}


// Parallel Update Routines

// updates all check_to_bit edges
void BpDecoder::check_to_bit(int iteration)
{
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        check_to_bit_single(edge, iteration);
    }
    return;
}

// updates all bit to check messages
void BpDecoder::bit_to_check(int iteration)
{
    // std::cout << "** bit to check" << "\n";
    for (lemon::ListBpGraph::EdgeIt edge(g); edge != lemon::INVALID; ++edge)
    {
        // std::cout << "g.id(edge) = " << g.id(edge) << "\n";
        bit_to_check_single(edge, iteration);
    }
    return;
}

// Serial Update Routines

void BpDecoder::check_serial_update(int iteration)
{
    // for every check
    for (lemon::ListBpGraph::RedNodeIt check(g); check != lemon::INVALID; ++check)
    {
        // update incoming messages
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,check); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            bit_to_check_single(incoming_edge, iteration);
        }
        //update outgoing messages
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,check); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            check_to_bit_single(outgoing_edge, iteration);
        }
    }
    
    return;
}

void BpDecoder::bit_serial_update(int iteration)
{
    // for every qubit
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        // update incoming messages
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
            check_to_bit_single(incoming_edge, iteration);
        }
        //update outgoing messages
        for (lemon::ListBpGraph::IncEdgeIt outgoing_edge(g,qubit); outgoing_edge != lemon::INVALID; ++outgoing_edge)
        {
            bit_to_check_single(outgoing_edge, iteration);
        }
    }
    
    return;
}

// Sequential Update Routines

void BpDecoder::check_sequential_update(int iteration)
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

void BpDecoder::bit_sequential_update(int iteration)
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


void BpDecoder::marginals_and_hard_decision(int iteration)
{
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = p_initial;
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {

            q *= xt::row(m_cq[incoming_edge],iteration);

        }
        if (m_fractional || m_urw)
        {
            q = xt::pow(q,m_properties.m_alpha);
        }
        q /= xt::sum(q);
        xt::row(marginals[qubit],iteration) = q;

        int hd =  xt::argmax(q,0)();
        hard_decision(iteration,qubit_label[qubit]) = hd;
    }
}

void BpDecoder::marginals_and_hard_decision_serial(int iteration)
{
    for (lemon::ListBpGraph::BlueNodeIt qubit(g); qubit != lemon::INVALID; ++qubit)
    {
        xt::xarray<long double> q = p_initial;
        for (lemon::ListBpGraph::IncEdgeIt incoming_edge(g,qubit); incoming_edge != lemon::INVALID; ++incoming_edge)
        {
                q *= m_cq_current[incoming_edge];
        }
        if (m_fractional || m_urw)
        {
            q = xt::pow(q,m_properties.m_alpha);
        }
        q /= xt::sum(q);

        xt::row(marginals[qubit],iteration) = q;

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

} // end of namespace bp