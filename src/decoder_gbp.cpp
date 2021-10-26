#include <bp_gbp/decoder_gbp.hpp>

void print(const std::vector<std::vector<int>>& v) {
  std::cout << "{ ";
  for (const auto& p : v) {
    std::cout << "(";
    for (const auto& e : p) {
      std::cout << e << " ";
    }
    std::cout << ") ";
  }
  std::cout << "}" << std::endl;
}

auto product(const std::vector<std::vector<int>>& lists) {
  std::vector<std::vector<int>> result;
  if (std::find_if(std::begin(lists), std::end(lists), 
    [](auto e) -> bool { return e.size() == 0; }) != std::end(lists)) {
    return result;
  }
  for (auto& e : lists[0]) {
    result.push_back({ e });
  }
  for (size_t i = 1; i < lists.size(); ++i) {
    std::vector<std::vector<int>> temp;
    for (auto& e : result) {
      for (auto f : lists[i]) {
        auto e_tmp = e;
        e_tmp.push_back(f);
        temp.push_back(e_tmp);
      }
    }
    result = temp;
  }
  return result;
} //https://rosettacode.org/wiki/Cartesian_product_of_two_or_more_lists#C.2B.2B

// GbpDecoder constructor
GbpDecoder::GbpDecoder(xt::xarray<int> H, int max_iterations,int n_checks_per_r0, int rg_type) : H(H), max_iterations(max_iterations), RG(H, n_checks_per_r0, rg_type)
{
  n_c = H.shape(0);
  n_q = H.shape(1);
  rank_H = gf4_rank(H.data(),n_c,n_q);
  syndromes = xt::ones<int>({max_iterations,n_c})*-1;
  hard_decisions = xt::zeros<int>({max_iterations,n_q});
  
  for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
  { 
        // construct message dummy
        variable_type::coordinate_map coord_map;
        dimension_type::label_list dim_list = {};
        int n_target_qubits = RG.region_qubits[RG.rg.target(edge)].size();
        for (int i = 0; i < n_target_qubits; i++)
        {
            std::string qubit_name = std::to_string(RG.region_qubits[RG.rg.target(edge)](i));
            const char *cstr = qubit_name.c_str();
            coord_map[cstr] = xf::axis({"0", "1"});
            dim_list.push_back(cstr);
        }
        std::vector<int> shape(n_target_qubits, 2);
        xt::xarray<long double> md;
        md.resize(shape);
        md.fill(0.5);
       
       variable_type message_dummy(md,coord_map,dim_list);

        //    std::cout << "message_dummy = \n" << message_dummy << "\n";

        std::vector<variable_type> messages_iter(max_iterations+1);
        RG.message[edge]=messages_iter;
        for (int i = 0; i<max_iterations+1;i++)
        {
        RG.message[edge][i] = message_dummy;
        }

        // construct message_base dummy
        variable_type::coordinate_map coord_map_base;
        dimension_type::label_list dim_list_base = {};
        int n_source_qubits = RG.region_qubits[RG.rg.source(edge)].size();
        for (int i = 0; i < n_source_qubits; i++)
        {
            std::string qubit_name = std::to_string(RG.region_qubits[RG.rg.source(edge)](i));
            const char *cstr = qubit_name.c_str();
            coord_map_base[cstr] = xf::axis({"0", "1"});
            dim_list_base.push_back(cstr);
        }
        std::vector<int> shape_base(n_source_qubits, 2);
        
        md.resize(shape_base);
        md.fill(1.0);
       
       variable_type message_base_dummy(md,coord_map_base,dim_list_base);
        //    std::cout << "message_base_dummy = \n" << message_base_dummy << "\n";
       
        RG.message_base[edge] = message_base_dummy;

        // construct belief_dummy (for source: same variables as message_base, for target: same as message)
        RG.belief_base[RG.rg.source(edge)] = message_base_dummy;
        RG.belief_base[RG.rg.target(edge)] = message_dummy;

        
        // get dim_of_vars_to_marg
        auto dimension_map = message_base_dummy.dimension_labels();

        std::vector<size_t> dim_of_vars_to_marg;
        std::vector<size_t> dim_names_of_vars(message_base_dummy.dimension());
        size_t j = 0;

        for (auto it = dimension_map.begin(); it != dimension_map.end(); ++it)
        {
            xt::xarray<int> dim_name = std::stoi(it[0].c_str());

            if (xt::any(xt::isin(RG.vars_to_marginalize[edge], dim_name)))
            {
                dim_of_vars_to_marg.push_back(j);
            }
            j++;
        }

        RG.dim_of_vars_to_marg[edge] = dim_of_vars_to_marg;
    }

  marginals.resize(max_iterations);
  
  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
      marginals[iteration].resize(n_q);
      for (int p = 0; p<n_q; p++)
      {
          marginals[iteration][p] = xt::ones<long double>({2});
      }
  }

    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        std::vector<variable_type> beliefs_iter(max_iterations+1);
        RG.belief[region]=beliefs_iter;
        RG.belief[region][0] = RG.belief_base[region];
    }
  
    lemon::mapFill(RG.rg,RG.edge_converged,false);


}

GbpDecoder::GbpDecoder(xt::xarray<int> H, int max_iterations,int n_checks_per_r0, xt::xarray<int> check_list, int rg_type) : H(H), max_iterations(max_iterations), RG(H, n_checks_per_r0, check_list, rg_type)
{
  n_c = H.shape(0);
  n_q = H.shape(1);
  rank_H = gf4_rank(H.data(),n_c,n_q);
  syndromes = xt::ones<int>({max_iterations,n_c})*-1;
  hard_decisions = xt::zeros<int>({max_iterations,n_q});
  for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
  { 
        // construct message dummy
        variable_type::coordinate_map coord_map;
        dimension_type::label_list dim_list = {};
        int n_target_qubits = RG.region_qubits[RG.rg.target(edge)].size();
        for (int i = 0; i < n_target_qubits; i++)
        {
            std::string qubit_name = std::to_string(RG.region_qubits[RG.rg.target(edge)](i));
            const char *cstr = qubit_name.c_str();
            coord_map[cstr] = xf::axis({"0", "1"});
            dim_list.push_back(cstr);
        }
        std::vector<int> shape(n_target_qubits, 2);
        xt::xarray<long double> md;
        md.resize(shape);
        md.fill(0.5);
       
       variable_type message_dummy(md,coord_map,dim_list);

        std::vector<variable_type> messages_iter(max_iterations+1);
        RG.message[edge]=messages_iter;
        for (int i = 0; i<max_iterations+1;i++)
        {
        RG.message[edge][i] = message_dummy;
        }

        // construct message_base dummy
        variable_type::coordinate_map coord_map_base;
        dimension_type::label_list dim_list_base = {};
        int n_source_qubits = RG.region_qubits[RG.rg.source(edge)].size();
        for (int i = 0; i < n_source_qubits; i++)
        {
            std::string qubit_name = std::to_string(RG.region_qubits[RG.rg.source(edge)](i));
            const char *cstr = qubit_name.c_str();
            coord_map_base[cstr] = xf::axis({"0", "1"});
            dim_list_base.push_back(cstr);
        }
        std::vector<int> shape_base(n_source_qubits, 2);
        
        md.resize(shape_base);
        md.fill(1.0);
       
       variable_type message_base_dummy(md,coord_map_base,dim_list_base);

       
        RG.message_base[edge] = message_base_dummy;

        // construct belief_dummy (for source: same variables as message_base, for target: same as message)
        RG.belief_base[RG.rg.source(edge)] = message_base_dummy;
        RG.belief_base[RG.rg.target(edge)] = message_dummy;

        
        // get dim_of_vars_to_marg
        auto dimension_map = message_base_dummy.dimension_labels();

        std::vector<size_t> dim_of_vars_to_marg;
        std::vector<size_t> dim_names_of_vars(message_base_dummy.dimension());
        size_t j = 0;

        for (auto it = dimension_map.begin(); it != dimension_map.end(); ++it)
        {
            xt::xarray<int> dim_name = std::stoi(it[0].c_str());

            if (xt::any(xt::isin(RG.vars_to_marginalize[edge], dim_name)))
            {
                dim_of_vars_to_marg.push_back(j);
            }
            j++;
        }

        RG.dim_of_vars_to_marg[edge] = dim_of_vars_to_marg; 
    }

  marginals.resize(max_iterations);
  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
      marginals[iteration].resize(n_q);
      for (int p = 0; p<n_q; p++)
      {
          marginals[iteration][p] = xt::ones<long double>({2});
      }
  }

    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        std::vector<variable_type> beliefs_iter(max_iterations+1);
        RG.belief[region]=beliefs_iter;
        RG.belief[region][0] = RG.belief_base[region];
    }
  
    lemon::mapFill(RG.rg,RG.edge_converged,false);

    free_energy = xt::zeros<long double>({max_iterations});

}

void GbpDecoder::fill_q_factors(xt::xarray<long double> p_initial)
{
    q_factors.resize(n_q);
    for (int q = 0; q < n_q; q++)
    {
        variable_type::coordinate_map coord_map;
        std::string qubit_name = std::to_string(q);
        dimension_type::label_list dim_list = {};
        const char *cstr = qubit_name.c_str();
        coord_map[cstr] = xf::axis({"0", "1"});
        dim_list.push_back(cstr);
        variable_type q_factor(p_initial, coord_map, dim_list);
        q_factors[q] = q_factor;
    }
}

void GbpDecoder::fill_c_factors()
{
     c_factors.resize(2);
     c_factors[0].resize(n_c);
     c_factors[1].resize(n_c);
    for (int c = 0; c < n_c; c++)
    {
        xt::xarray<int> qubits_in_support = xt::from_indices(xt::nonzero(xt::row(H,c)));
        int n_qubits_in_support = qubits_in_support.size();

        variable_type::coordinate_map coord_map;
        dimension_type::label_list dim_list = {};

        for (int q = 0; q < n_qubits_in_support; q++)
        {

        std::string q_name = std::to_string(qubits_in_support(q));
        const char *cstr = q_name.c_str();
        coord_map[cstr] = xf::axis({"0", "1"});
        dim_list.push_back(cstr);
        }

        std::vector<int> shape(n_qubits_in_support, 2);
        xt::xarray<long double> all_zeros;
        all_zeros.resize(shape);
        all_zeros.fill(0);
        xt::xarray<long double> all_ones;
        all_ones.resize(shape);
        all_ones.fill(0);


        std::vector<int> v1{0, 1};
        const int nv = 7; // maximum degree of checks currently supported


        for (auto &&t : iter::product<nv>(v1))
        {
            std::valarray<int> indices(n_qubits_in_support);
            indices = to_valarray(t);
            
            std::valarray<int> bar = indices[std::slice(0, n_qubits_in_support, 1)];
            std::vector<int> shape = {n_qubits_in_support};
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

        c_factors[0][c] = c_factor_0;
        c_factors[1][c] = c_factor_1;

    }
}

void GbpDecoder::prepare_messages(const xt::xarray<int> *s_0)
{
    for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
    {
        int n_source_qubits = RG.region_qubits[RG.rg.source(edge)].size();

        for (int q = 0; q < n_source_qubits; q++)
        {
            int qubit = RG.region_qubits[RG.rg.source(edge)](q);
            RG.message_base[edge] *= q_factors[qubit];
        }

        int n_source_checks = RG.region_checks[RG.rg.source(edge)].size();
        for (int c = 0; c < n_source_checks; c++)
        {
            int check = RG.region_checks[RG.rg.source(edge)](c);
            RG.message_base[edge] *= c_factors[s_0->at(check)][check];
        }
    }
}

void GbpDecoder::prepare_beliefs(const xt::xarray<int> *s_0)
{
    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        int n_qubits = RG.region_qubits[region].size();
        for (int q = 0; q < n_qubits; q++)
        {
            int qubit = RG.region_qubits[region](q);
            RG.belief_base[region] *= q_factors[qubit];
        }

        int n_checks = RG.region_checks[region].size();
        for (int c = 0; c < n_checks; c++)
        {
            int check = RG.region_checks[region](c);
            RG.belief_base[region] *= c_factors[s_0->at(check)][check];
            RG.belief[region][0] *= c_factors[s_0->at(check)][check];
        }
    }
}



xt::xarray<int> GbpDecoder::decode(xt::xarray<int> s_0,xt::xarray<long double> p_initial,int type_gbp,long double w_gbp, int return_strategy, bool return_if_success)
{
    lemon::mapFill(RG.rg,RG.edge_converged,false);
    lemon::mapFill(RG.rg,RG.region_converged,false);
    
    fill_q_factors(p_initial);
    fill_c_factors();
    prepare_messages(&s_0);
    prepare_beliefs(&s_0);
    int hd_i_to_return = 0;

    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        if (type_gbp == 0)
        {
            update_messages(&s_0,p_initial,w_gbp,iteration);//update messages by using YFMs formula
            update_beliefs(&s_0,iteration);
            get_marginals_and_hard_dec(iteration); // hard decision as average of single qubit marginals
        }
        
        else if (type_gbp == 1)
        {
            update_messages(&s_0,p_initial,w_gbp,iteration);//update messages by using YFMs formula
            update_beliefs(&s_0,iteration);
            get_marginals_and_hard_dec_2(iteration); // hard decision as argmax of superregion marginals
        }
        else if (type_gbp == 2)
        {

            update_messages_2(&s_0,p_initial,w_gbp,iteration); //update messages by marginalization of belief
            update_beliefs(&s_0,iteration);
            get_marginals_and_hard_dec_2(iteration); // hard decision as argmax of superregion marginals
        }

        xt::xarray<int> hd  = xt::row(hard_decisions,iteration);

        xt::xarray<int> s = gf2_syndrome(&hd, &H);
        xt::row(syndromes,iteration) = s^s_0;

        calculate_free_energy(iteration);


        if (s == s_0)
        {
            hd_i_to_return = iteration;
            took_iterations = iteration;
            if (return_if_success)
            {
                return xt::row(hard_decisions,iteration);
            }
        }

    }

    if (hd_i_to_return == 0)
    {
        if (return_strategy == 0) // last nonzero = converged error guess
        {
            for (int it = max_iterations-1; it > 1; it--)
            {
                if (xt::sum(xt::row(hard_decisions,it))() > 0)
                {
                    took_iterations = max_iterations;
                    hd_i_to_return = it;
                    break;
                }
            }
        }
        else if (return_strategy == 1) // minimum residual syndrome weight
        {
            int min_weight_res_s = n_c;
            for (int it = max_iterations-1; it > 1; it--)
            {
                xt::xarray<int> hd = xt::row(hard_decisions,it);
                if (xt::sum(hd)() > 0)
                {
                    xt::xarray<int> s = gf2_syndrome(&hd, &H);
                    xt::xarray<int> res_s = s^s_0;
                    if (xt::sum(res_s)() < min_weight_res_s)
                    {   
                        took_iterations = it;
                        hd_i_to_return = it;
                        min_weight_res_s = xt::sum(res_s)();
                    }
                }
            }
        }
        else if (return_strategy == 2) // minimum free energy 
        {
            long double min_free_F = xt::amax(free_energy)();
            for (int it = max_iterations-1; it > 1; it--)
            {
                xt::xarray<int> hd = xt::row(hard_decisions,it);
                if (xt::sum(hd)() != 0)
                {
                    xt::xarray<int> s = gf2_syndrome(&hd, &H);
                    xt::xarray<int> res_s = s^s_0;
                    if (free_energy(it) < min_free_F)
                    {   
                        took_iterations = it;
                        hd_i_to_return = it;
                        min_free_F = free_energy(it);
                    }
                }
            }
        }
    }
    

    if (hd_i_to_return == 0)
    {
        took_iterations = max_iterations;
        hd_i_to_return = max_iterations-1;
    }
    
    return xt::row(hard_decisions,hd_i_to_return);

}


void GbpDecoder::update_messages(const xt::xarray<int> *s_0, xt::xarray<long double> p_initial,long double w_gbp, int iteration)
{
    for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
    {
        variable_type update_message_bm = RG.message_base[edge];

        // N
        for (int n = 0; n < RG.N[edge].size(); n++)
        {
            update_message_bm *= RG.message[RG.rg.arcFromId(RG.N[edge](n))][iteration-1];
        }

        // marginalization
        variable_type update_message = RG.message[edge][iteration];
        update_message.data() = xt::sum(update_message_bm.data(), RG.dim_of_vars_to_marg[edge]);

        // D
        // for (int d = 0; d < RG.D[edge].size(); d++)
        // {
        //     auto tmp = update_message / RG.message[RG.rg.arcFromId(RG.D[edge](d))][iteration];
        //     update_message = tmp;
        // }

        //normalize update_message
        update_message.data() /= xt::sum(update_message.data());


        RG.message[edge][iteration] = (1-w_gbp) * RG.message[edge][iteration-1] + w_gbp * update_message;

        if (xt::allclose(RG.message[edge][iteration].data().value(),RG.message[edge][iteration-1].data().value()),1e-100,1e-100)
        {
            RG.edge_converged[edge] = true;
            // std::cout << "edge " << RG.rg.id(edge) << " converged\n";
        }
    }

}

void GbpDecoder::update_messages_2(const xt::xarray<int> *s_0, xt::xarray<long double> p_initial,long double w_gbp, int iteration)
{
    for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
    {
        // if (RG.edge_converged[edge] == false)
        // {
            variable_type belief_source = RG.belief[RG.rg.source(edge)][iteration-1];
            variable_type belief_target = RG.belief[RG.rg.target(edge)][iteration-1];


            variable_type update_message = RG.message[edge][iteration-1];
            update_message.data() *= xt::sum(belief_source.data(), RG.dim_of_vars_to_marg[edge]);

            update_message /= belief_target;

            // normalize update_message
            update_message.data() /= xt::sum(update_message.data());


            RG.message[edge][iteration] = (1-w_gbp) * RG.message[edge][iteration-1] + w_gbp * update_message;

            if (xt::allclose(RG.message[edge][iteration].data().value(),RG.message[edge][iteration-1].data().value()),1e-100,1e-100)
            {
                RG.edge_converged[edge] = true;
            }
    }

}


void GbpDecoder::update_beliefs(const xt::xarray<int> *s_0,int iteration)
{
    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        // if (RG.region_converged[region] == false)
            xt::xarray<int> i_region = {RG.rg.id(region)};
            xt::xarray<int> E_R = union1d(i_region,RG.descendants[region]);

            // local factors are already calculated in belief_base
            variable_type bf = RG.belief_base[region];

            // messages from parents
            for (lemon::ListDigraph::InArcIt edge_from_parent(RG.rg,region); edge_from_parent != lemon::INVALID; ++edge_from_parent)
            {
                bf *= RG.message[edge_from_parent][iteration];
            }

            // messages into descendants from other parents
            for (auto d = RG.descendants[region].begin(); d != RG.descendants[region].end(); ++d)
            {
                for (lemon::ListDigraph::InArcIt edge_from_parent_of_descendant(RG.rg,RG.rg.nodeFromId(*d)); edge_from_parent_of_descendant != lemon::INVALID; ++edge_from_parent_of_descendant)
                {
                    auto condition = !xt::isin(RG.rg.id(RG.rg.source(edge_from_parent_of_descendant)),E_R);
                    if (condition())
                    {
                        auto tmp = bf * RG.message[edge_from_parent_of_descendant][iteration];
                        bf = tmp;
                    }
                }
            }
            // bf.data() /= xt::sum(bf.data());

            RG.belief[region][iteration] = bf;

            if (xt::allclose(bf.data().value(),bf.data().value()),1e-100,1e-100)
            {
                RG.region_converged[region] = true;
            }

    }
}


void GbpDecoder::get_marginals_and_hard_dec(int iteration)
{
    for (int q = 0; q < n_q; q++)
        marginals[iteration][q] = q_factors[q].data().value();
    
    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        xt::xarray<int> qubits_in_region = RG.region_qubits[region];
        for (int q = 0; q < qubits_in_region.size(); q++)
        {
            xt::xarray<int> qubit_to_marg = qubits_in_region(q);
            xt::xarray<int> diff = xt::setdiff1d(qubits_in_region, qubit_to_marg);
            if (diff.size() > 1)
            {
                std::vector<int> other_qubits(diff.size());
                std::copy(diff.crbegin(), diff.crend(), other_qubits.begin());

                auto dimension_map = RG.belief[region][iteration].dimension_labels();

                std::vector<size_t> dim_of_qubits_to_marg;
                std::vector<size_t> dim_names_of_qubits(RG.belief[region][iteration].dimension());
                size_t j = 0;

                for (auto it = dimension_map.begin(); it != dimension_map.end(); ++it)
                {
                    xt::xarray<int> dim_name = std::stoi(it[0].c_str());
                    if (xt::any(xt::isin(diff, dim_name)))
                    {
                        dim_of_qubits_to_marg.push_back(j);
                    }
                    j++;
                }

                xt::xarray<long double> bf = RG.belief[region][iteration].data().value();

                xt::xarray<long double> marginal = xt::sum(bf, dim_of_qubits_to_marg);

                marginals[iteration].at(qubit_to_marg(0)) += marginal;
            }
        }
    
    }

    for (size_t q = 0; q < n_q; q++)
    {
        if (xt::sum(marginals[iteration][q])() != 0)
        {
            marginals[iteration][q] /= xt::sum(marginals[iteration][q]);
        }

        if (marginals[iteration][q](1) > marginals[iteration][q](0)) 
        {
            hard_decisions(iteration,q) = 1;
        }
        else
        {
            hard_decisions(iteration,q) = 0;
        }
    }
}

void GbpDecoder::get_marginals_and_hard_dec_2(int iteration)
{
    for (int q = 0; q < n_q; q++)
    {
        marginals[iteration][q] = {0,0}; 
    }
        

    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        xt::xarray<int> qubits_in_region = RG.region_qubits[region];

        xt::xarray<long double> bf = RG.belief[region][iteration].data().value();

        if (RG.counting_number[region] == 1)
        {
            xt::xarray<int> argmax_belief = xt::argmax(bf);
            auto argwhere_belief = xt::from_indices((xt::argwhere(xt::isclose(bf,xt::amax(bf)))));
            auto array_index = xt::flatten(xt::from_indices(xt::unravel_indices(argmax_belief, bf.shape())));

            for (int q = 0; q < qubits_in_region.size(); q++)
            {
                if (array_index(q) == 1)
                {
                    hard_decisions(iteration,qubits_in_region(q)) = 1;
                }
            }
        }

    }

}

void GbpDecoder::calculate_free_energy(int iteration)
{
    long double average_energy = 0;
    long double entropy = 0;
    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        // region average energy
        long double region_average_energy = 0;
        long double region_entropy = 0;
        variable_type bf = RG.belief[region][iteration];
        variable_type tmp = bf;
        tmp.data().fill(0.0);
        // log(q_factors[RG.region_qubits[region](0)]);
        for (size_t i = 0; i < RG.region_qubits[region].size(); i++)
        {
            tmp += log(q_factors[RG.region_qubits[region](i)]);
        }

        tmp = -1*tmp*bf;
        tmp.data().value() = xt::nan_to_num(tmp.data().value());
        region_average_energy = xt::sum(tmp.data().value())();

        tmp = bf * log(bf);
        tmp.data().value() = xt::nan_to_num(tmp.data().value());
        // std::cout << "tmp = " << tmp << "\n";


        region_entropy = -1.0 * xt::sum(tmp.data().value())();

        average_energy += (long double)RG.counting_number[region] * region_average_energy;
        entropy += (long double)RG.counting_number[region] * region_entropy;
    }
    free_energy(iteration) = average_energy - entropy;
}

xt::xarray<long double> GbpDecoder::get_messages()
{
    int max_dim = 0;
    for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
    {
        if (RG.region_qubits[RG.rg.target(edge)].size() > max_dim)
        max_dim = RG.region_qubits[RG.rg.target(edge)].size() ;
    }
    max_dim = pow(2,max_dim);

    xt::xarray<long double> messages =  xt::zeros<long double>({max_iterations,RG.n_edges,max_dim});;

    for (int iteration = 0; iteration<max_iterations; iteration ++)
    {
        for (int edge = 0; edge < RG.n_edges; edge++)
        {
            for (int i = 0; i <  max_dim ; i++)
            {
                messages(iteration,edge,i) = RG.message[RG.rg.arcFromId(edge)][iteration].data().value().flat(i);
            }
        }
    }

    return messages;
}


xt::xarray<long double> GbpDecoder::get_marginals()
{
    xt::xarray<long double> marginals_ret = xt::zeros<long double>({max_iterations,n_q,2});

    for (int iteration = 0; iteration<max_iterations; iteration ++)
    {
        for (int qubit = 0; qubit < n_q; qubit++)
        {
            marginals_ret(iteration,qubit,0) = marginals[iteration][qubit](0);
            marginals_ret(iteration,qubit,1) = marginals[iteration][qubit](1);
        }
  }
  
  return marginals_ret;
}


xt::xarray<int> GbpDecoder::get_hard_decisions()
{
    return hard_decisions;
}

xt::xarray<int> GbpDecoder::get_syndromes()
{
    return syndromes;
}

xt::xarray<long double> GbpDecoder::get_free_energy()
{  
  return free_energy;
}

xt::xarray<bool> GbpDecoder::get_convergence()
{
    xt::xarray<bool> convergence_ret = xt::zeros<bool>({RG.n_edges});
    for (int edge = 0; edge < RG.n_edges; edge++)
    {
        convergence_ret(edge) = RG.edge_converged[RG.rg.arcFromId(edge)];
    }
    return convergence_ret;
}