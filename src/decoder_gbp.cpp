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
GbpDecoder::GbpDecoder(xt::xarray<int> H, int max_iterations,int n_checks_per_r0) : H(H), max_iterations(max_iterations), RG(H, n_checks_per_r0)
{
  n_c = H.shape(0);
  n_q = H.shape(1);
  rank_H = gf4_rank(H.data(),n_c,n_q);
  
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


    //   std::vector< xt::xarray<long double> > va(max_iterations);
    //   RG.message[edge] = va;
    //   for (int iteration = 0; iteration < max_iterations; iteration++)
    //   {
    //       RG.message[edge][iteration] = xt::ones<long double>({2});
    //   }

 
  }

  marginals.resize(max_iterations);
  hard_dec.resize(max_iterations);
  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
      marginals[iteration].resize(n_q);
      for (int p = 0; p<n_q; p++)
      {
          marginals[iteration][p] = xt::ones<long double>({2});
      }
      hard_dec[iteration] = xt::zeros<int>({n_q});
  }

    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        std::vector<variable_type> beliefs_iter(max_iterations+1);
        RG.belief[region]=beliefs_iter;
    }
  
    lemon::mapFill(RG.rg,RG.edge_converged,false);


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
        // std::cout << "qubits_in_support = " << qubits_in_support << "\n";
        int n_qubits_in_support = qubits_in_support.size();
        // std::cout << "n_qubits_in_support = " << n_qubits_in_support << "\n";

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
        const int nv = 7;

        // std::vector< std::vector<int> > v2 = product({v1,v1});

        // for (int i = 0; i < 4; i++)
        // {
        //     v2 = product({v2,v2});
        // }
        
        // std::cout << "v2 = " << "\n";
        // print(v2);

        for (auto &&t : iter::product<nv>(v1))
        {
            std::valarray<int> indices(n_qubits_in_support);
            indices = to_valarray(t);
            std::cout << "indices = " << "\n";
            for (auto i: indices) std::cout << i << ' ';
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

        std::cout << "c = " << c << "\nc_factor_0 = \n" << c_factor_0 << "\n";

        c_factors[0][c] = c_factor_0;
        c_factors[1][c] = c_factor_1;

        // std::cout << "check " << c << "  c_factors[0][c] = \n" << c_factors[0][c] << "\n";
    }
}

void GbpDecoder::prepare_messages(const xt::xarray<int> *s_0)
{
    for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
    {
        int n_source_qubits = RG.region_qubits[RG.rg.source(edge)].size();
        // std::cout << "edge " << RG.rg.id(edge) << " n_source_qubits = " << n_source_qubits << "\n";
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
        // std::cout << "edge " << RG.rg.id(edge) << " message_base = \n" << RG.message_base[edge] << "\n";
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
        }
    }
}



xt::xarray<int> GbpDecoder::decode(xt::xarray<int> s_0,xt::xarray<long double> p_initial,int type_marg,long double w_gbp)
{
    // std::cout << "0\n";
    fill_q_factors(p_initial);
    // std::cout << "1\n";
    fill_c_factors();
    // std::cout << "2\n";
    prepare_messages(&s_0);
    // std::cout << "3\n";
    prepare_beliefs(&s_0);
    // std::cout << "4\n";
    int hd_i_to_return = 0;

    for (int iteration = 1; iteration < max_iterations; iteration++)
    {
        update_messages(&s_0,p_initial,w_gbp,iteration);

        update_beliefs(&s_0,iteration);

        if (type_marg == 0)
        {
            get_marginals_and_hard_dec(iteration);
        }
        else if (type_marg == 1)
        {
            overall_belief(iteration);
        }
            

        // std::cout << "2\n";

        xt::xarray<int> s = xt::zeros<int>({n_c});
        xt::xarray<int> hd  = hard_dec[iteration];
        gf2_syndrome(&s, &hd, &H);

        // std::cout << "it " << iteration << "\n s = " << s << "\nhd = " << hd << "\n";


        if (s == s_0)
        {
            hd_i_to_return = iteration;
            return hard_dec[iteration];
        }

    }

    if (hd_i_to_return == 0)
    {
        for (int it = hard_dec.size()-1; it > -1; it--)
        {
            if (xt::sum(hard_dec[it])() != 0)
            {
                hd_i_to_return = it;
            }
        }
    }

    if (hd_i_to_return == 0)
    {
        hd_i_to_return = max_iterations-1;
    }
    
    return hard_dec[hd_i_to_return];

}


void GbpDecoder::update_messages(const xt::xarray<int> *s_0, xt::xarray<long double> p_initial,long double w_gbp, int iteration)
{
    for (lemon::ListDigraph::ArcIt edge(RG.rg); edge != lemon::INVALID; ++edge)
    {
        variable_type update_message_bm = RG.message_base[edge];
        // std::cout << "edge " << RG.rg.id(edge) << " update_message_bm b/f = \n" << update_message_bm << "\n";
        // N
        for (int n = 0; n < RG.N[edge].size(); n++)
        {
            update_message_bm *= RG.message[RG.rg.arcFromId(RG.N[edge](n))][iteration-1];
        }

        // std::cout << "edge " << RG.rg.id(edge) << " update_message_bm a/f = \n" << update_message_bm << "\n";

        // marginalization
        variable_type update_message = RG.message[edge][iteration];
        //  std::cout << "edge " << RG.rg.id(edge) << " update_message b/f = \n" << update_message << "\n";
        update_message.data() = xt::sum(update_message_bm.data(), RG.dim_of_vars_to_marg[edge]);

        // D
        for (int d = 0; d < RG.D[edge].size(); d++)
        {
            update_message /= RG.message[RG.rg.arcFromId(RG.D[edge](d))][iteration-1];
        }

        //normalize update_message
        update_message.data() /= xt::sum(update_message.data());

        // std::cout << "edge " << RG.rg.id(edge) << " update_message a/f = \n" << update_message << "\n";


        RG.message[edge][iteration] = (1-w_gbp) * RG.message[edge][iteration-1] + w_gbp * update_message;
        if (RG.message[edge][iteration] == RG.message[edge][iteration-1])
        {
            RG.edge_converged[edge] = true;
        }
    }

}

void GbpDecoder::update_beliefs(const xt::xarray<int> *s_0,int iteration)
{
    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        xt::xarray<int> i_region = {RG.rg.id(region)};
        xt::xarray<int> E_R = union1d(i_region,RG.descendants[region]);

        // loval factors are already calculated in belief_base
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
                    bf *= RG.message[edge_from_parent_of_descendant][iteration];
                }
            }
        }
        bf.data() /= xt::sum(bf.data());

        RG.belief[region][iteration] = bf;
    }
}

void GbpDecoder::overall_belief(int iteration)
{
    // construct message dummy
    variable_type::coordinate_map coord_map;
    dimension_type::label_list dim_list = {};
    
    for (int q = 0; q < n_q; q++)
    {
        std::string qubit_name = std::to_string(q);
        const char *cstr = qubit_name.c_str();
        coord_map[cstr] = xf::axis({"0", "1"});
        dim_list.push_back(cstr);
    }
    std::vector<int> shape(n_q, 2);
    xt::xarray<long double> obf;
    obf.resize(shape);
    obf.fill(1.0);
    
    variable_type overall_belief(obf,coord_map,dim_list);

    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
        // std::cout << "RG.belief[region = " << RG.rg.id(region) << "][iteration] = \n" << RG.belief[region][iteration] << "\n xf::pow(RG.belief[region][iteration],RG.counting_number[region]) = \n" << xf::pow(RG.belief[region][iteration],RG.counting_number[region]) << "\n";
        overall_belief *= xf::pow(RG.belief[region][iteration],RG.counting_number[region]);
        
    }

    // overall_belief.data() /= xt::sum(overall_belief.data());

    for (int q = 0; q < n_q; q++)
    {
        std::vector<size_t> other_qubits;

        for (int oq = 0; oq < n_q; oq++)
        {
            if (oq != q)
            {
                other_qubits.push_back(oq);
            }
        }
        xt::xarray<long double> marginal = xt::sum(overall_belief.data().value(), other_qubits);
        // marginal /= xt::sum(marginal);
        marginals[iteration].at(q) = marginal / xt::sum(marginal);

        if (marginal.at(1) > marginal.at(0)) 
        {
            hard_dec[iteration][q] = 1;
        }
        else
        {
            hard_dec[iteration][q] = 0;
        }
    }

    // xt::xarray<int> argmax_overall_belief = xt::argmax(overall_belief.data().value());
    // auto array_index = xt::from_indices(xt::unravel_indices(argmax_overall_belief, overall_belief.shape()));
    // // std::cout << "argmax(overall_belief) = \n" << array_index << "\n"; 
    // hard_dec[iteration] = xt::flatten(array_index);
}

void GbpDecoder::get_marginals_and_hard_dec(int iteration)
{
    for (int q = 0; q < n_q; q++)
        marginals[iteration][q] = q_factors[q].data().value();
    
    for (lemon::ListDigraph::NodeIt region(RG.rg); region != lemon::INVALID; ++region)
    {
    // std::cout << "  - region " << RG.rg.id(region) << std::endl; 
    // std::cout << "  - belief: \n\t" << RG.belief[region][iteration] << std::endl;
    // std::cout << "  - argmax(belief): " << xt::argmax(RG.belief[region][iteration].data().value()) << std::endl << std::endl;
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

                // xt::xarray<long double> bf = xt::pow(RG.belief[region][iteration].data().value(),RG.counting_number[region]);

                xt::xarray<long double> bf = RG.belief[region][iteration].data().value();

                xt::xarray<long double> marginal = xt::sum(bf, dim_of_qubits_to_marg);

                marginals[iteration].at(qubit_to_marg(0)) += marginal;
                // xt::pow(marginal,RG.counting_number[region]);///xt::sum(marginal);
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
            hard_dec[iteration][q] = 1;
        }
        else
        {
            hard_dec[iteration][q] = 0;
        }
    }
}

xt::xarray<long double> GbpDecoder::get_messages()
{
    // xt::xarray<long double> messages = xt::zeros<long double>({max_iterations,RG.n_edges,2});
    xt::xarray<long double> messages =  xt::zeros<long double>({max_iterations,RG.n_edges,2});;
    // messages.resize({max_iterations,RG.n_edges,2});
    
    for (int iteration = 0; iteration<max_iterations; iteration ++)
    {
        for (int edge = 0; edge < RG.n_edges; edge++)
        {
            messages(iteration,edge,0) = RG.message[RG.rg.arcFromId(edge)][iteration].data().value()(0);
            messages(iteration,edge,1) = RG.message[RG.rg.arcFromId(edge)][iteration].data().value()(1);
        }
    }

    return messages;
}


xt::xarray<long double> GbpDecoder::get_marginals()
{
    xt::xarray<long double> marginals_ret = xt::zeros<long double>({max_iterations,n_q,2});
    // marginals_ret.resize({max_iterations,n_q,2});

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
    xt::xarray<int> hd = xt::zeros<long double>({max_iterations,n_q});
    return hd;
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