#include <bp_gbp/constructor_gbp.hpp>

RegionGraph::RegionGraph(xt::xarray<int> H, int n_checks_per_r0, int rg_type) : H(H), region_qubits(rg), region_checks(rg), region_level(rg), belief(rg), belief_base(rg), vars_to_marginalize(rg),dim_of_vars_to_marg(rg), message(rg), edge_converged(rg), region_converged(rg), message_base(rg), N(rg), D(rg),counting_number(rg), ancestors(rg), descendants(rg)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    checks = xt::arange<int>(n_c);
    qubits = xt::arange<int>(n_q);
    construct_rg(n_checks_per_r0,rg_type);
}

RegionGraph::RegionGraph(xt::xarray<int> H, int n_checks_per_r0, xt::xarray<int> check_list, int rg_type) : H(H), region_qubits(rg), region_checks(rg), region_level(rg), belief(rg), belief_base(rg), vars_to_marginalize(rg),dim_of_vars_to_marg(rg), message(rg), edge_converged(rg), region_converged(rg), message_base(rg), N(rg), D(rg),counting_number(rg), ancestors(rg), descendants(rg)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    checks = xt::arange<int>(n_c);
    qubits = xt::arange<int>(n_q);
    construct_rg(n_checks_per_r0,check_list,rg_type);
}

void RegionGraph::construct_rg(int n_checks_per_r0, int rg_type)
{
    xt::random::seed(time(NULL));
    // R_0 (basic)
    // for (int c = 0; c < checks.size()-1; c++)
    // {
    //     lemon::ListDigraph::Node new_region = rg.addNode();
        
    //     region_checks[new_region] = xt::xarray<int>{c};
    //     xt::xarray<int>row = xt::row(H,c);
    //     xt::xarray<int> qubits = xt::flatten(xt::from_indices(xt::argwhere(row > 0)));
    //     region_qubits[new_region] = qubits;
    //     region_level[new_region] = 0;
    // }

    // R_0 (3 checks per region)
    // xt::xarray<int> check_list = {0,4,6,1,7,10,2,8,9,3,5,11};
    // xt::xarray<int> check_list = {3,5,6,0,1,2,7,9,10,4,8,11};
    // xt::xarray<int> check_list = {0,4,7,1,8,10,2,5,6,3,9,11};
    // // xt::xarray<int> check_list = {0,1,2,3,4,5,6,7,8,9,10,11};
    // // xt::random::shuffle(check_list);

    // for (int c = 0; c < checks.size()-2; c+=3)
    // {
    //     int c1 = check_list(c);
    //     int c2 = check_list(c+1);
    //     int c3 = check_list(c+2);
    //     lemon::ListDigraph::Node new_region = rg.addNode();
        
    //     region_checks[new_region] = xt::xarray<int>{c1,c2,c3};
    //     std::cout << "R_0: region_checks["<< rg.id(new_region) << "] = " << region_checks[new_region] << "\n";
    //     xt::xarray<int>row_1 = xt::row(H,c1);
    //     xt::xarray<int>row_2 = xt::row(H,c2);
    //     xt::xarray<int>row_3 = xt::row(H,c3);
    //     xt::xarray<int> qubits_1 = xt::flatten(xt::from_indices(xt::argwhere(row_1 > 0)));
    //     xt::xarray<int> qubits_2 = xt::flatten(xt::from_indices(xt::argwhere(row_2 > 0)));
    //     xt::xarray<int> qubits_3 = xt::flatten(xt::from_indices(xt::argwhere(row_3 > 0)));

    //     region_qubits[new_region] = union1d(union1d(qubits_1,qubits_2),qubits_3);
    //     region_level[new_region] = 0;
    // }

    // R_0 (4 checks per region)
    // xt::xarray<int> check_list = {0,4,6,1,7,10,2,8,9,3,5,11};
    // xt::xarray<int> check_list = {3,5,6,0,1,2,7,9,10,4,8,11};

    // xt::xarray<int> check_list = {0,5,1,6,2,7,3,8,4,9,10,15,11,16,12,17,13,18,14,19};
    // n_checks_per_r0 = 4;
    xt::xarray<int> check_list = xt::arange<int>(n_c);
    // // xt::xarray<int> check_list = xt::zeros<int> ({n_c});
    
    
    if (rg_type == 0)
        xt::random::shuffle(check_list);
    else if (rg_type == 2)
    {
        int dist = (int)((1 + sqrt(1+4*n_c))/2);
        // std::cout << "dist = " << dist << "\n";
        int i = 0;
        int j = 0;

        for (int c = 0; c < n_c; c++)
        {
            check_list(c) = j+i*dist;
            i++;
            if (i % (dist-1) == 0)
            {
                j++;
                i = 0;
            }
        }
    }
    // // std::cout << "check_list = " << check_list << "\n";

    // xt::xarray<int> check_list = {2,11,5,10,0,3,8,1,4,6,7,9};

    std::vector<int> used_checks;

    int n_r0 =  ceil((double)n_c / (double) n_checks_per_r0);
    // std::cout << "n_r0 = " << n_r0 << "\n";
    int c_p = 0;
    for (int i_r0 = 0; i_r0 < n_r0; i_r0++)
    {
        lemon::ListDigraph::Node new_region = rg.addNode();
        std::vector<int> checks_in_r0;
        std::set<int> qubits_in_r0;
        for (int c = c_p; c < c_p + n_checks_per_r0; c++)
        {
            int c0 = check_list(c);
            checks_in_r0.push_back(c0);
            xt::xarray<int>row = xt::row(H,c0);
            xt::xarray<int> qubits = xt::flatten(xt::from_indices(xt::argwhere(row > 0)));
            for (auto q = qubits.begin(); q != qubits.end(); ++q)
            {
                qubits_in_r0.insert(*q);
            }
            if (c == n_c-1) break;
        }
        c_p+=n_checks_per_r0;
        region_checks[new_region] = xt::adapt(checks_in_r0);
        std::vector<double> qir(qubits_in_r0.begin(), qubits_in_r0.end());
        region_qubits[new_region] = xt::adapt(qir);
        // std::cout << "region_checks[new_region] = " << region_checks[new_region] << "\n";
        // std::cout << "region_qubits[new_region] = " << region_qubits[new_region] << "\n";
        region_level[new_region] = 0;
    }



    // for (int c = 0; c < checks.size()-3; c+=4)
    // {
    //     int c0 = check_list(c);
    //     int c1 = check_list(c+1);
    //     int c2 = check_list(c+2);
    //     int c3 = check_list(c+3);
    //     lemon::ListDigraph::Node new_region = rg.addNode();
        
    //     region_checks[new_region] = xt::xarray<int>{c0,c1,c2,c3};
    //     // std::cout << "R_0: region_checks["<< rg.id(new_region) << "] = " << region_checks[new_region] << "\n";
    //     xt::xarray<int>row_0 = xt::row(H,c0);
    //     xt::xarray<int>row_1 = xt::row(H,c1);
    //     xt::xarray<int>row_2 = xt::row(H,c2);
    //     xt::xarray<int>row_3 = xt::row(H,c3);
    //     xt::xarray<int> qubits_0 = xt::flatten(xt::from_indices(xt::argwhere(row_0 > 0)));
    //     xt::xarray<int> qubits_1 = xt::flatten(xt::from_indices(xt::argwhere(row_1 > 0)));
    //     xt::xarray<int> qubits_2 = xt::flatten(xt::from_indices(xt::argwhere(row_2 > 0)));
    //     xt::xarray<int> qubits_3 = xt::flatten(xt::from_indices(xt::argwhere(row_3 > 0)));

    //     region_qubits[new_region] = union1d(union1d(union1d(qubits_0,qubits_1),qubits_2),qubits_3);
    //     region_level[new_region] = 0;
    // }



    // other regions
    bool done = false;
    int current_level = 1;
    xt::xarray<int> single_qubits_used = xt::arange<int>(n_q);
    
    while (done == false)
    {
        // std::cout << "current level = " << current_level << std::endl;
        // std::set< xt::xarray<int> ,xarrayCmp>
        std::unordered_set< xt::xarray<int>, xarrayHash > intersections = find_intersections(current_level-1);
        // std::cout << "intersections = ";
        // for (auto it = intersections.begin(); it != intersections.end(); ++it)
        // {
        //     std::cout << *it << " ";
        // }
        // std::cout << std::endl;
        if (intersections.size() > 0)
        {
            for (auto it = intersections.begin(); it != intersections.end(); ++it)
            {
                // std::cout << "*it = " << *it << std::endl;
                lemon::ListDigraph::Node new_region = rg.addNode();
                region_checks[new_region] = xt::empty<int>({0});
                region_qubits[new_region] = *it;
                region_level[new_region] = current_level;
                if ((*it).size() == 1)
                {
                    single_qubits_used(it->at(0)) = -1;
                }
            }
            current_level++;
        }
        else
        {
            done = true;
        }
    }

    max_level = current_level;

    // not used variables (?)
    // for (auto it = single_qubits_used.begin(); it != single_qubits_used.end(); ++it)
    // {
    //     if (*it != -1)
    //     {
    //         lemon::ListDigraph::Node new_region = rg.addNode();
    //         region_checks[new_region] = -1;
    //         xt::xarray<int> q = xt::ones<int>({1})*(*it);
    //         region_qubits[new_region] = q;
    //         region_level[new_region] = current_level;
    //     }
    // }

    // edges
    make_edges();

    n_regions=0;
    for (lemon::ListDigraph::NodeIt r(rg); r!=lemon::INVALID; ++r) ++n_regions;

    n_edges=0;
    for (lemon::ListDigraph::ArcIt e(rg); e!=lemon::INVALID; ++e) ++n_edges;

    get_descendants_N_D();

    get_ancestors_counting_numbers();

    std::string suffix = std::to_string(n_checks_per_r0);
    print_regiongraph(suffix);

}

void RegionGraph::construct_rg(int n_checks_per_r0, xt::xarray<int> check_list, int rg_type)
{
    xt::random::seed(time(NULL));
    // R_0 (basic)
    // std::cout << "check_list = " << check_list << "\n";

    std::vector<int> used_checks;

    int n_r0 =  ceil((double)n_c / (double) n_checks_per_r0);
    // std::cout << "n_r0 = " << n_r0 << "\n";
    int c_p = 0;
    for (int i_r0 = 0; i_r0 < n_r0; i_r0++)
    {
        lemon::ListDigraph::Node new_region = rg.addNode();
        std::vector<int> checks_in_r0;
        std::set<int> qubits_in_r0;
        for (int c = c_p; c < c_p + n_checks_per_r0; c++)
        {
            int c0 = check_list(c);
            checks_in_r0.push_back(c0);
            xt::xarray<int>row = xt::row(H,c0);
            xt::xarray<int> qubits = xt::flatten(xt::from_indices(xt::argwhere(row > 0)));
            for (auto q = qubits.begin(); q != qubits.end(); ++q)
            {
                qubits_in_r0.insert(*q);
            }
            if (c == n_c-1) break;
        }
        c_p+=n_checks_per_r0;
        region_checks[new_region] = xt::adapt(checks_in_r0);
        std::vector<double> qir(qubits_in_r0.begin(), qubits_in_r0.end());
        region_qubits[new_region] = xt::adapt(qir);
        // std::cout << "region_checks[new_region] = " << region_checks[new_region] << "\n";
        // std::cout << "region_qubits[new_region] = " << region_qubits[new_region] << "\n";
        region_level[new_region] = 0;
    }



    // for (int c = 0; c < checks.size()-3; c+=4)
    // {
    //     int c0 = check_list(c);
    //     int c1 = check_list(c+1);
    //     int c2 = check_list(c+2);
    //     int c3 = check_list(c+3);
    //     lemon::ListDigraph::Node new_region = rg.addNode();
        
    //     region_checks[new_region] = xt::xarray<int>{c0,c1,c2,c3};
    //     // std::cout << "R_0: region_checks["<< rg.id(new_region) << "] = " << region_checks[new_region] << "\n";
    //     xt::xarray<int>row_0 = xt::row(H,c0);
    //     xt::xarray<int>row_1 = xt::row(H,c1);
    //     xt::xarray<int>row_2 = xt::row(H,c2);
    //     xt::xarray<int>row_3 = xt::row(H,c3);
    //     xt::xarray<int> qubits_0 = xt::flatten(xt::from_indices(xt::argwhere(row_0 > 0)));
    //     xt::xarray<int> qubits_1 = xt::flatten(xt::from_indices(xt::argwhere(row_1 > 0)));
    //     xt::xarray<int> qubits_2 = xt::flatten(xt::from_indices(xt::argwhere(row_2 > 0)));
    //     xt::xarray<int> qubits_3 = xt::flatten(xt::from_indices(xt::argwhere(row_3 > 0)));

    //     region_qubits[new_region] = union1d(union1d(union1d(qubits_0,qubits_1),qubits_2),qubits_3);
    //     region_level[new_region] = 0;
    // }



    // other regions
    bool done = false;
    int current_level = 1;
    xt::xarray<int> single_qubits_used = xt::arange<int>(n_q);
    
    while (done == false)
    {
        // std::cout << "current level = " << current_level << std::endl;
        // std::set< xt::xarray<int> ,xarrayCmp>
        std::unordered_set< xt::xarray<int>, xarrayHash > intersections = find_intersections(current_level-1);
        // std::cout << "intersections = ";
        // for (auto it = intersections.begin(); it != intersections.end(); ++it)
        // {
        //     std::cout << *it << " ";
        // }
        // std::cout << std::endl;
        if (intersections.size() > 0)
        {
            for (auto it = intersections.begin(); it != intersections.end(); ++it)
            {
                // std::cout << "*it = " << *it << std::endl;
                lemon::ListDigraph::Node new_region = rg.addNode();
                region_checks[new_region] = xt::empty<int>({0});
                region_qubits[new_region] = *it;
                region_level[new_region] = current_level;
                if ((*it).size() == 1)
                {
                    single_qubits_used(it->at(0)) = -1;
                }
            }
            current_level++;
        }
        else
        {
            done = true;
        }
    }

    max_level = current_level;

    // not used variables (?)
    // for (auto it = single_qubits_used.begin(); it != single_qubits_used.end(); ++it)
    // {
    //     if (*it != -1)
    //     {
    //         lemon::ListDigraph::Node new_region = rg.addNode();
    //         region_checks[new_region] = -1;
    //         xt::xarray<int> q = xt::ones<int>({1})*(*it);
    //         region_qubits[new_region] = q;
    //         region_level[new_region] = current_level;
    //     }
    // }

    // edges
    make_edges();

    n_regions=0;
    for (lemon::ListDigraph::NodeIt r(rg); r!=lemon::INVALID; ++r) ++n_regions;

    n_edges=0;
    for (lemon::ListDigraph::ArcIt e(rg); e!=lemon::INVALID; ++e) ++n_edges;

    get_descendants_N_D();

    get_ancestors_counting_numbers();

    std::string suffix = std::to_string(n_checks_per_r0);
    print_regiongraph(suffix);

}

// std::set< xt::xarray<int> ,xarrayCmp>
std::unordered_set< xt::xarray<int>, xarrayHash > RegionGraph::find_intersections(int level)
{
//    std::set< xt::xarray<int> ,xarrayCmp> intersections;
   std::unordered_set< xt::xarray<int>, xarrayHash > intersections;

    for (lemon::ListDigraph::NodeIt region1(rg); region1 != lemon::INVALID; ++region1)
    {
        for (lemon::ListDigraph::NodeIt region2(rg); region2 != lemon::INVALID; ++region2)
        {
            // std::cout << region_level[region1] << region_level[region2] 
            if ((region1 != region2) && (region_level[region1] == region_level[region2]) && (region_level[region2] == level))
            {
                xt::xarray<int> intersection = intersect1d(region_qubits[region1],region_qubits[region2]);
                // std::cout << "r1 = " << region_checks[region1] << " r2 = " << region_checks[region2] << " intersection = " << intersection << std::endl;
                if (intersection.size() > 0)
                {
                    intersections.insert(intersection);
                }
                
            }
        }
    }
    return intersections;
}


void RegionGraph::make_edges()
{
    for (lemon::ListDigraph::NodeIt region1(rg); region1 != lemon::INVALID; ++region1)
    {
        for (lemon::ListDigraph::NodeIt region2(rg); region2 != lemon::INVALID; ++region2)
        {
            if ((region1 != region2) && (region_level[region1] == region_level[region2]-1))
            {
                // std::cout << "region_qubits[region1] = " << region_qubits[region1] << "\n";
                // std::cout << "region_qubits[region2] = " << region_qubits[region2] << "\n";
                // std::cout << "xt::all(xt::isin(region_qubits[region2],region_qubits[region1]) = " << xt::all(xt::isin(region_qubits[region2],region_qubits[region1])) << "\n";
                if (xt::all(xt::isin(region_qubits[region2],region_qubits[region1]))==1) 
                {
                    lemon::ListDigraph::Arc new_edge = rg.addArc(region1,region2);
                    vars_to_marginalize[new_edge] = xt::setdiff1d(region_qubits[region1],region_qubits[region2]);
                }
            }
            // else if ((region1 != region2) && (region_level[region1]-1 == region_level[region2]))
            // {
            //     if (xt::sum(xt::isin(region_qubits[region2],region_qubits[region1]))()) 
            //     {
            //         lemon::ListDigraph::Arc new_edge = rg.addArc(region2,region1);
            //         vars_to_marginalize[new_edge] = xt::setdiff1d(region_qubits[region2],region_qubits[region1]);
            //     }
            // }
        }
    }
}

void RegionGraph::get_descendants_N_D()
{
    // construct depth-first search
    lemon::Dfs<lemon::ListDigraph> dfs(rg);
    // descendants
    for (lemon::ListDigraph::NodeIt region(rg); region != lemon::INVALID; ++region)
    {
        dfs.run(region);
        std::vector<int> desc;
        for (lemon::ListDigraph::NodeIt r2(rg); r2 != lemon::INVALID; ++r2)
        {
            if ((dfs.reached(r2) == true) && (r2 != region))
            {
                desc.push_back(rg.id(r2));
            }
        }
        descendants[region] = xt::adapt(desc);
        // std::cout << "descendants[region = " << rg.id(region) << "] = " << descendants[region] << "\n";
    }

    // N and D

    for (lemon::ListDigraph::ArcIt edge(rg); edge != lemon::INVALID; ++edge)
    {
        xt::xarray<int> i_source = {rg.id(rg.source(edge))};
        xt::xarray<int> i_target = {rg.id(rg.target(edge))};
        xt::xarray<int> E_P = union1d(i_source,descendants[rg.source(edge)]);
        xt::xarray<int> E_R = union1d(i_target,descendants[rg.target(edge)]);

        std::vector<int> n;
        std::vector<int> d;

        for (lemon::ListDigraph::ArcIt e2(rg); e2 != lemon::INVALID; ++e2)
        {
            auto condition_I_N = !(xt::isin(rg.id(rg.source(e2)),E_P));
            auto condition_J_N = xt::isin(rg.id(rg.target(e2)),xt::setdiff1d(E_P,E_R));

            if (condition_I_N() && condition_J_N())
            {
                n.push_back(rg.id(e2));
            }

            auto condition_I_D = xt::isin(rg.id(rg.source(e2)),xt::setdiff1d(E_P,E_R));
            auto condition_J_D = xt::isin(rg.id(rg.target(e2)),E_R);

            // if ((condition_I_D() && condition_J_D()) && (e2 != edge))
            // {
            //     d.push_back(rg.id(e2));
            // }
            if (condition_I_D() && condition_J_D())
            {
                d.push_back(rg.id(e2));
            }
        }

        N[edge] = xt::adapt(n);
        D[edge] = xt::adapt(d);

        // std::cout << "N[edge = " << rg.id(edge) << "] = " << N[edge] << "\n";
        // std::cout << "D[edge = " << rg.id(edge) << "] = " << D[edge] << "\n";


    }
}

void RegionGraph::get_ancestors_counting_numbers()
{
    // construct depth-first search on reversed graph
    lemon::ReverseDigraph<lemon::ListDigraph> reversed_rg(rg);
    lemon::Dfs<lemon::ReverseDigraph<lemon::ListDigraph>> dfs(reversed_rg);
    // ancestors
    for (lemon::ReverseDigraph<lemon::ListDigraph>::NodeIt region(reversed_rg); region != lemon::INVALID; ++region)
    {
        dfs.run(region);
        std::vector<int> ancs;
        for (lemon::ReverseDigraph<lemon::ListDigraph>::NodeIt r2(reversed_rg); r2 != lemon::INVALID; ++r2)
        {
            if ((dfs.reached(r2) == true) && (r2 != region))
            {
                ancs.push_back(rg.id(r2));
            }
        }
        ancestors[region] = xt::adapt(ancs);

        if (ancestors[region].size() == 0)
        {
            counting_number[region] = 1;
        }
    }

    for (int r_id = 0; r_id < n_regions; r_id++)
    {
        if (counting_number[rg.nodeFromId(r_id)] == 0)
        {
            counting_number[rg.nodeFromId(r_id)] = 1;
            for (auto a = ancestors[rg.nodeFromId(r_id)].begin(); a != ancestors[rg.nodeFromId(r_id)].end(); ++a)
            {
                counting_number[rg.nodeFromId(r_id)] -= counting_number[rg.nodeFromId(*a)];
            }
        }
    }
    // for (lemon::ListDigraph::NodeIt region(rg); region != lemon::INVALID; ++region)
    // {
    //     std::cout << "ancestors[region = " << rg.id(region) << "] = " << ancestors[region] << "\n";
    //     std::cout << "counting_number[region = " << rg.id(region) << "] = " << counting_number[region] << "\n";
    // }

}

void RegionGraph::print_regiongraph(std::string suffix)
{
    typedef lemon::dim2::Point<int> Point;
    lemon::ListDigraph::NodeMap<Point>coords(rg);
    lemon::ListDigraph::NodeMap<int> shapes(rg,1);
    lemon::ListDigraph::ArcMap<double> widths(rg,0.5);

    lemon::ListDigraph::NodeMap<std::string> node_texts(rg);

    xt::xarray<int> left_coords = xt::zeros<int>({max_level});

    for (lemon::ListDigraph::NodeIt region(rg); region != lemon::INVALID; ++region)
    {
        int right_coord = 100 - region_level[region] * 40;
        coords[region] = Point(left_coords[region_level[region]],right_coord);
        left_coords[region_level[region]] += 30;
        std::stringstream nt;
        nt << rg.id(region) << " | " << region_checks[region] << " | " << region_qubits[region];
        node_texts[region] =  nt.str();
        // std::cout << nt.str() << "\n";
    }

    std::string name = "region_graph_" + suffix +".eps";

    lemon::graphToEps(rg,name)
        .coords(coords)
        .arcWidths(widths)
        .nodeTexts(node_texts).nodeTextSize(1).nodeShapes(shapes)
        .border(200,10)
        .drawArrows(true)
        .run();

}

// template <typename BPD>
// xt::xarray<int> get_H_sub(BPD* bpDecoder, xt::xarray<int> H, xt::xarray<int> s, xt::xarray<int> * s_sub, xt::xarray<int> * c_indices, xt::xarray<int> * q_indices, int pauli)
// {


//     unsigned int n_c = H.shape(0);
//     unsigned int n_q = H.shape(1);

//     xt::xarray<long double> marginals = bpDecoder->get_marginals();
//     xt::xarray<long double> messages = bpDecoder->get_messages();
//     xt::xarray<int> hard_decisions = bpDecoder->get_hard_decisions();
//     xt::xarray<int> syndromes = bpDecoder->get_syndromes();

//     xt::xarray<long double> abs_diffs = xt::abs(xt::diff(messages,1,1));
//     xt::xarray<long double> abs_diffs_cq = xt::sum(xt::view(abs_diffs,xt::all(),xt::range(-10,_),0),1);
//     xt::xarray<long double> abs_diffs_qc = xt::sum(xt::view(abs_diffs,xt::all(),xt::range(-10,_),1),1);

//     xt::xarray<int> abs_diffs_s = xt::abs(xt::diff(syndromes,1,0));
//     xt::xarray<int> diffs_s_sum = xt::sum(xt::view(abs_diffs_s,xt::range(-20,_),xt::all()),0);
//     xt::xarray<int> syndromes_changing = xt::from_indices(xt::argwhere(diffs_s_sum > 0));


//     std::cout << "syndromes_changing = " << syndromes_changing << "\n  shape = " << xt::adapt(syndromes_changing.shape()) << std::endl;

//     xt::xarray<int> abs_diffs_hd = xt::abs(xt::diff(hard_decisions,1,0));
//     xt::xarray<int> diffs_hd_sum = xt::sum(xt::view(abs_diffs_hd,xt::range(2,_),xt::all()),0);
//     xt::xarray<int> hd_changing = xt::from_indices(xt::argwhere(diffs_hd_sum > 0));

//     std::ofstream OUTPUT_FILE;
//     OUTPUT_FILE.open("test.out");

//     OUTPUT_FILE << "edge\tabs_diffs_cq\tabs_diffs_qc \n";
//     for (int i = 0; i<abs_diffs_cq.size();i++)
//     {
//         OUTPUT_FILE  << i << "\t" << abs_diffs_cq(i)  << "\t" << abs_diffs_qc(i)   << "\n";
//     }

//     OUTPUT_FILE.close();


//     xt::xarray<int> edges_changing_qc= xt::from_indices(xt::argwhere(abs_diffs_qc > 0));
//     std::cout << "# edges_changing_qc = " << xt::adapt(edges_changing_qc.shape())(0) << std::endl;

//     std::set<int> checks_involved;
//     std::set<int> qubits_involved;
//     // qubits and checks involved
//     // for (auto it = syndromes_changing.begin(); it != syndromes_changing.end(); ++it)
//     // {
//     //     checks_involved.insert(*it);
//         // for (int q = 0; q < n_q; q++)
//         // {
//         //     if (xt::isin(q,hd_changing))
//         //     {
//         //         qubits_involved.insert(q);
//         //     }
//         // }
//     // }

//     // for (auto it = hd_changing.begin(); it != hd_changing.end(); ++it)
//     // {
//     //     qubits_involved.insert(*it);
//     // }



//     for (auto it = edges_changing_qc.begin(); it != edges_changing_qc.end(); ++it)
//     {
//         xt::xarray<int> cq = bpDecoder->get_check_and_qubit(*it);

//         if (H(cq(0),cq(1)) == pauli)
//         {
//             checks_involved.insert(cq(0));
//             qubits_involved.insert(cq(1));
//         }
        
//     }

//     std::cout << "checks_involved (" << checks_involved.size() << ") : \n";
//     for (auto it = checks_involved.begin(); it != checks_involved.end(); ++it)
//     {
//         std::cout << *it << ",";
//     }
//     std::cout << std::endl;
//     std::cout << "qubits_involved (" << qubits_involved.size() << ") : \n";
//     for (auto it = qubits_involved.begin(); it != qubits_involved.end(); ++it)
//     {
//         std::cout << *it << ",";
//     }
//     std::cout << std::endl;

//     std::vector<int> ci_v(checks_involved.begin(), checks_involved.end()); 
//     std::vector<int> qi_v(qubits_involved.begin(), qubits_involved.end()); 

//     xt::xarray<int> ci = xt::adapt(ci_v);
//     xt::xarray<int> qi = xt::adapt(qi_v);


//     unsigned int n_qi = qi.size();
//     unsigned int n_ci = ci.size();

//     s_sub->resize({n_ci});

//     c_indices->resize({n_ci});
//     q_indices->resize({n_qi});

//     *c_indices = ci;
//     *q_indices = qi;


//     // H_sub
//     xt::xarray<int> H_sub_c;
//     H_sub_c.resize({n_ci,n_q});

//     H_sub_c = xt::view(H,xt::keep(ci),xt::all());

//     xt::xarray<int> H_sub;
//     H_sub.resize({n_ci,n_qi});
//     H_sub = xt::view(H_sub_c,xt::all(),xt::keep(qi));

//     for (int c = 0; c < n_ci; c++)
//     {
//         if (s(ci(c)) != 0)
//         {
//             s_sub->at(c) = 1;
//         }
//         else
//         {
//             s_sub->at(c) = 0;
//         }
//         for (int q = 0; q < n_qi; q++)
//         {
//             if (H_sub(c,q) == pauli)
//             {
//                 H_sub(c,q) = 1;
//             }
//         }
//     }

//     return H_sub;
// }