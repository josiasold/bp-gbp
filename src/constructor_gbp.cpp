#include <bp_gbp/constructor_gbp.hpp>

// constructor for region graph with linear check list
RegionGraph::RegionGraph(xt::xarray<int> H, int n_checks_per_r0, int rg_type) : H(H), region_qubits(rg), region_checks(rg), region_level(rg), belief(rg), belief_base(rg), vars_to_marginalize(rg),dim_of_vars_to_marg(rg), message(rg), edge_converged(rg), region_converged(rg), message_base(rg), N(rg), D(rg),counting_number(rg), ancestors(rg), descendants(rg)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    checks = xt::arange<int>(n_c);
    qubits = xt::arange<int>(n_q);
    construct_rg(n_checks_per_r0,rg_type,false);
}

// constructor for region graph with given check list determining the order of check put in superregions
RegionGraph::RegionGraph(xt::xarray<int> H, int n_checks_per_r0, xt::xarray<int> check_list, int rg_type) : H(H), region_qubits(rg), region_checks(rg), region_level(rg), belief(rg), belief_base(rg), vars_to_marginalize(rg),dim_of_vars_to_marg(rg), message(rg), edge_converged(rg), region_converged(rg), message_base(rg), N(rg), D(rg),counting_number(rg), ancestors(rg), descendants(rg)
{
    n_c = H.shape(0);
    n_q = H.shape(1);
    checks = xt::arange<int>(n_c);
    qubits = xt::arange<int>(n_q);
    construct_rg(n_checks_per_r0,check_list,rg_type,false);
}

void RegionGraph::construct_rg(int n_checks_per_r0, int rg_type, bool save_rg)
{
    xt::random::seed(time(NULL));
    xt::xarray<int> check_list = xt::arange<int>(n_c);
    
    if (rg_type == 0) // shuffle checks --> random assignement
    {
        xt::random::shuffle(check_list);
    }
    else if (rg_type == 2) // split checks in stripes
    {
        int dist = (int)((1 + sqrt(1+4*n_c))/2);
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

    std::vector<int> used_checks;

    int n_r0 =  ceil((double)n_c / (double) n_checks_per_r0);
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

        region_level[new_region] = 0;
    }



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


    // edges
    make_edges();

    n_regions=0;
    for (lemon::ListDigraph::NodeIt r(rg); r!=lemon::INVALID; ++r) ++n_regions;

    n_edges=0;
    for (lemon::ListDigraph::ArcIt e(rg); e!=lemon::INVALID; ++e) ++n_edges;

    get_descendants_N_D();

    get_ancestors_counting_numbers();

    std::string suffix = std::to_string(n_checks_per_r0);
    if (save_rg)
    {
        save_regiongraph(suffix);
    }
    

}

void RegionGraph::construct_rg(int n_checks_per_r0, xt::xarray<int> check_list, int rg_type, bool save_rg)
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
        region_level[new_region] = 0;
    }


    // other regions
    bool done = false;
    int current_level = 1;
    xt::xarray<int> single_qubits_used = xt::arange<int>(n_q);
    
    while (done == false)
    {
        std::unordered_set< xt::xarray<int>, xarrayHash > intersections = find_intersections(current_level-1);

        if (intersections.size() > 0)
        {
            for (auto it = intersections.begin(); it != intersections.end(); ++it)
            {
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

    // edges
    make_edges();

    n_regions=0;
    for (lemon::ListDigraph::NodeIt r(rg); r!=lemon::INVALID; ++r) ++n_regions;

    n_edges=0;
    for (lemon::ListDigraph::ArcIt e(rg); e!=lemon::INVALID; ++e) ++n_edges;

    get_descendants_N_D();

    get_ancestors_counting_numbers();

    std::string suffix = std::to_string(n_checks_per_r0);
    if (save_rg)
    {
        save_regiongraph(suffix);
    }

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
                if (xt::all(xt::isin(region_qubits[region2],region_qubits[region1]))==1) 
                {
                    lemon::ListDigraph::Arc new_edge = rg.addArc(region1,region2);
                    vars_to_marginalize[new_edge] = xt::setdiff1d(region_qubits[region1],region_qubits[region2]);
                }
            }
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

            if (condition_I_D() && condition_J_D())
            {
                d.push_back(rg.id(e2));
            }
        }

        N[edge] = xt::adapt(n);
        D[edge] = xt::adapt(d);

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
}

void RegionGraph::save_regiongraph(std::string suffix)
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
