#ifndef DECODER_GBP_HPP_
#define DECODER_GBP_HPP_

#include <bp_gbp/constructor_gbp.hpp>
#include <bp_gbp/la_tools.hpp>
#include <bp_gbp/io_tools.hpp>
#include <bp_gbp/timing.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include "xtensor/xmasked_view.hpp"
#include "xtensor/xnpy.hpp" // load_npy, dump_npy
#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xframe/xvariable_view.hpp"

#include <cppitertools/product.hpp>

#include <chrono>

// tuple to valarray based on https://stackoverflow.com/questions/42494715/c-transform-a-stdtuplea-a-a-to-a-stdvector-or-stddeque
template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::valarray<T> to_valarray(Tuple &&tuple)
{
  return std::experimental::apply([](auto &&...elems) {
    return std::valarray<T>{std::forward<decltype(elems)>(elems)...};
  },
                    std::forward<Tuple>(tuple));
}

class GbpDecoder
{
    using coordinate_type = xf::xcoordinate<xf::fstring>;
    using dimension_type = xf::xdimension<xf::fstring>;
    using variable_type = xf::xvariable<long double, coordinate_type>;
    using data_type = variable_type::data_type;
    private:
        int max_iterations;
        int n_c;
        int n_q;
        int rank_H;
        xt::xarray<int> H;

        std::vector<variable_type> q_factors;
        std::vector<std::vector<variable_type>> c_factors;

        std::vector<std::vector< xt::xarray<long double> > > marginals;

        xt::xarray<int> hard_decisions;
        xt::xarray<int> syndromes;

        xt::xarray<long double> free_energy;

        RegionGraph RG;

        void fill_c_factors();
        void fill_q_factors(xt::xarray<long double> p_initial);
        void prepare_messages(const xt::xarray<int> *s_0);
        void prepare_beliefs(const xt::xarray<int> *s_0);
        void update_messages(const xt::xarray<int> *s_0, xt::xarray<long double> p_initial,long double w_gbp, int iteration);
        void update_messages_2(const xt::xarray<int> *s_0, xt::xarray<long double> p_initial,long double w_gbp, int iteration);
        void update_beliefs(const xt::xarray<int> *s_0,int iteration);
        
        void get_marginals_and_hard_dec(int iteration);
        void get_marginals_and_hard_dec_2(int iteration);

        void calculate_free_energy(int iteration);

    public:
        int took_iterations;
        GbpDecoder(xt::xarray<int> H, int max_iterations, int n_checks_per_r0, int rg_type);

        GbpDecoder(xt::xarray<int> H, int max_iterations, int n_checks_per_r0, xt::xarray<int> check_list, int rg_type);

        xt::xarray<int> decode(xt::xarray<int> s_0,xt::xarray<long double> p_initial,int type_gbp,long double w_gbp, int return_strategy, bool return_if_success);
        xt::xarray<long double> get_messages();
        xt::xarray<long double> get_marginals();
        xt::xarray<long double> get_free_energy();
        xt::xarray<int> get_hard_decisions();
        xt::xarray<int> get_syndromes();
        xt::xarray<bool> get_convergence();
};

template <typename BPD>
xt::xarray<int> gbpPP(xt::xarray<int> residual_s, xt::xarray<int> H_X, xt::xarray<int> H_Z,BPD* bpDecoder, xt::xarray<long double> p_init ,int n_checks_per_r0, long double w_gbp, int type_gbp, int return_strategy, int repetitions_split, bool return_if_success, bool print_details, int* took_iterations, bool save_raw_data, std::string OUTPUT_DIR, bool surface)
{
    int n_c_X = H_X.shape(0);
    int n_c_Z = H_Z.shape(0);
    int n_c = n_c_X + n_c_Z;
    int n_q = H_X.shape(1);

    int n_q_base = (int) (sqrt(n_q-n_c) + sqrt(n_q+n_c))/2;
    int n_c_base = (int) (sqrt(n_q+n_c) - sqrt(n_q-n_c))/2;

    xt::xarray<int> H = xt::zeros<int>({n_c, n_q});
    auto H_X_view = xt::view(H, xt::range(0, n_c_X), xt::all());
    auto H_Z_view = xt::view(H, xt::range(n_c_X, n_q), xt::all());

    H_X_view = H_X;
    H_Z_view = 2 * H_Z;

    xt::xarray<int> from_gbp = xt::zeros<int>({n_q});
    
    xt::xarray<int> residual_error_sub;

    xt::xarray<int> residual_s_x = xt::view(residual_s, xt::range(n_c_X,n_c_X+n_c_Z));
    xt::xarray<int> residual_s_z = xt::view(residual_s, xt::range(0, n_c_X));

    if (print_details == true)
    {
        std::cout << "\n***** gbp: start *****\n";

        print_container(residual_s, "bp: residual_s", true);

        print_container(residual_s_x, "bp: residual_s_x", true);
        print_container(residual_s_z, "bp: residual_s_z", true);
    }

    xt::xarray<std::string> pauli_strings = {"z","x"};

    for (int pauli = 1; pauli <= 2; pauli++)
    {
        if ((pauli == 1) && (xt::sum(residual_s_x)() == 0))
        {
            pauli++;
        }
        if ((pauli == 2) && (xt::sum(residual_s_z)() == 0))
        {
            break;
        }
        xt::xarray<int> s_sub_0;
        xt::xarray<int> qi;
        xt::xarray<int> ci;

        xt::xarray<int> H_sub;
        if (surface == true)
        {
            if (pauli == 1)
            {
                H_sub = H_Z;
                s_sub_0 = residual_s_x;
                qi = xt::arange<int>(0,n_q,1);
                ci = xt::arange<int>(0,n_c_X,1);
            }
                
            else if (pauli == 2)
            {    
                H_sub = H_X;
                s_sub_0 = residual_s_z;
                qi = xt::arange<int>(0,n_q,1);
                ci = xt::arange<int>(0,n_c_Z,1);
            }
        }
        else
        {
            H_sub = get_H_sub(bpDecoder, H, residual_s, &s_sub_0, &ci, &qi, 3 - pauli,print_details);
        }

        int n_ci = H_sub.shape(0);
        int n_qi = H_sub.shape(1);

        if (print_details == true)
        {
            std::cout << "H_sub: (" << n_ci << "," << n_qi << ")\n";// << H_sub << std::endl;
            std::cout << "ci = " << ci << "\n";
            std::cout << "qi = " << qi << "\n";
            std::cout << "s_sub_0 = " << s_sub_0 << std::endl;
        }

        bool start_gbp = false;
        bool disjoint = false;
        int rg_type = 0;

        if (surface)
        {
            start_gbp = true;
            rg_type = pauli;
        }
        else
        {
            bool condition = (n_qi == n_q_base);
            bool condition_T = (n_qi == n_c_base);

            bool disjoint_H = ((n_qi % n_q_base == 0 ) && (n_qi > n_q_base) && (n_qi < 3*n_q_base));
            bool disjoint_T = ((n_qi % n_c_base == 0 ) && (n_qi > n_c_base) && (n_qi < 3*n_c_base));
            disjoint = (disjoint_H || disjoint_T);
            start_gbp = ((condition || condition_T) || disjoint );
        }

        if (start_gbp)
        {
            int max_iter_gbp = 35;
            long double new_p;
            if (surface == true)
            {
                new_p = p_init(pauli);
            }
            else
            {
                new_p = p_init(pauli);
            }

            xt::xarray<long double> p_initial_gf2 = {1 - new_p, new_p};
            
            xt::xarray<int> error_guess_sub = xt::zeros<int>({n_qi});
            if (disjoint)
            {
                // GbpDecoder gbpDecoder(H_sub, max_iter_gbp, n_checks_per_r0, rg_type);
                xt::xarray<int> check_list = xt::arange<int>({n_ci});
                int n_disjoint_blocks;
                if (n_ci > n_qi) 
                {
                    n_disjoint_blocks = (int) n_qi / n_c_base;
                }
                else
                {
                    n_disjoint_blocks = (int) n_qi / n_q_base;
                }

                if (n_disjoint_blocks == 0) {n_disjoint_blocks = 1;}
                
                int len_disjoint_check_block = (int) n_ci / n_disjoint_blocks;

                for (int i = 0; i < n_disjoint_blocks; i++)
                {
                    auto v = xt::view(check_list,xt::range(i*len_disjoint_check_block,(i+1)*len_disjoint_check_block));
                    xt::random::shuffle(v);
                }

                GbpDecoder gbpDecoder(H_sub, max_iter_gbp, n_checks_per_r0, check_list, rg_type);

                error_guess_sub = gbpDecoder.decode(s_sub_0, p_initial_gf2, type_gbp, w_gbp,return_strategy, return_if_success);
                *took_iterations = gbpDecoder.took_iterations;
                if (save_raw_data == true)
                {
                    xt::xarray<long double> marginals = gbpDecoder.get_marginals();
                    xt::xarray<long double> messages = gbpDecoder.get_messages();
                    xt::xarray<int> hard_decisions = gbpDecoder.get_hard_decisions();
                    xt::xarray<int> free_energy = gbpDecoder.get_free_energy();
                    xt::xarray<int> syndromes = gbpDecoder.get_syndromes();

                    xt::dump_npy(OUTPUT_DIR + "/marginals_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy", marginals);
                    xt::dump_npy(OUTPUT_DIR + "/messages_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy", messages);
                    xt::dump_npy(OUTPUT_DIR + "/hard_decisions_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy", hard_decisions);
                    xt::dump_npy(OUTPUT_DIR+"/syndromes_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy",syndromes);
                    xt::dump_npy(OUTPUT_DIR+"/free_energy_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy",free_energy);
                }
            }
            else
            {
                GbpDecoder gbpDecoder(H_sub, max_iter_gbp, n_checks_per_r0, rg_type);

                error_guess_sub = gbpDecoder.decode(s_sub_0, p_initial_gf2, type_gbp, w_gbp, return_strategy, return_if_success);
                *took_iterations = gbpDecoder.took_iterations;
                if (save_raw_data == true)
                {
                    xt::xarray<long double> marginals = gbpDecoder.get_marginals();
                    xt::xarray<long double> messages = gbpDecoder.get_messages();
                    xt::xarray<int> hard_decisions = gbpDecoder.get_hard_decisions();
                    xt::xarray<int> free_energy = gbpDecoder.get_free_energy();
                    xt::xarray<int> syndromes = gbpDecoder.get_syndromes();

                    xt::dump_npy(OUTPUT_DIR + "/marginals_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy", marginals);
                    xt::dump_npy(OUTPUT_DIR + "/messages_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy", messages);
                    xt::dump_npy(OUTPUT_DIR + "/hard_decisions_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy", hard_decisions);
                    xt::dump_npy(OUTPUT_DIR+"/syndromes_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy",syndromes);
                    xt::dump_npy(OUTPUT_DIR+"/free_energy_"+ pauli_strings(pauli-1) + "_" + std::to_string(repetitions_split) + ".npy",free_energy);
                }
            }
            
            for (int i = 0; i < n_qi; i++)
            {
                if (error_guess_sub(i) == 1)
                {
                    from_gbp(qi(i)) ^= pauli;
                }
            }
        }
    }
    return from_gbp;
}

#endif