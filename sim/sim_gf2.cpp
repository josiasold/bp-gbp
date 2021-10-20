#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <experimental/filesystem>

#include <valarray>
#include <vector>
#include <set>

#include <bp_gbp/io_tools.hpp>
#include <bp_gbp/la_tools.hpp>
#include <bp_gbp/error_channels.hpp>
#include <bp_gbp/decoder_bp2.hpp>
#include <bp_gbp/constructor_gbp.hpp>
#include <bp_gbp/decoder_gbp.hpp>
// #include <bp_gbp/gbp_functions.hpp>

#include "xtensor/xarray.hpp"
#include "xtensor/xnpy.hpp" // load_npy, dump_npy
#include "xtensor/xio.hpp"  // <<
#include "xtensor/xrandom.hpp" // for seeding random generator
#include <xtensor/xindex_view.hpp>

#include "nlohmann/json.hpp"

int main(int argc, char **argv)
{
    // Add Time
    time_t start_time_raw;
    struct tm *start_time;
    time(&start_time_raw);
    start_time = localtime(&start_time_raw);
    std::cout << "- start time = " << asctime(start_time);
    std::stringstream str_time;
    str_time << std::put_time(start_time, "%m%d_%H%M%S");

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " PATH_TO_INPUT_FILE PATH_TO_OUTPUT_DIR" << std::endl;
        return 1;
    }
    // handle input
    std::string PATH_TO_INPUT_FILE = argv[1];
    std::string PATH_TO_OUTPUT_DIR = argv[2];
    std::cout << "Input: " << PATH_TO_INPUT_FILE << std::endl;

    std::ifstream json_input_file(PATH_TO_INPUT_FILE);
    nlohmann::json json_input;
    json_input_file >> json_input;

    int n_errorsamples = json_input.at("n_errorsamples");
    long double p = json_input.at("p_error");
    long double p_initial_bp = json_input.at("p_initial_bp");
    std::string channel = json_input.at("channel");

    long double w_gbp = json_input.at("w_gbp");
    long double w_bp = json_input.at("w_bp");
    long double alpha_bp = json_input.at("alpha_bp");
    int type_gbp = json_input.at("type_gbp");
    int type_bp = json_input.at("type_bp");
    int max_checks_per_r0 = json_input.at("max_checks_per_r0");

    std::string pathToCodes = json_input.at("path_to_codes");
    std::string code = json_input.at("code");
    std::string pathToH_X = pathToCodes + code + "_hx.npy";
    std::string pathToH_Z = pathToCodes + code + "_hz.npy";

    bool gbp_pp = json_input.at("gbp_pp");
    bool repeat_p_init = json_input.value("repeat_p_init",false);
    bool repeat_split = json_input.at("repeat_split");
    int split_strategy = json_input.value("split_strategy",0);

    bool get_marg_mess = json_input.at("get_marg_mess");
    bool print_bp_details = json_input.at("print_bp_details");
    bool print_gbp_details = json_input.at("print_gbp_details");
    bool print_fails = json_input.at("print_fails");

    bool return_if_success = json_input.at("return_if_success");
    bool only_non_converged = json_input.at("only_non_converged");

    std::string output_suffix = json_input.at("output_suffix");
    std::string OUTPUT_DIR;
    if (output_suffix == "")
    {
        OUTPUT_DIR = PATH_TO_OUTPUT_DIR + "/" + str_time.str();
    }
    else if (output_suffix == "tmp")
    {
        OUTPUT_DIR = PATH_TO_OUTPUT_DIR + "/tmp";
        if (std::experimental::filesystem::exists(OUTPUT_DIR))
        {
            std::experimental::filesystem::remove_all(OUTPUT_DIR);
        }
    }
    else
    {
        OUTPUT_DIR = PATH_TO_OUTPUT_DIR + "/" + str_time.str() + "_" + output_suffix;
    }

    std::experimental::filesystem::create_directory(OUTPUT_DIR);
    std::experimental::filesystem::copy(PATH_TO_INPUT_FILE, OUTPUT_DIR);
    std::ofstream OUTPUT_FILE;
    OUTPUT_FILE.open(OUTPUT_DIR + "/" + code + ".out");

    OUTPUT_FILE << "- start time = " << asctime(start_time);

    xt::xarray<int> H_X = xt::load_npy<int>(pathToH_X);
    xt::xarray<int> H_Z = xt::load_npy<int>(pathToH_Z);

    int n_c_X = H_X.shape()[0];
    int n_c_Z = H_Z.shape()[0];
    int n_q = H_X.shape()[1];
    int n_c = n_c_X + n_c_Z;



    xt::xarray<int> H = xt::zeros<int>({n_c, n_q});
    auto H_X_view = xt::view(H, xt::range(0, n_c_X), xt::all());
    auto H_Z_view = xt::view(H, xt::range(n_c_X, n_q), xt::all());

    H_X_view = H_X;
    H_Z_view = 2 * H_Z;

    // std::cout << "H\n";


    int rank_H_X = gf2_rank(H_X.data(), n_c_X, n_q);
    int rank_H_Z = gf2_rank(H_Z.data(), n_c_Z, n_q);
    int rank_H = gf4_rank(H.data(), n_c, n_q);

    std::cout << "rank(H_X) = " << rank_H_X << std::endl;
    std::cout << "rank(H_Z) = " << rank_H_Z << std::endl;
    std::cout << "rank(H) = " << rank_H << std::endl;

    // construct random generator
    // std::random_device rd;                                  // obtain a random number from hardware
    // std::mt19937 random_generator(rd());                    // seed the generator
    // xt::random::seed(time(NULL));
    // construct channel
    NoisyChannel noisyChannel;

    // construct BpDecoder
    Bp2Decoder bpDecoder_X(H_Z); // decoder for X-errors --> use Z-pcm
    Bp2Decoder bpDecoder_Z(H_X); // decoder for Z-errors --> use X-pcm

    // error channel
    xt::xarray<long double> ps;
    if (p == -1)
    {
        // ps = {0.07,0.075,0.08,0.085,0.09,0.095,0.1};
        ps = {0.0001,0.001,0.005,0.01};
    }
    else if (p == -2)
    {
        ps = {0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09};
    }
    else if (p == -3)
    {
        ps.resize({100});
        for (size_t i = 1; i < 101; i++)
        {
            ps[i - 1] = i * 0.01;
        }
    }
    else if (p == -4)
    {
        ps = {0.09,0.095,0.1,0.105,0.11,0.115,0.12};
    }
    else
    {
        ps = {p};
    }

    int max_iter = 100;

    int ch_sc = 0;
    int dec_sci = 0;
    int dec_sce = 0;
    int dec_ler = 0;
    int dec_fail = 0;
    int dec_gbp_sci = 0;
    int dec_gbp_sce = 0;
    int dec_gbp_ler = 0;
    int dec_gbp_fail = 0;

    std::cout << "p\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tber" << std::endl;
    OUTPUT_FILE << "p\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tber" << std::endl;
    // initial word.
    xt::xarray<int> x = xt::zeros<int>({n_q});
    // std::cout << "x = " << x << std::endl;

    for (size_t i_p = 0; i_p < ps.size(); i_p++)
    {
        ch_sc = 0;
        dec_sci = 0;
        dec_sce = 0;
        dec_ler = 0;
        dec_fail = 0;
        dec_gbp_sci = 0;
        dec_gbp_sce = 0;
        dec_gbp_ler = 0;
        dec_gbp_fail = 0;

        double p_error = ps(i_p);
        xt::xarray<long double> p_initial;
        xt::xarray<long double> p_initial_gf2;
        if (channel == "xz")
        {
            long double p_x = sqrt(1-p_error) * (1-sqrt(1-p_error));
            long double p_y = (1-sqrt(1-p_error)) * (1-sqrt(1-p_error));
            p_initial = {1 - p_error, p_x,p_y,p_x};
            p_initial_gf2 = {1 - p_x+p_y, p_x+p_y};
        }
        else
        {
            p_initial = {1 - p_error, p_error / 3.0, p_error / 3.0, p_error / 3.0};
            p_initial_gf2 = {1 - 2 * p_error / 3, 2 * p_error / 3};
        }
       
        
        if (p_initial_bp == -1)
        {
            bpDecoder_X.initialize_bp(p_initial_gf2, max_iter);
            bpDecoder_Z.initialize_bp(p_initial_gf2, max_iter);
        }
        else
        {
            xt::xarray<long double> p_initial_for_decoder = {1 - 2 * p_initial_bp / 3.0, 2 * p_initial_bp / 3.0};
            bpDecoder_X.initialize_bp(p_initial_for_decoder, max_iter);
            bpDecoder_Z.initialize_bp(p_initial_for_decoder, max_iter);
        }
        
        

        for (size_t i_e = 0; i_e < n_errorsamples; i_e++)
        {
            xt::xarray<int> y = x;
            if (channel == "depolarizing")
            {
                // depolarizing_channel(&y, p_error,&random_generator);
                noisyChannel.send_through_pauli_channel(&y,p_error,0);
            }
            else if (channel == "xz")
            {
                // xz_channel(&y, p_error,&random_generator);
                noisyChannel.send_through_pauli_channel(&y,p_error,1);
            }
            else if (channel == "custom")
            {
                y(2) = 1;
                y(3) = 1;
            }
            else if (channel == "const_weight")
            {
                int weight = (int)(p_error * n_q);
                // const_weight_error_channel(&y, weight,&random_generator);
                noisyChannel.const_weight_error_channel(&y, weight);
            }
            else if (channel == "const_weight_restricted")
            {
                int weight = (int)(p_error * n_q);
                // const_weight_error_channel(&y, weight, 16, 1, &random_generator);
                noisyChannel.const_weight_error_channel(&y, weight, 16, 1);
            }
            else
            {
                std::cerr << "Not a valid channel, choose frome {'depolarizing', 'custom', 'const_weight', 'const_weight_restricted'}" << std::endl;
                return 1;
            }
            if (y == x)
            {
                ch_sc++;
            }
            else
            {
                xt::xarray<int> y_x = get_x(y);
                xt::xarray<int> y_z = get_z(y);
                xt::xarray<int> s_0_x = gf2_syndrome(&y_x, &H_Z);
                xt::xarray<int> s_0_z = gf2_syndrome(&y_z, &H_X);

                xt::xarray<int> s_0 = gf4_syndrome(&y, &H);

                xt::xarray<int> error_guess_x = bpDecoder_X.decode_bp(s_0_x, w_bp, alpha_bp, type_bp, return_if_success, only_non_converged);
                xt::xarray<int> error_guess_z = bpDecoder_Z.decode_bp(s_0_z, w_bp, alpha_bp, type_bp,return_if_success, only_non_converged);

                xt::xarray<int> error_guess = error_guess_x + 2 * error_guess_z;

                xt::xarray<int> residual_error = error_guess ^ y;

                if (print_bp_details)
                {
                    print_container(y,"y",true);
                    print_container(s_0,"s_0",true);
                    print_container(error_guess,"error_guess",true);
                    print_container(residual_error,"residual_error",true);
                }

                // std::cout << "error_guess = " << error_guess << std::endl;
                if (get_marg_mess == true)
                {
                    xt::xarray<long double> marginals_x = bpDecoder_X.get_marginals();
                    xt::xarray<long double> m_qc_x = bpDecoder_X.get_m_qc();
                    xt::xarray<long double> m_cq_x = bpDecoder_X.get_m_cq();
                    xt::xarray<int> hard_decisions_x = bpDecoder_X.get_hard_decisions();
                    xt::xarray<int> syndromes_x = bpDecoder_X.get_syndromes();
                    xt::xarray<long double> free_energy_x = bpDecoder_X.get_free_energy();

                    xt::xarray<long double> marginals_z = bpDecoder_Z.get_marginals();
                    xt::xarray<long double> m_qc_z = bpDecoder_Z.get_m_qc();
                    xt::xarray<long double> m_cq_z = bpDecoder_Z.get_m_cq();
                    xt::xarray<int> hard_decisions_z = bpDecoder_Z.get_hard_decisions();
                    xt::xarray<int> syndromes_z = bpDecoder_Z.get_syndromes();
                    xt::xarray<long double> free_energy_z = bpDecoder_Z.get_free_energy();

                    xt::dump_npy(OUTPUT_DIR + "/marginals_x_bp.npy", marginals_x);
                    xt::dump_npy(OUTPUT_DIR + "/m_qc_x_bp.npy", m_qc_x);
                    xt::dump_npy(OUTPUT_DIR + "/m_cq_x_bp.npy", m_cq_x);
                    xt::dump_npy(OUTPUT_DIR + "/hard_decisions_x_bp.npy", hard_decisions_x);
                    xt::dump_npy(OUTPUT_DIR + "/syndromes_x_bp.npy", syndromes_x);
                    xt::dump_npy(OUTPUT_DIR+"/free_energy_x_bp.npy",free_energy_x);

                    xt::dump_npy(OUTPUT_DIR + "/marginals_z_bp.npy", marginals_z);
                    xt::dump_npy(OUTPUT_DIR + "/m_qc_z_bp.npy", m_qc_z);
                    xt::dump_npy(OUTPUT_DIR + "/m_cq_z_bp.npy", m_cq_z);
                    xt::dump_npy(OUTPUT_DIR + "/hard_decisions_z_bp.npy", hard_decisions_z);
                    xt::dump_npy(OUTPUT_DIR + "/syndromes_z_bp.npy", syndromes_z);
                    xt::dump_npy(OUTPUT_DIR+"/free_energy_z_bp.npy",free_energy_z);
                }


                xt::xarray<int> s = gf4_syndrome(&error_guess, &H);
                residual_error = y ^ error_guess;

                xt::xarray<int> residual_s = gf4_syndrome(&residual_error, &H);

                if (print_bp_details)
                {
                    print_container(residual_error,"residual_error",true);
                    print_container(residual_s,"residual_s",true);
                }

                if (s == s_0)
                {
                    if (residual_error == x)
                    {
                        dec_sci++;
                    }
                    else
                    {
                        if (gf4_isEquiv(residual_error, H, n_c, n_q))
                        {
                            dec_sce++;
                        }
                        else
                        {
                            dec_ler++;
                        }
                    }
                }
                else
                {
                    dec_fail++;
                    if (print_fails == true)
                    {
                        std::cout << "XX fail XX";
                        print_container(y, "y", true);
                    }
                }
            }
        }
        long double ber = (long double)(dec_ler + dec_fail) / (long double)n_errorsamples;
        if (print_gbp_details) std::cout << "p\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tber" << std::endl;
        std::cout << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << ber << std::endl;
        OUTPUT_FILE << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << ber << std::endl;
    }

    time_t stop_time_raw;
    time(&stop_time_raw);
    struct tm *stop_time;
    stop_time = localtime(&stop_time_raw);
    int total_elapsed_s = difftime(stop_time_raw, start_time_raw);
    std::cout << "- stop time = " << asctime(stop_time) << "total elapsed time = " << total_elapsed_s << " s" << std::endl;
    OUTPUT_FILE << "- stop time = " << asctime(stop_time) << "total elapsed time = " << total_elapsed_s << " s" << std::endl;
    OUTPUT_FILE.close();
}
