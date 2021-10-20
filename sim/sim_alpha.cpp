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
#include <bp_gbp/decoder_bp.hpp>
#include <bp_gbp/decoder_bp_kuolai.hpp>
#include <bp_gbp/decoder_bp_klm.hpp>
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
    long double p_error = json_input.at("p_error");
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

    int rank_H_X = gf2_rank(H_X.data(), n_c_X, n_q);
    int rank_H_Z = gf2_rank(H_Z.data(), n_c_Z, n_q);
    int rank_H = gf4_rank(H.data(), n_c, n_q);

    std::cout << "rank(H_X) = " << rank_H_X << std::endl;
    std::cout << "rank(H_Z) = " << rank_H_Z << std::endl;
    std::cout << "rank(H) = " << rank_H << std::endl;

    // construct random generator
    std::random_device rd;                                  // obtain a random number from hardware
    std::mt19937 random_generator(rd());                    // seed the generator
    xt::random::seed(time(NULL));
    // condtruct BpDecoder
    BpDecoderKL bpDecoder(H);

    // error channel
    xt::xarray<long double> alphas;
    if (alpha_bp == -1)
    {
        alphas = {0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5};
    }
    else if (alpha_bp == -2)
    {
        alphas = {1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7};
    }
    else if (alpha_bp == -3)
    {
        alphas = {0.7,0.75,0.8,0.85,0.9,0.92,0.94,0.96,0.98,0.99,1.0};
    }
    else if (alpha_bp == -4)
    {
        alphas = {0.65,0.75,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.3,1.4};
    }
    else if (alpha_bp == -5)
    {
        alphas = {0.5,0.55,0.6,0.65,0.7,0.75,0.85,0.9,0.95,1.0};
    }
    else if (alpha_bp == -6)
    {
        alphas = {0.8,1.0,1.2};
    }
    else
    {
        alphas = {alpha_bp};
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
    int avg_iter = 0;

    std::cout << "alpha\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tbp_iter\tber" << std::endl;
    OUTPUT_FILE << "alpha\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tbp_iter\tber" << std::endl;
    // initial word.
    xt::xarray<int> x = xt::zeros<int>({n_q});
    // std::cout << "x = " << x << std::endl;

    for (size_t i_alpha = 0; i_alpha < alphas.size(); i_alpha++)
    {
        long double alpha = alphas(i_alpha);
        ch_sc = 0;
        dec_sci = 0;
        dec_sce = 0;
        dec_ler = 0;
        dec_fail = 0;
        dec_gbp_sci = 0;
        dec_gbp_sce = 0;
        dec_gbp_ler = 0;
        dec_gbp_fail = 0;
        avg_iter = 0;

        xt::xarray<long double> p_initial = {1 - p_error, p_error / 3.0, p_error / 3.0, p_error / 3.0};
        if (p_initial_bp == -1)
        {
            bpDecoder.initialize_bp(p_initial, max_iter);
        }
        else
        {
            xt::xarray<long double> p_initial_for_decoder = {1 - p_initial_bp, p_initial_bp / 3.0, p_initial_bp / 3.0, p_initial_bp / 3.0};
            bpDecoder.initialize_bp(p_initial_for_decoder, max_iter);
        }
        

        for (size_t i_e = 0; i_e < n_errorsamples; i_e++)
        {
            xt::xarray<int> y = x;
            if (channel == "depolarizing")
            {
                depolarizing_channel(&y, p_error,&random_generator);
            }
            else if (channel == "custom")
            {
                y(0) = 1;
                y(1) = 1;
                y(2) = 1;
                
                // y(0) = 1;
                // y(8) = 1;
                // y(31) = 1;
                // y(6) = 1;
                // y(11) = 1;
                // y(15) = 1;

                // y(0) = 1;
                // y(1) = 1;
                // y(7) = 1;
                // y(14) = 1;

                
                // y(328) = 2;
                // y(331) = 2;
                // y(334) = 2;
                // y(337) = 2;
                
            }
            else if (channel == "const_weight")
            {
                int weight = (int)(p_error * n_q);
                const_weight_error_channel(&y, weight,&random_generator);
            }
            else if (channel == "const_weight_restricted")
            {
                int weight = (int)(p_error * n_q);
                const_weight_error_channel(&y, weight, 16, 1, &random_generator);
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
                xt::xarray<int> s_0 = gf4_syndrome(&y, &H);
                xt::xarray<int> error_guess = bpDecoder.decode_bp(s_0, w_bp, alpha, type_bp, return_if_success, only_non_converged);
                // xt::xarray<int> error_guess = xt::zeros<int>({n_q});
                xt::xarray<int> residual_error = error_guess ^ y;

                if (print_bp_details)
                {
                    print_container(y,"y",true);
                    print_container(s_0,"s_0",true);
                    print_container(error_guess,"error_guess",true);
                }

                // std::cout << "error_guess = " << error_guess << std::endl;
                if (get_marg_mess == true)
                {
                    xt::xarray<long double> marginals = bpDecoder.get_marginals();
                    xt::xarray<long double> messages = bpDecoder.get_messages();
                    xt::xarray<int> hard_decisions = bpDecoder.get_hard_decisions();
                    xt::xarray<int> syndromes = bpDecoder.get_syndromes();
                    xt::xarray<long double> free_energy = bpDecoder.get_free_energy();

                    xt::dump_npy(OUTPUT_DIR + "/marginals_bp.npy", marginals);
                    xt::dump_npy(OUTPUT_DIR + "/messages_bp.npy", messages);
                    xt::dump_npy(OUTPUT_DIR + "/hard_decisions_bp.npy", hard_decisions);
                    xt::dump_npy(OUTPUT_DIR + "/syndromes_bp.npy", syndromes);
                    xt::dump_npy(OUTPUT_DIR+"/free_energy_bp.npy",free_energy);
                }

                bool done = false;
                bool did_gbp_pp = false;

                xt::xarray<int> s = gf4_syndrome(&error_guess, &H);
                if (s == s_0)
                {
                    done = true;
                    avg_iter += bpDecoder.took_iterations;
                }
                else if (gbp_pp == true)
                {
                     int max_iter_gbp = 50;

                            // long double new_p = (long double)n_ci/(long double)n_c;
                            // long double new_p = 0.15;
                            // std::cout << "new_p = " << new_p << "\n";

                            // xt::xarray<long double> p_initial_gf2 = {1 - new_p, new_p};
                    xt::xarray<int> residual_s = s^s_0;
                // xt::xarray<int> from_gbp = gbpPP(residual_s,H_X,H_Z,&bpDecoder,n_checks_per_r0, w_for_gbp, type_gbp, return_if_success, print_gbp_details, get_marg_mess, OUTPUT_DIR,surface);
                    int took_iterations = 0;
                    int n_checks_per_r0 = 2;
                    bool surface = false;
                    xt::xarray<int> from_gbp = gbpPP(residual_s,H_X,H_Z,&bpDecoder, p_initial,n_checks_per_r0, w_gbp, type_gbp, split_strategy, 0, return_if_success, print_gbp_details, &took_iterations, get_marg_mess, OUTPUT_DIR, surface);
                    did_gbp_pp = true;
                    // xt::xarray<int> from_gbp = xt::zeros<int>({n_q});
                    // xt::xarray<int> residual_error_sub;

                    // xt::xarray<int> residual_s = gf4_syndrome(&residual_error, &H);

                    // xt::xarray<int> error_guess_x = get_x(error_guess);
                    // xt::xarray<int> error_guess_z = get_z(error_guess);
                    // xt::xarray<int> s_eg_x = gf2_syndrome(&error_guess_x, &H_Z);
                    // xt::xarray<int> s_eg_z = gf2_syndrome(&error_guess_z, &H_X);

                    // xt::xarray<int> residual_error_x = get_x(residual_error);
                    // xt::xarray<int> residual_error_z = get_z(residual_error);

                    // xt::xarray<int> residual_s_x = gf2_syndrome(&residual_error_x, &H_Z);
                    // xt::xarray<int> residual_s_z = gf2_syndrome(&residual_error_z, &H_X);

                    // if (print_gbp_details == true)
                    // {
                    //     std::cout << "\n***** gbp: start *****\n";
                    //     print_container(y, "bp: y", true);
                    //     print_container(s_0, "bp: s_0", true);

                    //     print_container(error_guess_x, "bp: error_guess_x", true);
                    //     print_container(error_guess_z, "bp: error_guess_z", true);

                    //     print_container(s_eg_x, "bp: s_eg_x", true);
                    //     print_container(s_eg_z, "bp: s_eg_z", true);

                    //     print_container(residual_error_x, "bp: residual_error_x", true);
                    //     print_container(residual_error_z, "bp: residual_error_z", true);

                    //     print_container(residual_s_x, "bp: residual_s_x", true);
                    //     print_container(residual_s_z, "bp: residual_s_z", true);
                    // }
                    // for (int pauli = 1; pauli <= 2; pauli++)
                    // {
                    //     if ((pauli == 1) && (xt::sum(residual_s_x)() == 0))
                    //     {
                    //         pauli++;
                    //     }
                    //     if ((pauli == 2) && (xt::sum(residual_s_z)() == 0))
                    //     {
                    //         break;
                    //     }
                    //     xt::xarray<int> s_sub_0;
                    //     xt::xarray<int> qi;
                    //     xt::xarray<int> ci;

                    //     xt::xarray<int> H_sub = get_H_sub(&bpDecoder, H, residual_s, &s_sub_0, &ci, &qi, 3 - pauli,print_gbp_details);

                    //     int n_ci = H_sub.shape(0);
                    //     int n_qi = H_sub.shape(1);

                    //     if (print_gbp_details == true)
                    //     {
                    //         std::cout << "H_sub: (" << n_ci << "," << n_qi << ")\n";// << H_sub << std::endl;
                    //         std::cout << "s_sub_0 = \n" << s_sub_0 << std::endl;
                    //     }

                        
                        
                    //     if (n_qi < 50)
                    //     {

                    //         int max_iter_gbp = 50;

                    //         long double new_p = (long double)n_ci/(long double)n_c;
                    //         // long double new_p = 0.15;
                    //         // std::cout << "new_p = " << new_p << "\n";

                    //         xt::xarray<long double> p_initial_gf2 = {1 - new_p, new_p};

                    //         GbpDecoder gbpDecoder(H_sub, max_iter_gbp, max_checks_per_r0);
                    //         std::cout << "constructed decoder\n";

                    //         // GbpDecoder2 gbpDecoder2(H_sub, max_iter_gbp,false);

                    //         xt::xarray<int> error_guess_sub = xt::zeros<int>({n_qi});

                    //         error_guess_sub = gbpDecoder.decode(s_sub_0, p_initial_gf2, type_gbp, w_gbp, return_if_success);
                    //         if (get_marg_mess == true)
                    //             {
                    //                 xt::xarray<long double> marginals = gbpDecoder.get_marginals();
                    //                 xt::xarray<long double> messages = gbpDecoder.get_messages();
                    //                 xt::xarray<int> hard_decisions = gbpDecoder.get_hard_decisions();
                    //                 // xt::xarray<int> syndromes = gbpDecoder2.get_syndromes();

                    //                 xt::dump_npy(OUTPUT_DIR + "/marginals.npy", marginals);
                    //                 xt::dump_npy(OUTPUT_DIR + "/messages.npy", messages);
                    //                 xt::dump_npy(OUTPUT_DIR + "/hard_decisions.npy", hard_decisions);
                    //                 // xt::dump_npy(OUTPUT_DIR+"/syndromes.npy",syndromes);
                    //             }
                    //         xt::xarray<int> s_sub = gf2_syndrome(&error_guess_sub, &H_sub);

                    //         int trys_p_init = 0;
                    //         std::vector<long double> p_rep = {new_p, 0.15, 0.2, 0.25, 0.3, 0.4};
                    //         int max_trys_p_init = p_rep.size();

                    //         for (int i = 0; i < n_qi; i++)
                    //         {
                    //             if (error_guess_sub(i) == 1)
                    //             {
                    //                 from_gbp(qi(i)) = pauli;
                    //             }
                    //         }
                    //         // print_container(from_gbp, "from_gbp", true);
                    //         // print_container(error_guess, "error_guess b/f", true);
                    //     }
                    // }
                    // error_guess ^= from_gbp;
                    // if (print_gbp_details == true)
                    // {
                    //     print_container(error_guess, "error_guess a/f gbp", true);
                    // }
                    
                }
                residual_error = y ^ error_guess;

                //    print_container(y,"y",true);
                //    print_container(residual_error,"residual_error",true);
                xt::xarray<int> residual_s = gf4_syndrome(&residual_error, &H);
                //    print_container(residual_s,"residual_s",true);

                gf4_syndrome(&s, &error_guess, &H);
                if (s == s_0)
                {
                    if (residual_error == x)
                    {
                        dec_sci++;
                        if (gbp_pp && did_gbp_pp)
                        {
                            dec_sci--;
                            dec_gbp_sci++;
                            if (print_gbp_details) std::cout << "***** gbp_pp: sci *****\n";
                        } 
                    }
                    else
                    {
                        if (gf4_isEquiv(residual_error, H, n_c, n_q))
                        {
                            dec_sce++;
                            if (gbp_pp && did_gbp_pp)
                            {
                                dec_sce--;
                                dec_gbp_sce++;
                                if (print_gbp_details) std::cout << "***** gbp_pp: sce *****\n";
                            }
                        }
                        else
                        {
                            dec_ler++;
                            if (gbp_pp && did_gbp_pp)
                            {
                                dec_ler--;
                                dec_gbp_ler++;
                                if (print_gbp_details) std::cout << "***** gbp_pp: ler *****\n";
                            }
                        }
                    }
                }
                else
                {
                    dec_fail++;
                    avg_iter += bpDecoder.took_iterations;
                    if (gbp_pp && did_gbp_pp) 
                    {
                        dec_fail--;
                        dec_gbp_fail++;
                        if (print_gbp_details) std::cout << "***** gbp_pp: fail *****\n";
                    }
                    if (print_fails == true)
                    {
                        std::cout << "XX fail XX";
                        print_container(y, "y", true);
                    }
                }
            }
        }
        long double ber = (long double)(dec_ler + dec_fail) / (long double)n_errorsamples;
        long double avg_bp_iterations = (long double)avg_iter / (long double)(n_errorsamples - ch_sc);
        if (print_gbp_details) std::cout << "alpha\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tbp_iter\tber" << std::endl;
        std::cout << alpha << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << avg_bp_iterations  << "\t" << ber << std::endl;
        OUTPUT_FILE << alpha << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << avg_bp_iterations  << "\t" << ber << std::endl;
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
