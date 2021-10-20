#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <experimental/filesystem>

#include "omp.h"

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
#include <bp_gbp/timing.hpp>

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

    std::string pathToCodes = json_input.at("path_to_codes");
    std::string code = json_input.at("code");

    std::string code_type = strip_(code);
    bool surface = false;
    if (code_type == "surface") surface = true;

    std::string pathToH_X = pathToCodes + code + "_hx.npy";
    std::string pathToH_Z = pathToCodes + code + "_hz.npy";

    long double p = json_input.at("p_error");
    long double bias_x = json_input.value("bias_x",1.0);
    long double bias_y = json_input.value("bias_y",1.0);
    long double bias_z = json_input.value("bias_z",1.0);
    std::string channel = json_input.at("channel");
    int n_errorsamples = json_input.at("n_errorsamples");

    bool decode_bp = json_input.value("decode_bp",true);
    int type_bp = json_input.at("type_bp");
    long double p_initial_bp = json_input.at("p_initial_bp");
    long double w_bp = json_input.at("w_bp");
    long double alpha_bp = json_input.at("alpha_bp");

    bool gbp_pp = json_input.value("gbp_pp",false);
    int type_gbp = json_input.at("type_gbp");
    long double p_initial_gbp = json_input.at("p_initial_gbp");
    long double w_gbp = json_input.at("w_gbp");
    int min_checks_per_r0 = json_input.value("min_checks_per_r0",1);
    int max_checks_per_r0 = json_input.value("max_checks_per_r0",1);

    bool repeat_split = json_input.value("repeat_split",false);
    int split_strategy = json_input.value("split_strategy",0);
    int n_split = json_input.value("n_split",10);

    bool get_marg_mess = json_input.value("get_marg_mess",false);
    bool print_fails = json_input.value("print_fails",false);
    bool print_bp_details = json_input.value("print_bp_details",false);
    bool print_gbp_details = json_input.value("print_gbp_details",false);
    bool print_gbp_status = json_input.value("print_gbp_status",false);
    

    bool return_if_success = json_input.value("return_if_success",true);
    bool only_non_converged = json_input.value("only_non_converged",true);

    int max_iter = 100;
    std::cout << "max_iter = " << max_iter << "\n";

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



    // make error probability 
    xt::xarray<long double> ps;
    if (p == -1) ps = {0.001,0.0025,0.005,0.0075,0.01,0.02,0.03,0.04};
    else if (p == -2) ps = {0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12};
    else if (p == -3) ps = xt::arange<long double>(0.01,0.5,0.001);
    else if (p == -4) ps = {0.15,0.153,0.156,0.159,0.162,0.165,0.168,0.171};
    else if (p == -5) ps = {7.0/n_q,8.0/n_q};
    else if (p == -6) ps = {0.001,0.003,0.007,0.01,0.02,0.03,0.04,0.05};
    else if (p == -7) ps = {0.001, 0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14};
    else if (p == -8) ps = xt::arange<long double>(0.09,0.101,0.003);
    else if (p == -9) ps = {0.12,0.1225,0.125,0.1275,0.13,0.1325,0.135,0.1375};
    // else if (p == -10) ps = xt::arange<long double>(0.128,0.141,0.003);
    else if (p == -11) ps = xt::arange<long double>(0.0,0.5,0.05);
    else ps = {p};


    // int ch_sc = 0;
    // int dec_sci = 0;
    // int dec_sce = 0;
    // int dec_ler = 0;
    // int dec_fail = 0;
    // int dec_gbp_sci = 0;
    // int dec_gbp_sce = 0;
    // int dec_gbp_ler = 0;
    // int dec_gbp_fail = 0;
    // int avg_iter = 0;
    // int avg_gbp_iter = 0;
    // int avg_gbp_checks_per_r0 = 0;
    // int gbp_repeatsplit  = 0;

    std::cout << "p\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tbp_iter\tgbp_it\tgbp_cpr\tgpb_rs\tber" << std::endl;
    OUTPUT_FILE << "p\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tbp_iter\tgbp_it\tgbp_cpr\tgpb_rs\tber" << std::endl;
    // initial word.
    xt::xarray<int> x = xt::zeros<int>({n_q});
    // std::cout << "x = " << x << std::endl;
    #pragma omp parallel for
    for (size_t i_p = 0; i_p < ps.size(); i_p++)
    {
        // construct channel
        NoisyChannel noisyChannel;

        // construct BpDecoder
        BpDecoderKL bpDecoder(H);

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
        int avg_gbp_iter = 0;
        int avg_gbp_checks_per_r0 = 0;
        int gbp_repeatsplit  = 0;

        long double p_error = ps(i_p);
        xt::xarray<long double> p_initial;
        if (channel == "x")
        {
            p_initial = {1 - p_error, p_error,0,0};
        }
        else if (channel == "xz")
        {
            long double p_xz = 1-sqrt(1-p_error);
            long double p_x = p_xz*(1-p_xz);
            long double p_y = p_xz*p_xz;
            long double p_z = (1-p_xz)*p_xz;
            p_initial = {1 - p_error, p_x,p_z,p_y};
        }
        else if ((channel == "biased") || (channel == "biased_erasure"))
        {
            long double p_x = p_error * bias_x;
            long double p_y = p_error * bias_y;
            long double p_z = p_error * bias_z;
            long double norm = 1.0/(bias_x+bias_y+bias_z);
            p_x *= norm; p_y *= norm; p_z *= norm;
            p_initial = {1 - p_error, p_x,p_z,p_y};
        }
        else
        {
            p_initial = {1 - p_error, p_error / 3.0, p_error / 3.0, p_error / 3.0};
        }

        xt::xarray<long double> p_initial_for_bp;
        xt::xarray<long double> p_initial_for_gbp;
        if (p_initial_bp == -1) p_initial_for_bp = p_initial;
        else  p_initial_for_bp = {1 - p_initial_bp, p_initial_bp / 3.0, p_initial_bp / 3.0, p_initial_bp / 3.0};

        bpDecoder.initialize_bp(p_initial_for_bp, max_iter);

        if (p_initial_gbp == -1) p_initial_for_gbp = p_initial;
        else
        {
            if (channel == "x")
            {
                 p_initial_for_gbp = {1 - p_initial_gbp, p_initial_gbp, 0, 0};
            }
            else
            {
                 p_initial_for_gbp = {1 - p_initial_gbp, p_initial_gbp / 3.0, p_initial_gbp / 3.0, p_initial_gbp / 3.0};
            }
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
            else if (channel == "x")
            {
                // x_channel(&y, p_error,&random_generator);
                noisyChannel.send_through_pauli_channel(&y,p_error,2);
            }
            else if (channel == "biased")
            {
                noisyChannel.send_through_biased_channel(&y,p_initial);
            }
            else if (channel == "custom")
            {
                y(35) = 1;
                y(23) = 1;
                y(24) = 1;
            }
            else if (channel == "custom2")
            {
                xt::xarray<int> erasures = xt::zeros<int>({n_q});

                // y(10) = 1;  erasures(10) = 1;
                // y(33) = 1; erasures(33) = 1;
                // y(37) = 1; erasures(37) = 1;
                // y(20) = 1; erasures(20) = 1;
                // y(21) = 2; erasures(21) = 1;
                // y(38) = 1; erasures(38) = 1;

                y(8) = 2; erasures(8) = 1;
                y(10) = 1; erasures(10) = 1;
                y(18) = 2; erasures(18) = 1;
                y(19) = 2; erasures(19) = 1;

                bpDecoder.initialize_erasures(&erasures);

            }
            else if (channel == "custom3")
            {
                // y(0) = 2;
                y(1) = 2;
                y(13) = 2;

                y(14) = 3;//
                y(19) = 3;//
                y(26) = 3;
                y(27) = 3;
                y(33) = 3;//
                y(35) = 2;
                // y(12) = 2;
            }
            else if (channel == "custom4")
            {
                y(4) = 3;
                y(14) = 3;
                y(32) = 1;
            }
            else if (channel == "custom5")
            {
                y(0) = 2;
                y(4) = 2;
                // y(10) = 2;
            }
            else if (channel == "custom6")
            {
                y(3) = 2;
                y(5) = 2;
                y(14) = 2;
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
                int size_classical = (int) (sqrt(n_q-n_c) + sqrt(n_q+n_c))/2;

                noisyChannel.const_weight_error_channel(&y, weight, size_classical, 2);
            }
            else if (channel == "const_weight_restricted2")
            {
                int weight = (int)(p_error * n_q);
                // const_weight_error_channel(&y, weight, 16, 1, &random_generator);
                
                int size_classical = (int) (sqrt(n_q-n_c) + sqrt(n_q+n_c))/2;
                int size_classical_T = (int) (sqrt(n_q+n_c) - sqrt(n_q-n_c))/2;

                int base_qubit =0;//size_classical * size_classical;



                noisyChannel.const_weight_error_channel_T(&y, weight, base_qubit, size_classical, 2);
            }
            else if (channel == "erasure")
            {
                xt::xarray<int> erasures = xt::zeros<int>({n_q});
                noisyChannel.erasure_channel(&y, &erasures,p_error);
                bpDecoder.initialize_erasures(&erasures);
            }
            else if (channel == "biased_erasure") //|| (channel == "custom2"))
            {
                xt::xarray<int> erasures = xt::zeros<int>({n_q});
                noisyChannel.biased_erasure_channel(&y, &erasures,p_initial);
                bpDecoder.initialize_erasures(&erasures);

            }
            else
            {
                std::cerr << "Not a valid channel, choose frome {'depolarizing', 'xz', 'x', 'custom', 'const_weight', 'const_weight_restricted', 'erasure'}" << std::endl;
                // return 1;
            }
            if (y == x)
            {
                ch_sc++;
            }
            else
            {
                xt::xarray<int> s_0 = gf4_syndrome(&y, &H);
                xt::xarray<int> error_guess = xt::zeros<int>({n_q});
                if (decode_bp)
                    error_guess =  bpDecoder.decode_bp(s_0, w_bp, alpha_bp, type_bp, return_if_success, only_non_converged);
                xt::xarray<int> error_guess_bp = error_guess;
                xt::xarray<int> residual_error = error_guess ^ y;

                if (decode_bp && print_bp_details)
                {
                    print_container(y,"y",true);
                    print_container(s_0,"s_0",true);
                    print_container(error_guess,"error_guess",true);
                }

                if (decode_bp && get_marg_mess == true)
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
                int n_checks_per_r0 = min_checks_per_r0;

                xt::xarray<int> s = gf4_syndrome(&error_guess, &H);
                if (s == s_0)
                {
                    done = true;
                    avg_iter += bpDecoder.took_iterations;
                }
                else if (gbp_pp == true)
                {
                    if (print_fails == true)
                    {
                        std::cout << "XX fail XX\n";
                        print_container(y, "y", true);
                        print_container(error_guess, "error_guess", true);
                        print_container(residual_error, "residual_error", true);
                    }
                    
                    int r_p_i = 0;
                    int repetitions_split = 0;
                    // xt::xarray<long double> other_ws;

                    // int n_ws = other_ws.size();
                    // int i_w = 0;
                    while (done == false)
                    {
                        // long double w_for_gbp = other_ws(i_w);
                        did_gbp_pp = true;
                        xt::xarray<int> residual_s = s^s_0;
                        // xt::xarray<int> from_gbp = gbpPP(residual_s,H_X,H_Z,&bpDecoder,n_checks_per_r0, w_for_gbp, type_gbp, return_if_success, print_gbp_details, get_marg_mess, OUTPUT_DIR,surface);
                        int took_iterations = 0;
                        Timer t;
                        xt::xarray<int> from_gbp = gbpPP(residual_s,H_X,H_Z,&bpDecoder, p_initial_for_gbp,n_checks_per_r0, w_gbp, type_gbp, split_strategy, repetitions_split, return_if_success, print_gbp_details, &took_iterations, get_marg_mess, OUTPUT_DIR,surface);
                        if (print_gbp_details == true)
                        {
                            std::cout << "gbpPP: t = " << t.elapsed() << "s\n";
                        }
                        t.reset();
                        xt::xarray<int> new_error_guess = error_guess ^ from_gbp;
                        xt::xarray<int> new_s = gf4_syndrome(&new_error_guess, &H);
                        xt::xarray<int> new_res_s = s_0^new_s;
                        if (print_gbp_details == true)
                        {
                            std::cout << "| n_checks_per_r0 = " << n_checks_per_r0 << "\n";
                            std::cout << "| took_iterations = " << took_iterations << "\n";
                            std::cout << "| "; print_container(from_gbp, "from_gbp", true);
                            std::cout << "| "; print_container(new_error_guess, "error_guess a/f gbp", true);
                            std::cout << "| "; print_container(new_res_s, "new_res_s a/f gbp", true);
                            std::cout << "| "; print_container(residual_error, "residual_error b/f gbp", true);
                            xt::xarray<int> new_res_e = new_error_guess^y;
                            std::cout << "| "; print_container(new_res_e, "residual_error a/f gbp", true);
                            std::cout << "| repetitions_split = " << repetitions_split << "\n";
                            std::cout << "** gbp done **" << "\n";
                        }

                        if (new_s == s_0)
                        {
                            error_guess = new_error_guess;
                            avg_gbp_checks_per_r0 += n_checks_per_r0;
                            avg_gbp_iter += took_iterations;
                            gbp_repeatsplit  += repetitions_split;
                            done = true;
                        }

                        else if ((repeat_split) && (repetitions_split < n_split))
                        {
                            s = new_s;
                            error_guess = new_error_guess;
                            repetitions_split++;
                        }
                        else if ((n_checks_per_r0 < max_checks_per_r0)) // && (i_w == n_ws))
                        {
                            if (repeat_split)
                            {
                                repetitions_split = 0;
                                s = s_0;
                                error_guess = error_guess_bp;
                            }
                            n_checks_per_r0++;

                        }
                        else
                        {
                            done = true;
                        }
                    }
                    
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
                    if (print_fails == true)
                    {
                        std::cout << "XX fail XX\n";
                        print_container(y, "y", true);
                        print_container(error_guess, "error_guess", true);
                        print_container(residual_error, "residual_error", true);
                    }
                    if (gbp_pp && did_gbp_pp) 
                    {
                        dec_fail--;
                        dec_gbp_fail++;
                        if (print_gbp_details) std::cout << "***** gbp_pp: fail *****\n";
                    }
                }
                if (gbp_pp && did_gbp_pp && print_gbp_status)
                {
                    long double ber = (long double)(dec_ler + dec_fail + dec_gbp_ler + dec_gbp_fail) / (long double)(i_e+1);
                    int n_gbp_runs = dec_gbp_sci + dec_gbp_sce + dec_gbp_ler + dec_gbp_fail;
                    long double avg_bp_iterations = (long double)avg_iter / (long double)(i_e+1 - ch_sc - n_gbp_runs);
                    long double avg_gbp_iterations = (long double) avg_gbp_iter / (long double)(n_gbp_runs);
                    long double avg_gbp_checks = (long double)avg_gbp_checks_per_r0 / (long double)(n_gbp_runs);
                    long double avg_gbp_repeatsplit = (long double)(gbp_repeatsplit) / (long double)(n_gbp_runs);
                    std::cout << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << avg_bp_iterations << "\t" << avg_gbp_iterations << "\t" << avg_gbp_checks << "\t" << avg_gbp_repeatsplit << "\t" << ber << std::endl;
                    // OUTPUT_FILE << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << ber << std::endl;
                }
            }
            if (n_errorsamples >= 100)
            {
                if (i_e % (int)(n_errorsamples * 0.1) == 0)
                {
                    long double ber = (long double)(dec_ler + dec_fail) / (long double)(i_e+1);
                    if (gbp_pp)
                    {
                        ber = (long double)(dec_ler + dec_fail + dec_gbp_ler + dec_gbp_fail) / (long double)(i_e+1);
                    }
                    int n_gbp_runs = dec_gbp_sci + dec_gbp_sce + dec_gbp_ler + dec_gbp_fail;
                    long double avg_bp_iterations = (long double)avg_iter / (long double)(i_e+1 - ch_sc - n_gbp_runs);
                    long double avg_gbp_iterations = (long double) avg_gbp_iter / (long double)(n_gbp_runs);
                    long double avg_gbp_checks = (long double)avg_gbp_checks_per_r0 / (long double)(n_gbp_runs);
                    long double avg_gbp_repeatsplit = (long double)(gbp_repeatsplit) / (long double)(n_gbp_runs);
                    
                    
                    if (print_gbp_details) std::cout << "p\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tbp_iter\tgbp_it\tgbp_cpr\tgpb_rs\tber" << std::endl;
                    std::cout << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << avg_bp_iterations << "\t" << avg_gbp_iterations << "\t" << avg_gbp_checks << "\t" << avg_gbp_repeatsplit << "\t" << ber << std::endl;

                }
            }
            

        }
        long double ber = (long double)(dec_ler + dec_fail) / (long double)n_errorsamples;
        if (gbp_pp)
        {
            ber = (long double)(dec_ler + dec_fail + dec_gbp_ler + dec_gbp_fail) / (long double)(n_errorsamples);
        }
        int n_gbp_runs = dec_gbp_sci + dec_gbp_sce + dec_gbp_ler + dec_gbp_fail;
        long double avg_bp_iterations = (long double)avg_iter / (long double)(n_errorsamples - ch_sc - n_gbp_runs);
        long double avg_gbp_iterations = (long double) avg_gbp_iter / (long double)(n_gbp_runs);
        long double avg_gbp_checks = (long double)avg_gbp_checks_per_r0 / (long double)(n_gbp_runs);
        long double avg_gbp_repeatsplit = (long double)(gbp_repeatsplit) / (long double)(n_gbp_runs);
        
        
        if (print_gbp_details) std::cout << "p\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tbp_iter\tgbp_it\tgbp_cpr\tgpb_rs\tber" << std::endl;
        std::cout << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << avg_bp_iterations << "\t" << avg_gbp_iterations << "\t" << avg_gbp_checks << "\t" << avg_gbp_repeatsplit << "\t" << ber << std::endl;

        OUTPUT_FILE << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << avg_bp_iterations << "\t" << avg_gbp_iterations << "\t" << avg_gbp_checks << "\t" << avg_gbp_repeatsplit << "\t" << ber << std::endl;

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
