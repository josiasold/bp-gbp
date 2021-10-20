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

    std::string code_type = strip_(code);
    bool surface = false;
    if (code_type == "surface") surface = true;

    std::string pathToH_X = pathToCodes + code + "_hx.npy";
    std::string pathToH_Z = pathToCodes + code + "_hz.npy";

    bool gbp_pp = json_input.at("gbp_pp");
    bool repeat_p_init = json_input.value("repeat_p_init",false);
    bool repeat_split = json_input.at("repeat_split");
    int split_strategy = json_input.value("split_strategy",0);

    bool get_marg_mess = json_input.at("get_marg_mess");
    bool print_bp_details = json_input.at("print_bp_details");
    bool print_gbp_details = json_input.at("print_gbp_details");
    bool print_gbp_status = json_input.at("print_gbp_status");
    bool print_fails = json_input.at("print_fails");

    bool return_if_success = json_input.at("return_if_success");
    bool only_non_converged = json_input.at("only_non_converged");    std::string output_suffix = json_input.at("output_suffix");
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
    BpDecoderKLM bpDecoder(H);

    // error channel
    xt::xarray<long double> ps;
    if (p == -1)
    {
        // ps = {0.07,0.075,0.08,0.085,0.09,0.095,0.1};
        ps = {0.11,0.12,0.13,0.14};
    }
    else if (p == -2)
    {
        ps = {0.001,0.01};
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
        ps = {0.001, 0.01,0.02, 0.03,0.04, 0.05, 0.06,0.07,0.08,0.09,0.1,0.11};
    }
    else if (p == -5)
    {
        ps = {1.0/n_q,2.0/n_q,3.0/n_q,4.0/n_q,5.0/n_q,6.0/n_q,7.0/n_q,8.0/n_q};
    }
    else
    {
        ps = {p};
    }

    int max_iter = 150;

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
            else if (channel == "xz")
            {
                xz_channel(&y, p_error,&random_generator);
            }
            else if (channel == "x")
            {
                x_channel(&y, p_error,&random_generator);
            }
            else if (channel == "custom")
            {
                y(1) = 1;
                y(4) = 3;
                y(5) = 2;
                y(13) = 3;
                y(20) = 3;
                y(38) = 1;
                y(39) = 1;
                y(46) = 1;
                y(107) = 2;
                y(114) = 1;
                y(117) = 3;
                y(121) = 2;
                y(130) = 1;
            }
            else if (channel == "custom2")
            {
                y(37) = 2;
                y(69) = 2;
                y(128) = 1;
                y(129) = 1;
                y(134) = 1;
                y(139) = 1;
                y(140) = 1;
                y(172) = 2;
                y(193) = 3;
                y(194) = 3;
                y(195) = 1;
                y(197) = 3;
                y(201) = 1;
                y(203) = 3;
                y(204) = 3;
                y(205) = 1;
                y(207) = 1;
                y(208) = 2;
                y(222) = 2;
                y(236) = 2;
                y(240) = 2;
                y(260) = 2;
                y(261) = 2;
                y(273) = 2;
                y(309) = 2;
                y(368) = 2;
                y(380) = 2;
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
                xt::xarray<int> error_guess = bpDecoder.decode_bp(s_0, w_bp, alpha_bp, type_bp, return_if_success);
                xt::xarray<int> residual_error = error_guess ^ y;

                if (print_bp_details)
                {
                    print_container(y,"y",true);
                    print_container(s_0,"s_0",true);
                    print_container(error_guess,"error_guess",true);
                }

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
                int n_checks_per_r0;

                xt::xarray<int> s = gf4_syndrome(&error_guess, &H);
                if (s == s_0)
                {
                    done = true;
                }
                else if (gbp_pp == true)
                {
                    if (print_fails == true)
                    {
                        std::cout << "XX fail XX\n";
                        print_container(y, "y", true);
                        print_container(residual_error, "residual_error", true);
                    }
                    
                    int r_p_i = 0;
                    xt::xarray<long double> other_ws;
                    if (surface)
                    { 
                        other_ws = {1.2,1.1,1.3,1.4,1.5,1.05,1.25,1.15};
                        n_checks_per_r0 = 1;
                    }
                    else 
                    {
                        other_ws = {0.9,0.8,0.7};
                        n_checks_per_r0 = 2;
                    }
                    int n_ws = other_ws.size();
                    int i_w = 0;
                    while (done == false)
                    {
                        long double w_for_gbp = other_ws(i_w);
                        did_gbp_pp = true;
                        xt::xarray<int> residual_s = s^s_0;
                        int took_iterations;
                        xt::xarray<int> from_gbp = gbpPP(residual_s,H_X,H_Z,&bpDecoder, p_initial,n_checks_per_r0, w_for_gbp, type_gbp, split_strategy, 0, return_if_success, print_gbp_details, &took_iterations, get_marg_mess, OUTPUT_DIR,surface);

                        xt::xarray<int> new_error_guess = error_guess ^ from_gbp;
                        xt::xarray<int> new_s = gf4_syndrome(&new_error_guess, &H);
                        xt::xarray<int> new_res_s = s_0^new_s;
                        if (print_gbp_details == true)
                        {
                            print_container(from_gbp, "from_gbp", true);
                            print_container(new_error_guess, "error_guess a/f gbp", true);
                            print_container(new_res_s, "new_res_s a/f gbp", true);
                        }
                        if (new_s == s_0)
                        {
                            error_guess = new_error_guess;
                            done = true;
                        }
                        else if (i_w < n_ws)
                        {
                            i_w++;
                        }
                        else if ((n_checks_per_r0 < max_checks_per_r0) && (i_w == n_ws))
                        {
                            n_checks_per_r0++;
                            i_w = 0;
                        }
                        // else if ((repeat_p_init) && (r_p_i < 5))
                        // {
                        //     double o_p = other_ps(r_p_i);
                        //     xt::xarray<long double> other_p = {1 - o_p, o_p / 3.0, o_p / 3.0, o_p / 3.0};
                        //     bpDecoder.initialize_bp(other_p,max_iter);
                        //     // error_guess = new_error_guess;
                        //     r_p_i++;
                        // }
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
                    if (gbp_pp && did_gbp_pp) 
                    {
                        dec_fail--;
                        dec_gbp_fail++;
                        if (print_gbp_details) std::cout << "***** gbp_pp: fail *****\n";
                    }
                }
                if (gbp_pp && did_gbp_pp && print_gbp_status)
                {
                    long double ber = (long double)(dec_ler + dec_fail) / (long double)n_errorsamples;
                    std::cout << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << ber << std::endl;
                    // OUTPUT_FILE << p_error << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << ber << std::endl;
                }
            }
            // if (n_errorsamples >= 1000)
            // {
            //     if (i_e % (int)(n_errorsamples * 0.05) == 0)
            //     {
            //         long double ber = (long double)(dec_ler + dec_fail) / (long double)i_e;
            //         if (print_gbp_details) std::cout << "p\ti_e\tch_sc\tbp_sci\tbp_sce\tbp_ler\tbp_fail\tgbp_sci\tgbp_sce\tgbp_ler\tgb_fail\tber" << std::endl;
            //         std::cout << p_error << "\t" << i_e << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << ber << std::endl;
            //         OUTPUT_FILE << p_error<< "\t" << i_e << "\t" << ch_sc << "\t" << dec_sci << "\t" << dec_sce << "\t" << dec_ler << "\t" << dec_fail << "\t" << dec_gbp_sci << "\t" << dec_gbp_sce << "\t" << dec_gbp_ler << "\t" << dec_gbp_fail << "\t" << ber << std::endl;
            //     }
            // }
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
