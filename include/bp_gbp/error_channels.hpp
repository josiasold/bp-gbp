#ifndef ERROR_CHANNELS_HPP_
#define ERROR_CHANNELS_HPP_

#include <valarray>
#include <stdlib.h>
#include <random>
#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>  // <<

class NoisyChannel
{
        private:
                std::random_device rd;
                std::mt19937 random_generator;

                void depolarizing_channel(xt::xarray<int> *y, double p_error);

                void xz_channel(xt::xarray<int> *y, double p_error);

                void x_channel(xt::xarray<int> *y, double p_error);

        
        public:
                NoisyChannel();

                void send_through_pauli_channel(xt::xarray<int> *y, double p_error, int type);

                void send_through_biased_channel(xt::xarray<int> *y,xt::xarray<double> p_initial);

                void erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, double p_error);

                void biased_erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, xt::xarray<double> p_initial);

                void const_weight_error_channel(xt::xarray<int> *y, int weight, int max_qubit, int pauli);

                void const_weight_error_channel_T(xt::xarray<int> *y, int weight,int base_qubit , int max_qubit, int pauli);

                void const_weight_error_channel(xt::xarray<int> *y, int weight);


};

void xz_channel(xt::xarray<int> *y, double p_error,std::mt19937 *random_generator);

void x_channel(xt::xarray<int> *y, double p_error,std::mt19937 *random_generator);

void depolarizing_channel(xt::xarray<int> *y, double p_error,std::mt19937 *random_generator);

void const_weight_error_channel(xt::xarray<int> *y, int weight, std::mt19937 *random_generator);

void const_weight_error_channel(xt::xarray<int> *y, int weight, int max_qubit, int n_paulis,std::mt19937 *random_generator);

void erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, double p_error, std::mt19937 *random_generator);


#endif