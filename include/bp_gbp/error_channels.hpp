#ifndef ERROR_CHANNELS_HPP_
#define ERROR_CHANNELS_HPP_

#include <valarray>
#include <stdlib.h>
#include <random>
#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

void xz_channel(xt::xarray<int> *y, double p_error);

void x_channel(xt::xarray<int> *y, double p_error);

void depolarizing_channel(xt::xarray<int> *y, double p_error);

void const_weight_error_channel(xt::xarray<int> *y, int weight);
void const_weight_error_channel(xt::xarray<int> *y, int weight, int max_qubit, int n_paulis);

void erasure_channel(xt::xarray<int> *y,xt::xarray<int> *erasures, double p_error);


#endif