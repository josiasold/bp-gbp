#ifndef LA_TOOLS_HPP_
#define LA_TOOLS_HPP_

#include <algorithm> //min
#include <iostream>

#include <NTL/mat_GF2.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

void gf2_syndrome(xt::xarray<int> *s, xt::xarray<int> *y,xt::xarray<int> *H);
xt::xarray<int> gf2_syndrome(xt::xarray<int> *y,xt::xarray<int> *H);

void gf2_rank(int* Matrix, int n_c, int n_v, int* r);
int gf2_rank(int* Matrix, int n_c, int n_v);

bool gf2_isEquiv(xt::xarray<int> e, xt::xarray<int> H, int n_c, int n_v);

void gf4_syndrome(xt::xarray<int> *s, xt::xarray<int> *y,xt::xarray<int> *H);
xt::xarray<int> gf4_syndrome(xt::xarray<int> *y,xt::xarray<int> *H);

void gf4_rank(int* Matrix, int n_c, int n_v, int* r);
int gf4_rank(int* Matrix, int n_c, int n_v);

bool gf4_isEquiv(xt::xarray<int> e, xt::xarray<int> H, int n_c, int n_v);

int gf4_mul(int a, int b);
int gf4_conj(int a);

int hamming_weight(xt::xarray<int>x);

xt::xarray<int> get_x(xt::xarray<int> y);
xt::xarray<int> get_z(xt::xarray<int> y);

#endif