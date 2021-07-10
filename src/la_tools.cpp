#include <bp_gbp/la_tools.hpp>

void gf2_syndrome(xt::xarray<int> *s, xt::xarray<int> *y,xt::xarray<int> *H)
{
  *s = xt::linalg::dot(*H, *y) % 2;
}

xt::xarray<int> gf2_syndrome(xt::xarray<int> *y,xt::xarray<int> *H)
{
  xt::xarray<int> s = xt::linalg::dot(*H, *y) % 2;
  return s;
}

void gf2_rank(int* Matrix, int n_c, int n_q, int* r)
{
  NTL::Mat<NTL::GF2> M;
  M.SetDims(n_c,n_q);
  for (long i = 0; i < n_c; i++) 
  {
    for (long j = 0; j < n_q; j++) 
    {
      M[i][j] = Matrix[i*n_q+j];
    }
  }
  *r = NTL::gauss(M);
}

int gf2_rank(int* Matrix, int n_c, int n_q)
{
  NTL::Mat<NTL::GF2> M;
  M.SetDims(n_c,n_q);
  for (long i = 0; i < n_c; i++) 
  {
    for (long j = 0; j < n_q; j++) 
    {
      M[i][j] = Matrix[i*n_q+j];
    }
  }
  int r = NTL::gauss(M);
  return r;
}

bool gf2_isEquiv(xt::xarray<int> e, xt::xarray<int> H, int n_c, int n_q)
{
  int rank_init = 0;
  gf2_rank(H.data(), n_c, n_q, &rank_init);

  xt::xarray<int> H_larger = xt::zeros_like(H);
  H_larger.resize({static_cast<unsigned long>(n_c + 1), static_cast<unsigned long>(n_q)});
  xt::view(H_larger, xt::range(0, n_c), xt::all()) = H;
  xt::row(H_larger, n_c) = e;

  int rank_after = 0;
  gf2_rank(H_larger.data(), n_c + 1, n_q, &rank_after);
  // std::cout << "H = " << std::endl << H << std::endl  << "H_arger = " << std::endl << H_larger << std::endl;
  // std::cout << "rank_init = " << rank_init <<  std::endl << "rank_after = " << rank_after << std::endl;
  if (rank_after == rank_init)
    return true;
  else
    return false;
}

void gf4_syndrome(xt::xarray<int> *s, xt::xarray<int> *y,xt::xarray<int> *H)
{
  int mul_table[16] = {0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 3, 1, 0, 3, 1, 2};
  int conj_table[4] = {0, 1, 3, 2};
  int trace_table[4] = {0, 0, 1, 1};
  int stride = 4;

  int n_c = H->shape(0);
  int n_q = H->shape(1);

  for (size_t c = 0; c < n_c; c++) 
  {
    int s_value = 0;
    for (size_t q = 0; q < n_q; q++) 
    {
        s_value ^= mul_table[H->at(c,q) * stride + conj_table[y->at(q)]];
    }
    s->at(c) = trace_table[s_value];
  }
}

xt::xarray<int> gf4_syndrome(xt::xarray<int> *y,xt::xarray<int> *H)
{
  int mul_table[16] = {0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 3, 1, 0, 3, 1, 2};
  int conj_table[4] = {0, 1, 3, 2};
  int trace_table[4] = {0, 0, 1, 1};
  int stride = 4;

  int n_c = H->shape(0);
  int n_q = H->shape(1);

  xt::xarray<int> s = xt::zeros<int>({n_c});

  for (size_t c = 0; c < n_c; c++) 
  {
    int s_value = 0;
    for (size_t q = 0; q < n_q; q++) 
    {
        s_value ^= mul_table[H->at(c,q) * stride + conj_table[y->at(q)]];
    }
    s(c) = trace_table[s_value];
  }
  return s;
}

void gf4_rank(int* Matrix, int n_c, int n_q, int* r)
{
  NTL::Mat<NTL::GF2> M_X;
  M_X.SetDims(n_c,n_q);

  NTL::Mat<NTL::GF2> M_Z;
  M_Z.SetDims(n_c,n_q);

  for (long i = 0; i < n_c; i++) 
  {
    for (long j = 0; j < n_q; j++) 
    {
      if (Matrix[i*n_q+j] == 1) 
      {
        M_X[i][j] = 1;
        M_Z[i][j] = 0;
      }
      else if (Matrix[i*n_q+j] == 2) 
      {
        M_X[i][j] = 0;
        M_Z[i][j] = 1;
      }
      else if (Matrix[i*n_q+j] == 3) 
      {
        M_X[i][j] = 1;
        M_Z[i][j] = 1;
      }
      else
      {
        M_X[i][j] = 0;
        M_Z[i][j] = 0;
      }
    }
  }

  int r_X = NTL::gauss(M_X);
  int r_Z = NTL::gauss(M_Z);

  *r = r_X + r_Z;
}

int gf4_rank(int* Matrix, int n_c, int n_q)
{
  NTL::Mat<NTL::GF2> M_X;
  M_X.SetDims(n_c,n_q);

  NTL::Mat<NTL::GF2> M_Z;
  M_Z.SetDims(n_c,n_q);

  for (long i = 0; i < n_c; i++) 
  {
    for (long j = 0; j < n_q; j++) 
    {
      if (Matrix[i*n_q+j] == 1) 
      {
        M_X[i][j] = 1;
        M_Z[i][j] = 0;
      }
      else if (Matrix[i*n_q+j] == 2) 
      {
        M_X[i][j] = 0;
        M_Z[i][j] = 1;
      }
      else if (Matrix[i*n_q+j] == 3) 
      {
        M_X[i][j] = 1;
        M_Z[i][j] = 1;
      }
      else
      {
        M_X[i][j] = 0;
        M_Z[i][j] = 0;
      }
    }
  }

  int r_X = NTL::gauss(M_X);
  int r_Z = NTL::gauss(M_Z);

  int r = r_X + r_Z;
  return r;
}

bool gf4_isEquiv(xt::xarray<int> e, xt::xarray<int> H, int n_c, int n_q)
{
  int rank_init = 0;
  gf4_rank(H.data(), n_c, n_q, &rank_init);

  xt::xarray<int> H_larger = xt::zeros_like(H);
  H_larger.resize({static_cast<unsigned long>(n_c + 1), static_cast<unsigned long>(n_q)});
  xt::view(H_larger, xt::range(0, n_c), xt::all()) = H;
  xt::row(H_larger, n_c) = e;

  int rank_after = 0;
  gf4_rank(H_larger.data(), n_c + 1, n_q, &rank_after);
  // std::cout << "H = " << std::endl << H << std::endl  << "H_arger = " << std::endl << H_larger << std::endl;
  // std::cout << "rank_init = " << rank_init <<  std::endl << "rank_after = " << rank_after << std::endl;
  if (rank_after == rank_init)
    return true;
  else
    return false;
}

int gf4_conj(int a)
{
  xt::xarray<int> conj_table = {0, 1, 3, 2};
  return conj_table(a);
}

int gf4_mul(int a,int b)
{
  xt::xarray<int> mul_table = {{0, 0, 0, 0}, {0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}};
  return mul_table(a,b);
}

int hamming_weight(xt::xarray<int>x)
{
  int hw = 0;
  for (auto it = x.begin(); it != x.end(); ++it)
  {
    if (*it != 0) hw++;
  }
  return hw;
}

xt::xarray<int> get_x(xt::xarray<int> y)
{
  auto x = xt::zeros_like(y);
  for (int i = 0; i<y.size(); i++)
  {
    if ((y(i) == 1) | (y(i) == 3))
    {
      x(i) = 1;
    }
  }
  return x;
}

xt::xarray<int> get_z(xt::xarray<int> y)
{
  auto z = xt::zeros_like(y);
  for (int i = 0; i<y.size(); i++)
  {
    if ((y(i) == 2) | (y(i) == 3))
    {
      z(i) = 1;
    }
  }
  return z;
}