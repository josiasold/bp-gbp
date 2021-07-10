#ifndef DECODER_GBP_HPP_
#define DECODER_GBP_HPP_

#include <bp_gbp/constructor_gbp.hpp>
#include <bp_gbp/la_tools.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>

#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xframe/xvariable_view.hpp"

#include <cppitertools/product.hpp>

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::valarray<T> to_valarray(Tuple &&tuple)
{
  return std::experimental::apply([](auto &&...elems) {
    return std::valarray<T>{std::forward<decltype(elems)>(elems)...};
  },
                    std::forward<Tuple>(tuple));
} // https://stackoverflow.com/questions/42494715/c-transform-a-stdtuplea-a-a-to-a-stdvector-or-stddeque

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
        std::vector< xt::xarray<int> > hard_dec;

        RegionGraph RG;

        void fill_c_factors();
        void fill_q_factors(xt::xarray<long double> p_initial);
        void prepare_messages(const xt::xarray<int> *s_0);
        void prepare_beliefs(const xt::xarray<int> *s_0);
        void update_messages(const xt::xarray<int> *s_0, xt::xarray<long double> p_initial,long double w_gbp, int iteration);
        void update_beliefs(const xt::xarray<int> *s_0,int iteration);
        void overall_belief(int iteration);
        void get_marginals_and_hard_dec(int iteration);

        

    public:
        GbpDecoder(xt::xarray<int> H, int max_iterations, int n_checks_per_r0);

        xt::xarray<int> decode(xt::xarray<int> s_0,xt::xarray<long double> p_initial,int type_marg,long double w_gbp);
        xt::xarray<long double> get_messages();
        xt::xarray<long double> get_marginals();
        xt::xarray<int> get_hard_decisions();
        xt::xarray<bool> get_convergence();





};

#endif