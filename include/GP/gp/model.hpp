#pragma once
#include <matrix/matrix.hpp>
#include <linalg/linalg.hpp>
#include <cmath>
#include <numeric>
#include <vector>

namespace GP{

template<class T>
class GPRegression{
private:
    matrix<T> C_inv_, X_, Y_;
    double gamma_;
    T beta_;
    matrix<T> rbf_kernel_(const matrix<T>&, const matrix<T>&);
public:
    GPRegression(double g = 0.1, T b = T{1}):
        C_inv_{}, X_{}, Y_{}, gamma_{g}, beta_{b} {}
    void fit(const matrix<T>& X, const matrix<T>& Y){
        using namespace GP::linalg;
        auto&& [xr, xc] = X.shape();
        auto&& [yr, yc] = Y.shape();
        if(xr != yr){
            throw typename matrix<T>::DimensionalityException();
        }
        X_ = X; Y_ = Y;
        C_inv_ = ~(rbf_kernel_(X, X) + identity<T>(xr) * beta_);
    }
    auto predict(const matrix<T>& X_test){
        using namespace GP::linalg;
        auto&& [xr, xc] = X_.shape();
        auto&& [xtest_r, xtest_c] = X_test.shape();
        if(xc != xtest_c){
            throw typename matrix<T>::DimensionalityException();
        }
        auto k = rbf_kernel_(X_, X_test);
        auto ktCinv = transpose<T>(k) ^ C_inv_;
        return std::pair<matrix<T>, matrix<T>>{
            ktCinv ^ Y_,
            rbf_kernel_(X_test, X_test) + identity<T>(xtest_r) * beta_ - (ktCinv ^ k)
        };
    }
};

template<class T>
matrix<T> GPRegression<T>::rbf_kernel_(const matrix<T>& X1, const matrix<T>& X2){
    using namespace GP::linalg;
    auto&& [n1, feats1] = X1.shape();
    auto&& [n2, feats2] = X2.shape();
    if(feats1 != feats2){
        throw typename matrix<T>::DimensionalityException();
    }
    matrix<T> kernel(n1, n2);
    for(size_t r = 0; r < n1; ++r){
        for(size_t c = 0; c < n2; ++c){
            std::vector<T> vec_dif(feats1);
            for(size_t k = 0; k < feats1; ++k)
                vec_dif[k] = X1(r, k) - X2(c, k);
            T dot_product = std::inner_product(
                vec_dif.begin(), vec_dif.end(), vec_dif.begin(), T{});
            kernel(r, c) = std::exp(-gamma_*dot_product);
        }
    }
    return kernel;
}

}