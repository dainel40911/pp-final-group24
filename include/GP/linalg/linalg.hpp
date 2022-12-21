#pragma once
#include <matrix/matrix.hpp>
#include <random>
namespace GP{

namespace linalg{

const size_t CACHE_LINESIZE = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);


template<class T>
matrix<T> inv_impl(matrix<T>& mat){
    // implement Gauss-Jordan
    auto&& [row_, col_] = mat.shape();
    if(row_ != col_){
        throw typename matrix<T>::DimensionalityException();
    }
    size_t n = row_;
    auto inv_mat = identity<T>(n);
    T* self_ptr = mat.ptr();
    T* inv_ptr = inv_mat.ptr();
    for(size_t iter = 0; iter < n; ++iter){
        // divide #iter row by matrix(iter, iter)
        auto self_start_iter = self_ptr + iter*n;
        auto inv_start_iter = inv_ptr + iter*n;
        {
            T val = mat(iter, iter);
            for(size_t c = 0; c < n; ++c)
            { self_start_iter[c] /= val; inv_start_iter[c] /= val; }
        }

        // row sub
        for(size_t r = 0; r < n; ++r){
            if(r == iter) continue;
            auto self_start_r = self_ptr + r*n;
            auto inv_start_r = inv_ptr + r*n;
            T ratio = mat(r, iter);
            for(size_t c = 0; c < n; ++c){
                self_start_r[c] -= self_start_iter[c] * ratio;
                inv_start_r[c] -= inv_start_iter[c] * ratio;
            }
        }
    }
    return inv_mat;
}
template<class T>
matrix<T> inv(matrix<T>&& m){
    matrix<T> mat = std::forward<matrix<T>>(m);
    return inv_impl(mat);
}
template<class T>
matrix<T> inv(matrix<T>& m){
    matrix<T> mat = m;
    return inv_impl(mat);
}


template<class T>
matrix<T> matmul(const matrix<T>& a, const matrix<T>& _b){
    auto&& [lrow, lcol] = a.shape();
    auto&& [rrow, rcol] = _b.shape();
    if(lcol != rrow){
        throw typename matrix<T>::DimensionalityException();
    }
    auto b = transpose<T>(_b);
    const size_t cache_size = CACHE_LINESIZE / sizeof(T);
    auto row = lrow, col = rcol;
    matrix<T> res{row, rcol};

    // things go lil bit nasty
    T* a_ptr = a.ptr();
    T* b_ptr = b.ptr();
    T* res_ptr = res.ptr();
    for(size_t r = 0; r < row; r += cache_size){
        size_t r_max = std::min(r + cache_size, row);
        for(size_t c = 0; c < col; c += cache_size){
            size_t c_max = std::min(c + cache_size, col);
            for(size_t k = 0; k < lcol; k += cache_size){
                size_t k_max = std::min(k + cache_size, lcol);
                for(size_t r_tile = r; r_tile < r_max; ++ r_tile){
                    for(size_t c_tile = c; c_tile < c_max; ++ c_tile){
                        T sum{};
                        for(size_t k_tile = k; k_tile < k_max; ++ k_tile)
                            sum += a_ptr[r_tile*lcol + k_tile] *
                                b_ptr[c_tile*lcol + k_tile];
                        res_ptr[r_tile*col + c_tile] += sum;
                    }
                }
            }
        }
    }
    return res;
}

template<class T>
matrix<T> transpose(const matrix<T>& m){
    auto&& [r, c] = m.shape();
    matrix<T> trans{c, r};
    for(size_t i = 0;i < r;++i)
        for(size_t j = 0;j < c;++j)
            trans(j, i) = m(i, j);
    return trans;
}

template<class T>
matrix<T> operator^(const matrix<T>& lhs, const matrix<T>& rhs){
    return matmul<T>(lhs, rhs);
}

template<class T>
matrix<T>& operator^=(matrix<T>& lhs, const matrix<T>& rhs){
    lhs = matmul<T>(lhs, rhs);
    return lhs;
}
template<class T>
matrix<T> operator~(matrix<T>&& m){
    matrix<T> mat = std::forward<matrix<T>>(m);
    return inv_impl(mat);
}
template<class T>
matrix<T> operator~(matrix<T>& m){
    matrix<T> mat = m;
    return inv_impl(mat);
}

template<class T> 
matrix<T> identity(size_t n){
    matrix<T> res{n};
    for(size_t idx = 0; idx < n; ++idx)
        res(idx, idx) = T{1};
    return res;
}

template<class T> 
matrix<T> randn(size_t r, size_t c = 1){
    static std::mt19937 gen(time(nullptr));
    std::uniform_real_distribution<T> dis;
    matrix<T> res{r, c};
    auto n = r*c;
    for(size_t idx = 0; idx < n; ++idx)
        res(idx / c, idx % c) = dis(gen);
    return res;
}

template<class T> 
matrix<T> diag(const matrix<T>& m){
    auto&& [r, c] = m.shape();
    if(r == 1 or c == 1){
        int n = (r == 1 ? c : r);
        matrix<T> res(n);
        for(size_t idx = 0; idx < n; ++ idx)
            res(idx, idx) = (r == 1 ? m(0, idx) : m(idx, 0));
        return res;
    }
    else if(r == c){
        matrix<T> res(r, 1);
        for(size_t idx = 0; idx < r; ++ idx)
            res(idx, 0) = m(idx, idx);
        return res;
    }
    throw typename matrix<T>::DimensionalityException();
}

}
}