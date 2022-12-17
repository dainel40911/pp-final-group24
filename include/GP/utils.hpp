#pragma once
#include <GP/matrix/matrix.hpp>

namespace GP{

namespace utils{

template<class T>
T sum(const matrix<T>& m){
    T* m_ptr = m.ptr();
    auto && [row, col] = m.shape();
    auto n = row*col;
    T s{};
    for(size_t idx = 0; idx < n; ++idx)
        s += m_ptr[idx];
    return s;
}

template<class T>
T mean(const matrix<T>& m){
    T s{sum(m)};
    auto && [row, col] = m.shape();
    auto n = row*col;
    return s / n;
}

}

}