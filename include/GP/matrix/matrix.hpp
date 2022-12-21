#pragma once
#include <algorithm>
#include <memory>
#include <cstring>
#include <utility>
#include <iostream>
#include <iomanip>


namespace GP{

template<class T>
class matrix{
private:
    std::shared_ptr<T[]> buffer_;
    size_t row_;
    size_t col_;
public:
    static size_t copy_count;
    using value_type = T;
    class DimensionalityException : public std::exception{
    public:
        const char* what() const throw() { 
            return "GP::matrix dimensions don't match.";
        } 
    };

    static void matrix_get_copied(size_t space){
        // this function is for debug
        std::cout << "Matrix getting copied. "<< space<<" items are copied.\n";
        copy_count+=space;
    }

    // constructor
    matrix():
        row_{1}, col_{1}, buffer_{new value_type[1]{}}
        {}
    matrix(size_t w, size_t h):
        row_{w}, col_{h}, buffer_{new value_type[w*h]{}}
        {
            if(w * h == 0){
                throw DimensionalityException();
            }
        }
    matrix(size_t n):
        row_{n}, col_{n}, buffer_{new value_type[n*n]{}}
        {
            if(n == 0){
                throw DimensionalityException();
            }
        }
    matrix(const matrix<value_type>& m):
        row_{m.row_}, col_{m.col_}, buffer_{new value_type[m.row_*m.col_]}
    {
        size_t buffer_size = row_*col_;
        std::memcpy(
            buffer_.get(),
            m.buffer_.get(),
            buffer_size*sizeof(value_type));
        // matrix_get_copied(buffer_size);
    }
    matrix(matrix<value_type>&& m):
        row_{m.row_}, col_{m.col_}, buffer_{m.buffer_}
    {}

    // assign
    matrix<value_type>& operator=(const matrix<value_type>& m){
        row_ = m.row_; col_ = m.col_;
        size_t buffer_size = row_*col_;
        buffer_.reset(new value_type[buffer_size]);
        std::memcpy(
            buffer_.get(),
            m.buffer_.get(),
            buffer_size*sizeof(value_type));
        // matrix_get_copied(buffer_size);
        return *this;
    }
    matrix<value_type>& operator=(matrix<value_type>&& m){
        row_ = m.row_; col_ = m.col_;
        buffer_ = m.buffer_;
        return *this;
    }

    inline auto size() const -> size_t {
        return row_*col_;
    }
    inline auto shape() const -> std::pair<size_t, size_t> {
        return std::pair<size_t, size_t>{row_, col_};
    }
    auto shape(size_t dim) const -> size_t {
        switch (dim)
        {
        case 0:
            return row_;
            break;
        case 1:
            return col_;
            break;
        
        default:
            throw DimensionalityException();
        }
    }
    inline auto ptr() const {
        return buffer_.get();
    }
    inline value_type& operator()(size_t r, size_t c){
        return buffer_[r*col_ + c];
    }
    inline value_type operator()(size_t r, size_t c) const {
        return buffer_[r*col_ + c];
    }

    // operator
    matrix<value_type> operator+(value_type val){
        matrix<value_type> res{row_, col_};
        auto last = res.size();
        for(size_t idx = 0; idx < last; ++ idx) res.buffer_[idx] = buffer_[idx] + val;
        return res;
    }
    matrix<value_type> operator+(const matrix<value_type>& rhs){
        auto&& [rrow, rcol] = rhs.shape();
        if(row_ != rrow or col_ != rcol){
            throw matrix<value_type>::DimensionalityException();
        }
        matrix<value_type> res{row_, col_};
        auto last = res.size();
        for(size_t idx = 0; idx < last; ++ idx)
            res.buffer_[idx] = buffer_[idx] + rhs.buffer_[idx];
        return res;
    }
    inline matrix<value_type> operator-(){
        return (*this)*value_type{-1};
    }
    inline matrix<value_type> operator-(value_type val){
        return (*this) + (-val);
    }
    matrix<value_type> operator-(const matrix<value_type>& rhs){
        auto&& [rrow, rcol] = rhs.shape();
        if(row_ != rrow or col_ != rcol){
            throw matrix<value_type>::DimensionalityException();
        }
        matrix<value_type> res{row_, col_};
        auto last = res.size();
        for(size_t idx = 0; idx < last; ++ idx)
            res.buffer_[idx] = buffer_[idx] - rhs.buffer_[idx];
        return res;
    }
    matrix<value_type> operator*(value_type val){
        matrix<value_type> res{row_, col_};
        auto last = res.size();
        for(size_t idx = 0; idx < last; ++ idx) res.buffer_[idx] = buffer_[idx] * val;
        return res;
    }

    // any $= operator
    matrix<value_type>& operator=(value_type val){
        auto last = this->size();
        for(size_t idx = 0; idx < last; ++ idx) buffer_[idx] = val;
        return *this;
    }
    matrix<value_type>& operator+=(value_type val){
        auto last = this->size();
        for(size_t idx = 0; idx < last; ++ idx) buffer_[idx] += val;
        return *this;
    }
    inline matrix<value_type>& operator-=(value_type val){
        return (*this) += -val;
    }
    matrix<value_type>& operator*=(value_type val){
        auto last = this->size();
        for(size_t idx = 0; idx < last; ++ idx) buffer_[idx] *= val;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream &os, const matrix<T>& m){
        for(size_t r = 0; r < m.row_;++r){
            for(size_t c = 0; c < m.col_;++c)
                os << std::setprecision(3) << std::fixed << m(r, c) << '\t';
            os << '\n';
        }
        return os;
    }
    friend std::istream& operator>>(std::istream &is, matrix<T>& m){
        size_t row, col;
        is >> row >> col;
        m = matrix<T>{row, col};
        size_t n = row*col;
        for(size_t idx = 0; idx < n;++idx){
            is >> m.buffer_[idx];
        }
        return is;
    }
};
template<>
size_t matrix<double>::copy_count = 0;

using Matrix = matrix<double>;
}