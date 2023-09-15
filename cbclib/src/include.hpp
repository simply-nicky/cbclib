#ifndef INCLUDE_
#define INCLUDE_

#include <cassert>
#include <cstring>
#include <experimental/iterator>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>
#include <math.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fftw3.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define PYBIND11_DETAILED_ERROR_MESSAGES

namespace cbclib {

namespace py = pybind11;

namespace detail {

template <typename T, typename = void>
struct is_input_iterator : std::false_type {};

template <typename T>
struct is_input_iterator<T,
    std::void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>
> : std::true_type {};

template <typename T>
class any_container
{
protected:
    std::vector<T> vec;

public:
    any_container() = default;

    template <typename It, typename = std::enable_if_t<is_input_iterator<It>::value>>
    any_container(It first, It last) : vec(first, last) {}

    template <typename Container,
        typename = std::enable_if_t<
            std::is_convertible_v<decltype(*std::begin(std::declval<const Container &>())), T>
        >
    >
    any_container(const Container & c) : any_container(std::begin(c), std::end(c)) {}

    template <typename TIn, typename = std::enable_if_t<std::is_convertible_v<TIn, T>>>
    any_container(const std::initializer_list<TIn> & c) : any_container(c.begin(), c.end()) {}

    any_container(std::vector<T> && v) : vec(std::move(v)) {}

    operator std::vector<T> && () && { return std::move(this->vec); }

    std::vector<T> & operator*() {return this->vec;}
    const std::vector<T> & operator*() const {return this->vec;}

    std::vector<T> * operator->() {return &(this->vec);}
    const std::vector<T> * operator->() const {return &(this->vec);}
};

}

template <typename T>
class sequence : public detail::any_container<T>
{
public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using detail::any_container<T>::any_container;

    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    sequence(U value, size_t length = 1)
    {
        std::fill_n(std::back_inserter(this->vec), length, value);
    }

    template <typename Container, typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, T>>>
    sequence(const Container & vec, size_t length)
    {
        if (vec.size() < length) throw std::invalid_argument("rank of vector is less than the required length");
        std::copy_n(vec.begin(), length, std::back_inserter(this->vec));
    }

    size_t size() const {return this->vec.size();}
    sequence & unwrap(T max)
    {
        for (size_t i = 0; i < this->size(); i++)
        {
            this->vec[i] = (this->vec[i] >= 0) ? this->vec[i] : max + this->vec[i];
            if (this->vec[i] >= max)
                throw std::invalid_argument("axis is out of bounds");
        }
        return *this;
    }

    template <class Array>
    Array swap_axes(Array && arr) const
    {
        size_t counter = 0;
        for (py::ssize_t i = 0; i < arr.ndim(); i++)
        {
            if (std::find(this->vec.begin(), this->vec.end(), i) == this->vec.end())
            {
                auto obj = reinterpret_cast<PyArrayObject *>(arr.release().ptr());
                arr = py::reinterpret_steal<std::decay_t<Array>>(PyArray_SwapAxes(obj, counter++, i));
            }
        }
        return std::forward<Array>(arr);
    }

    template <class Array>
    Array swap_axes_back(Array && arr) const
    {
        size_t counter = arr.ndim() - this->size();
        for (py::ssize_t i = arr.ndim() - 1; i >= 0; i--)
        {
            if (std::find(this->vec.begin(), this->vec.end(), i) == this->vec.end())
            {
                auto obj = reinterpret_cast<PyArrayObject *>(arr.release().ptr());
                arr = py::reinterpret_steal<std::decay_t<Array>>(PyArray_SwapAxes(obj, --counter, i));
            }
        }
        return std::forward<Array>(arr);
    }

    T & operator[] (size_t index) {return this->vec[index];}
    const T & operator[] (size_t index) const {return this->vec[index];}

    iterator begin() {return this->vec.begin();}
    const_iterator begin() const {return this->vec.begin();}
    iterator end() {return this->vec.end();}
    const_iterator end() const {return this->vec.end();}
};

template <typename F, typename = std::enable_if_t<std::is_floating_point<F>::value>>
bool isclose(F a, F b, F atol = F(1e-8), F rtol = F(1e-5))
{
    if (fabs(a - b) <= atol + rtol * std::fmax(fabs(a), fabs(b))) return true;
    return false;
}

inline void * import_numpy() {import_array(); return NULL;}

}

#endif