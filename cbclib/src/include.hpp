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

template <typename T>
struct sequence
{
    std::vector<T> data;

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    sequence(std::vector<T> vec = std::vector<T>()) : data(std::move(vec)) {}

    template <typename Container, typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, T>>>
    sequence(const Container & vec) : data(vec.begin(), vec.end()) {}

    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    sequence(U value, size_t length = 1)
    {
        std::fill_n(std::back_inserter(this->data), length, value);
    }

    template <typename Container, typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, T>>>
    sequence(const Container & vec, size_t length)
    {
        std::copy_n(vec.begin(), length, std::back_inserter(this->data));
    }

    size_t size() const {return this->data.size();}
    sequence & unwrap(T max)
    {
        for (size_t i = 0; i < this->size(); i++)
        {
            this->data[i] = (this->data[i] >= 0) ? this->data[i] : max + this->data[i];
            if (this->data[i] >= max)
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
            if (std::find(this->data.begin(), this->data.end(), i) == this->data.end())
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
            if (std::find(this->data.begin(), this->data.end(), i) == this->data.end())
            {
                auto obj = reinterpret_cast<PyArrayObject *>(arr.release().ptr());
                arr = py::reinterpret_steal<std::decay_t<Array>>(PyArray_SwapAxes(obj, --counter, i));
            }
        }
        return std::forward<Array>(arr);
    }

    T & operator[] (size_t index) {return this->data[index];}
    const T & operator[] (size_t index) const {return this->data[index];}

    iterator begin() {return this->data.begin();}
    const_iterator begin() const {return this->data.begin();}
    iterator end() {return this->data.end();}
    const_iterator end() const {return this->data.end();}
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