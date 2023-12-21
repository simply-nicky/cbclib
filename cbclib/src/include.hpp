#ifndef INCLUDE_
#define INCLUDE_

#include <cassert>
#include <cstring>
#include <experimental/iterator>
#include <iostream>
#include <memory>
#include <mutex>
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

template <typename Container, typename = std::enable_if_t<std::is_rvalue_reference_v<Container &&>>>
inline py::array_t<typename Container::value_type> as_pyarray(Container && seq)
{
    Container * seq_ptr = new Container(std::move(seq));
    auto capsule = py::capsule(seq_ptr, [](void * p) {delete reinterpret_cast<Container *>(p);});
    return py::array(seq_ptr->size(),  // shape of array
                     seq_ptr->data(),  // c-style contiguous strides for Container
                     capsule           // numpy array references this parent
    );
}

template <typename Container>
inline py::array_t<typename Container::value_type> to_pyarray(const Container & seq)
{
    return py::array(seq.size(), seq.data());
}

namespace detail {

struct Constants
{
    // 1 / sqrt(2 * pi)
    static constexpr double M_1_SQRT2PI = 0.3989422804014327; 
};

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

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
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

template <class InputIt, typename = std::enable_if_t<detail::is_input_iterator<InputIt>::value>>
auto get_size(InputIt first, InputIt last)
{
    using value_t = typename InputIt::value_type;
    return std::reduce(first, last, value_t(1), std::multiplies<value_t>());
}

template <typename Container>
void check_dimensions(const std::string & name, ssize_t axis, const Container & shape) {}

template <typename Container, typename... Ix>
void check_dimensions(const std::string & name, ssize_t axis, const Container & shape, ssize_t i, Ix... index)
{
    if (axis < 0)
    {
        auto text = name + " has the wrong number of dimensions: " + std::to_string(shape.size()) +
                    " < " + std::to_string(shape.size() - axis);
        throw std::invalid_argument(text);
    }
    if (axis >= static_cast<ssize_t>(shape.size()))
    {
        auto text = name + " has the wrong number of dimensions: " + std::to_string(shape.size()) +
                    " < " + std::to_string(axis + 1);
        throw std::invalid_argument(text);
    }
    if (i != static_cast<ssize_t>(shape[axis]))
    {
        auto text = name + " has an incompatible shape at axis " + std::to_string(i) + ": " +
                    std::to_string(shape[axis]) + " != " + std::to_string(i);
        throw std::invalid_argument(text);
    }
    check_dimensions(name, axis + 1, shape, index...);
}

template <typename T, typename V>
void check_optional(const std::string & name, const py::array_t<T, py::array::c_style | py::array::forcecast> & inp,
                    std::optional<py::array_t<V, py::array::c_style | py::array::forcecast>> & opt, V fill_value)
{
    py::buffer_info ibuf = inp.request();
    if (!opt)
    {
        opt = py::array_t<V>(ibuf.shape);
        auto fill = py::array_t<V>(py::ssize_t(1));
        fill.mutable_at(0) = fill_value;
        PyArray_CopyInto(reinterpret_cast<PyArrayObject *>(opt.value().ptr()),
                         reinterpret_cast<PyArrayObject *>(fill.ptr()));
    }
    py::buffer_info obuf = opt.value().request();
    if (!std::equal(obuf.shape.begin(), obuf.shape.end(), ibuf.shape.begin()))
    {
        std::ostringstream oss1, oss2;
        std::copy(obuf.shape.begin(), obuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument(name + " and inp arrays must have identical shapes: {" + oss1.str() +
                                    "}, {" + oss2.str() + "}");
    }
}

template <typename Container, typename Function, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
void check_shape(const Container & shape, Function && func)
{
    if (std::forward<Function>(func)(shape) || !get_size(shape.begin(), shape.end()))
    {
        std::ostringstream oss;
        std::copy(shape.begin(), shape.end(), std::experimental::make_ostream_joiner(oss, ", "));
        throw std::invalid_argument("invalid shape: {" + oss.str() + "}");
    }
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
        if (vec.size() < length)
            throw std::invalid_argument("rank of vector (" + std::to_string(vec.size()) +
                                        ") is less than the required length (" + std::to_string(length) + ")");
        std::copy_n(vec.begin(), length, std::back_inserter(this->vec));
    }

    size_t size() const {return this->vec.size();}
    sequence & unwrap(T max)
    {
        for (size_t i = 0; i < this->size(); i++)
        {
            this->vec[i] = (this->vec[i] >= 0) ? this->vec[i] : max + this->vec[i];
            if (this->vec[i] >= max)
                throw std::invalid_argument("axis " + std::to_string(this->vec[i]) +
                                            " is out of bounds (ndim = " + std::to_string(max) + ")");
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
                arr = py::reinterpret_steal<std::remove_cvref_t<Array>>(PyArray_SwapAxes(obj, counter++, i));
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
                arr = py::reinterpret_steal<std::remove_cvref_t<Array>>(PyArray_SwapAxes(obj, --counter, i));
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

class thread_exception
{
    std::exception_ptr ptr;
    std::mutex lock;

public:

    thread_exception() : ptr(nullptr) {}

    void rethrow()
    {
        if (this->ptr) std::rethrow_exception(this->ptr);
    }

    void capture_exception()
    { 
        std::unique_lock<std::mutex> guard(this->lock);
        this->ptr = std::current_exception(); 
    }   

    template <typename Function, typename... Args>
    void run(Function && f, Args &&... params)
    {
        try 
        {
            std::forward<Function>(f)(std::forward<Args>(params)...);
        }
        catch (...)
        {
            capture_exception();
        }
    }
};

}

#endif