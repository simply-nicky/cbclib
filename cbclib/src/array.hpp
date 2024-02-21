#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"

namespace cbclib {

namespace detail{

/* Returns a positive remainder of division */
template <typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
constexpr auto modulo(T a, U b) -> decltype(a % b)
{
    return (a % b + b) % b;
}

/* Returns a positive remainder of division */
template <typename T, typename U, typename = std::enable_if_t<std::is_floating_point_v<T> || std::is_floating_point_v<U>>>
constexpr auto modulo(T a, U b) -> decltype(std::fmod(a, b))
{
    return std::fmod(std::fmod(a, b) + b, b);
}

/* Returns a quotient: a = quotient * b + modulo(a, b) */
template <typename T, typename U>
constexpr auto quotient(T a, U b) -> decltype(modulo(a, b))
{
    return (a - modulo(a, b)) / b;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> mirror(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min) - 1;
    if (modulo(quotient(val, period), 2)) return period - modulo(val, period) + min;
    else return modulo(val, period) + min;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> reflect(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min);
    if (modulo(quotient(val, period), 2)) return period - 1 - modulo(val, period) + min;
    else return modulo(val, period) + min;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> wrap(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min);
    return modulo(val, period) + min;
}

template <typename InputIt1, typename InputIt2>
auto ravel_index_impl(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst)
{
    using value_t = decltype(+*std::declval<InputIt1 &>());
    value_t index = value_t();
    for (; cfirst != clast; cfirst++, ++sfirst) index += *cfirst * *sfirst;
    return index;
}

template <typename InputIt, typename OutputIt, typename T>
OutputIt unravel_index_impl(InputIt sfirst, InputIt slast, T index, OutputIt cfirst)
{
    for (; sfirst != slast; ++sfirst)
    {
        auto stride = index / *sfirst;
        index -= stride * *sfirst;
        *cfirst++ = stride;
    }
    return cfirst;
}

template <typename Strides>
size_t offset_along_dim(const Strides & strides, size_t index, size_t dim)
{
    if (dim == 0) return index;
    if (dim >= strides.size()) return 0;

    size_t offset = offset_along_dim(strides, index, dim - 1);
    return offset - (offset / strides[dim - 1]) * strides[dim - 1];
}

class shape_handler
{
public:
    size_t ndim;
    size_t size;
    std::vector<size_t> shape;

    using ShapeContainer = detail::any_container<size_t>;

    shape_handler(ShapeContainer sh, ShapeContainer st) : shape(std::move(sh)), strides(std::move(st))
    {
        ndim = shape.size();
        size = strides[ndim - 1];
        for (size_t i = 0; i < ndim; i++) size += (shape[i] - 1) * strides[i];
    }

    shape_handler(ShapeContainer sh) : shape(std::move(sh))
    {
        ndim = shape.size();
        size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies());
        size_t stride = size;
        for (auto length : shape)
        {
            stride /= length;
            strides.push_back(stride);
        }
    }

    ssize_t stride(size_t dim) const
    {
        if (dim >= this->ndim) fail_dim_check(dim, "invalid axis");
        return this->strides[dim];
    }

    size_t index_along_dim(size_t index, size_t dim) const
    {
        if (dim >= ndim) fail_dim_check(dim, "invalid axis");
        return offset_along_dim(strides, index, dim) / strides[dim];
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator<CoordIter>::value>>
    bool is_inbound(CoordIter first, CoordIter last) const
    {
        bool flag = true;
        for (size_t i = 0; first != last; ++first, ++i)
        {
            flag &= *first >= 0 && *first < static_cast<decltype(+*std::declval<CoordIter &>())>(this->shape[i]);
        }
        return flag;
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    bool is_inbound(const Container & coord) const
    {
        return is_inbound(coord.begin(), coord.end());
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    bool is_inbound(const std::initializer_list<T> & coord) const
    {
        return is_inbound(coord.begin(), coord.end());
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator<CoordIter>::value>>
    auto ravel_index(CoordIter first, CoordIter last) const
    {
        return ravel_index_impl(first, last, this->strides.begin());
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto ravel_index(const Container & coord) const
    {
        return ravel_index_impl(coord.begin(), coord.end(), this->strides.begin());
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto ravel_index(const std::initializer_list<T> & coord) const
    {
        return ravel_index_impl(coord.begin(), coord.end(), this->strides.begin());
    }

    template <
        typename CoordIter,
        typename = std::enable_if_t<
            std::is_integral_v<typename CoordIter::value_type> || 
            std::is_same_v<typename CoordIter::iterator_category, std::output_iterator_tag>
        >
    >
    CoordIter unravel_index(CoordIter first, size_t index) const
    {
        return unravel_index_impl(this->strides.begin(), this->strides.end(), index, first);
    }

protected:
    std::vector<size_t> strides;


    void fail_dim_check(size_t dim, const std::string & msg) const
    {
        throw std::out_of_range(msg + ": " + std::to_string(dim) + " (ndim = " + std::to_string(this->ndim) + ')');
    }
};

}

template <typename T, bool IsConst>
struct IteratorTraits;

template <typename T>
struct IteratorTraits<T, false>
{
  using value_type = T;
  using pointer = T *;
  using reference = T &;
};

template <typename T>
struct IteratorTraits<T, true>
{
  using value_type = const T;
  using pointer = const T *;
  using reference = const T &;
};

template <typename T>
class array;

template <typename T, bool IsConst>
class strided_iterator
{
    friend class strided_iterator<T, !IsConst>;
    friend class array<T>;
    using traits = IteratorTraits<T, IsConst>;

public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename traits::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename traits::pointer;
    using reference = typename traits::reference;

    // This is templated so that we can allow constructing a const iterator from
    // a nonconst iterator...
    template <bool RHIsConst, typename = std::enable_if_t<IsConst || !RHIsConst>>
    strided_iterator(const strided_iterator<T, RHIsConst> & rhs) : ptr(rhs.ptr), stride(rhs.stride) {}

    operator bool() const {return bool(ptr);}

    bool operator==(const strided_iterator<T, IsConst> & rhs) const {return ptr == rhs.ptr;}
    bool operator!=(const strided_iterator<T, IsConst> & rhs) const {return ptr != rhs.ptr;}
    bool operator<=(const strided_iterator<T, IsConst> & rhs) const {return ptr <= rhs.ptr;}
    bool operator>=(const strided_iterator<T, IsConst> & rhs) const {return ptr >= rhs.ptr;}
    bool operator<(const strided_iterator<T, IsConst> & rhs) const {return ptr < rhs.ptr;}
    bool operator>(const strided_iterator<T, IsConst> & rhs) const {return ptr > rhs.ptr;}

    strided_iterator<T, IsConst> & operator+=(const difference_type & step) {ptr += step * stride; return *this;}
    strided_iterator<T, IsConst> & operator-=(const difference_type & step) {ptr -= step * stride; return *this;}
    strided_iterator<T, IsConst> & operator++() {ptr += stride; return *this;}
    strided_iterator<T, IsConst> & operator--() {ptr -= stride; return *this;}
    strided_iterator<T, IsConst> operator++(int) {strided_iterator<T, IsConst> temp = *this; ++(*this); return temp;}
    strided_iterator<T, IsConst> operator--(int) {strided_iterator<T, IsConst> temp = *this; --(*this); return temp;}
    strided_iterator<T, IsConst> operator+(const difference_type & step) const
    {
        return {ptr + step * stride, stride};
    }
    strided_iterator<T, IsConst> operator-(const difference_type & step) const
    {
        return {ptr - step * stride, stride};
    }

    difference_type operator-(const strided_iterator<T, IsConst> & rhs) const {return (ptr - rhs.ptr) / stride;}

    reference operator[] (size_t index) const {return ptr[index * stride];}
    reference operator*() const {return *(ptr);}
    pointer operator->() const {return ptr;}
    
private:
    T * ptr;
    size_t stride;

    strided_iterator(T * ptr, size_t stride = 1) : ptr(ptr), stride(stride) {}
};

template <typename T>
class array : public detail::shape_handler
{
public:

    using value_type = T;
    using iterator = strided_iterator<T, false>;
    using const_iterator = strided_iterator<T, true>;

    operator py::array_t<T>() const {return {shape, ptr};}

    array(ShapeContainer shape, ShapeContainer strides, T * ptr) :
        shape_handler(std::move(shape), std::move(strides)), ptr(ptr) {}

    array(shape_handler handler, T * ptr) : shape_handler(std::move(handler)), ptr(ptr) {}

    array(ShapeContainer shape, T * ptr) : shape_handler(std::move(shape)), ptr(ptr) {}

    array(const py::buffer_info & buf) : array(buf.shape, static_cast<T *>(buf.ptr)) {}

    T & operator[] (size_t index) {return ptr[index];}
    const T & operator[] (size_t index) const {return ptr[index];}

    iterator begin() {return {ptr, strides[ndim - 1]};}
    iterator end() {return {ptr + size, strides[ndim - 1]};}
    const_iterator begin() const {return {ptr, strides[ndim - 1]};}
    const_iterator end() const {return {ptr + size, strides[ndim - 1]};}

    template <bool IsConst>
    typename strided_iterator<T, IsConst>::difference_type index(const strided_iterator<T, IsConst> & iter) const
    {
        return iter.ptr - ptr;
    }

    array<T> reshape(ShapeContainer new_shape) const
    {
        return {std::move(new_shape), ptr};
    }

    array<T> slice(size_t index, ShapeContainer axes) const
    {
        std::sort(axes->begin(), axes->end());

        std::vector<size_t> other_shape, new_shape, new_strides;
        for (size_t i = 0; i < ndim; i++)
        {
            if (std::find(axes->begin(), axes->end(), i) == axes->end()) other_shape.push_back(shape[i]);
        }
        std::transform(axes->begin(), axes->end(), std::back_inserter(new_shape), [this](size_t axis){return shape[axis];});
        std::transform(axes->begin(), axes->end(), std::back_inserter(new_strides), [this](size_t axis){return strides[axis];});

        std::vector<size_t> coord;
        shape_handler(std::move(other_shape)).unravel_index(std::back_inserter(coord), index);
        for (auto axis : *axes) coord.insert(std::next(coord.begin(), axis), 0);

        return array<T>(std::move(new_shape), std::move(new_strides), ptr + ravel_index(coord.begin(), coord.end()));
    }

    iterator line_begin(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t lsize = shape[axis] * strides[axis];
        T * iter = ptr + lsize * (index / strides[axis]) + index % strides[axis];
        return {iter, strides[axis]};
    }

    const_iterator line_begin(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t lsize = shape[axis] * strides[axis];
        T * iter = ptr + lsize * (index / strides[axis]) + index % strides[axis];
        return {iter, strides[axis]};
    }

    iterator line_end(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t lsize = shape[axis] * strides[axis];
        T * iter = ptr + lsize * (index / strides[axis]) + index % strides[axis];
        return {iter + lsize, strides[axis]};
    }

    const_iterator line_end(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t lsize = shape[axis] * strides[axis];
        T * iter = ptr + lsize * (index / strides[axis]) + index % strides[axis];
        return {iter + lsize, strides[axis]};
    }

    const T * data() const {return ptr;}
    T * data() {return ptr;}

protected:
    void check_index(size_t axis, size_t index) const
    {
        if (axis >= ndim || index >= (size / shape[axis]))
            throw std::out_of_range("index " + std::to_string(index) + " is out of bound for axis "
                                    + std::to_string(axis));
    }

    void set_data(T * new_ptr) {ptr = new_ptr;}

private:
    T * ptr;
};

template <typename T>
class vector_array : public array<T>
{
    std::vector<T> buffer;

public:
    vector_array(typename array<T>::ShapeContainer shape) : array<T>(std::move(shape), nullptr)
    {
        buffer = std::vector<T>(this->size, T());
        this->set_data(buffer.data());
    }
};

/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangular iterator ---------------------------*/
/*----------------------------------------------------------------------------*/

class rect_iterator : public detail::shape_handler
{
public:
    std::vector<size_t> coord;
    size_t index;

    rect_iterator(ShapeContainer shape) : shape_handler(std::move(shape)), index(0)
    {
        unravel_index(std::back_inserter(coord), index);
    }

    rect_iterator & operator++()
    {
        index++;
        unravel_index(coord.begin(), index);
        return *this;
    }

    rect_iterator operator++(int)
    {
        rect_iterator temp = *this;
        index++;
        unravel_index(coord.begin(), index);
        return temp;
    }

    bool is_end() const {return index >= size; }
};

/*----------------------------------------------------------------------------*/
/*------------------------------- Wirth select -------------------------------*/
/*----------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
    Function :  kth_smallest()
    In       :  array of elements, n elements in the array, rank k 
    Out      :  one element
    Job      :  find the kth smallest element in the array
    Notice   :  Buffer must be of size n

    Reference:
        Author: Wirth, Niklaus
        Title: Algorithms + data structures = programs
        Publisher: Englewood Cliffs: Prentice-Hall, 1976 Physical description: 366 p.
        Series: Prentice-Hall Series in Automatic Computation
---------------------------------------------------------------------------*/
template <class RandomIt, class Compare>
RandomIt wirthselect(RandomIt first, RandomIt last, typename std::iterator_traits<RandomIt>::difference_type k, Compare comp)
{
    auto l = first;
    auto m = std::prev(last);
    auto key = std::next(first, k);
    while (l < m)
    {
        auto value = *key;
        auto i = l;
        auto j = m;

        do
        {
            while (comp(*i, value)) ++i;
            while (comp(value, *j)) --j;
            if (i <= j) iter_swap(i++, j--);
        } while (i <= j);
        if (j < key) l = i;
        if (key < i) m = j;
    }
    
    return key;
}

template <class RandomIt, class Compare>
RandomIt wirthmedian(RandomIt first, RandomIt last, Compare comp)
{
    auto n = std::distance(first, last);
    return wirthselect(first, last, (n & 1) ? n / 2 : n / 2 - 1, comp);
}

/*----------------------------------------------------------------------------*/
/*--------------------------- Extend line modes ------------------------------*/
/*----------------------------------------------------------------------------*/
/*
    constant: kkkkkkkk|abcd|kkkkkkkk
    nearest:  aaaaaaaa|abcd|dddddddd
    mirror:   cbabcdcb|abcd|cbabcdcb
    reflect:  abcddcba|abcd|dcbaabcd
    wrap:     abcdabcd|abcd|abcdabcd
*/
enum class extend
{
    constant = 0,
    nearest = 1,
    mirror = 2,
    reflect = 3,
    wrap = 4
};

static std::unordered_map<std::string, extend> const modes = {{"constant", extend::constant},
                                                              {"nearest", extend::nearest},
                                                              {"mirror", extend::mirror},
                                                              {"reflect", extend::reflect},
                                                              {"wrap", extend::wrap}};

/*----------------------------------------------------------------------------*/
/*-------------------------------- Kernels -----------------------------------*/
/*----------------------------------------------------------------------------*/
/* All kernels defined with the support of [-1, 1]. */
namespace detail {

template <typename T>
T rectangular(T x, T sigma) {return (x <= sigma) ? T(1.0) : T(0.0);}

template <typename T>
T gaussian(T x, T sigma) {return exp(-std::pow(3 * x / sigma, 2) / 2) / Constants::M_1_SQRT2PI;}

template <typename T>
T triangular(T x, T sigma) {return std::max<T>(1 - std::abs(x / sigma), T());}

template <typename T>
T parabolic(T x, T sigma) {return T(0.75) * std::max<T>(1 - std::pow(x / sigma, 2), T());}

template <typename T>
T biweight(T x, T sigma) {return 15 / 16 * std::max<T>(std::pow(1 - std::pow(x / sigma, 2), 2), T());}

}

template <typename T>
struct kernels
{
    using kernel = T (*)(T, T);

    static inline std::map<std::string, kernel> registered_kernels = {{"biweight"   , detail::biweight<T>},
                                                                      {"gaussian"   , detail::gaussian<T>},
                                                                      {"parabolic"  , detail::parabolic<T>},
                                                                      {"rectangular", detail::rectangular<T>},
                                                                      {"triangular" , detail::triangular<T>}};

    static kernel get_kernel(std::string name, bool throw_if_missing = true)
    {
        auto it = registered_kernels.find(name);
        if (it != registered_kernels.end()) return it->second;
        if (throw_if_missing)
            throw std::invalid_argument("kernel is missing for " + name);
        return nullptr;
    }
};

}

#endif