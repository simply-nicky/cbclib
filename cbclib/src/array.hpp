#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"

namespace cbclib {

namespace detail{

template <typename T>
inline constexpr int signum(T val)
{
    return (T(0) < val) - (val < T(0));
}

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

template <size_t Dim = 0, typename Strides>
size_t ravel_index_var(const Strides & strides)
{
    return 0;
}

template <size_t Dim = 0, typename Strides, typename... Ix>
size_t ravel_index_var(const Strides & strides, size_t i, Ix... index)
{
    return i * strides[Dim] + ravel_index_var<Dim + 1>(strides, index...);
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

    shape_handler() = default;

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
            stride = (length) ? stride / length : stride;
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

    template <typename... Ix, typename = std::enable_if_t<(std::is_integral_v<Ix> && ...)>>
    auto ravel_index(Ix... index) const
    {
        if (sizeof...(index) > ndim) fail_dim_check(sizeof...(index), "too many indices for an array");

        return ravel_index_var(strides, size_t(index)...);
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
            std::is_integral_v<typename std::iterator_traits<CoordIter>::value_type> ||
            std::is_same_v<typename std::iterator_traits<CoordIter>::iterator_category, std::output_iterator_tag>
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

// Taken from the boost::hash_combine: https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
template <class T>
inline size_t hash_combine(size_t seed, const T & v)
{
    return seed ^ (std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename T, size_t N>
struct ArrayHasher
{
    size_t operator()(const std::array<T, N> & arr) const
    {
        size_t h = 0;
        for (auto elem : arr) h = hash_combine(h, elem);
        return h;
    }
};

// Recursive template code derived from Matthieu M.
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl
{
    static size_t apply(size_t seed, const Tuple & tuple)
    {
        seed = HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
        return hash_combine(seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0>
{
    static size_t apply(size_t seed, const Tuple & tuple)
    {
        return hash_combine(seed, std::get<0>(tuple));
    }
};


template <typename ... Ts>
struct TupleHasher
{
    size_t operator()(const std::tuple<Ts...> & tt) const
    {
        return HashValueImpl<std::tuple<Ts...>>::apply(0, tt);
    }
};

template <typename T1, typename T2>
struct PairHasher
{
    size_t operator()(const std::pair<T1, T2> & tt) const
    {
        return HashValueImpl<std::pair<T1, T2>>::apply(0, tt);
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

    strided_iterator() : ptr(nullptr), stride(1) {}

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

    array() : shape_handler(), ptr(nullptr) {}

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

    /* Line slice iterators:
        Take a slice of an array 'array' as follows:
        - array[..., :, ...] slice, where ':' is at 'axis'-th axis
        - ravel_index(i_0, i_1, ..., i_axis-1, i_axis+1, ...., i_n-1) = index
    */
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

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator_v<CoordIter>>>
    const T & at(CoordIter first, CoordIter last) const
    {
        return ptr[ravel_index(first, last)];
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator_v<CoordIter>>>
    T & at(CoordIter first, CoordIter last)
    {
        return ptr[ravel_index(first, last)];
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    const T & at(const Container & coord) const
    {
        return ptr[ravel_index(coord)];
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    T & at(const Container & coord)
    {
        return ptr[ravel_index(coord)];
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    const T & at(const std::initializer_list<I> & coord) const
    {
        return ptr[ravel_index(coord)];
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    T & at(const std::initializer_list<I> & coord)
    {
        return ptr[ravel_index(coord)];
    }

    template <typename... Ix, typename = std::enable_if_t<(std::is_integral_v<Ix> && ...)>>
    const T & at(Ix... index) const
    {
        return ptr[ravel_index(index...)];
    }

    template <typename... Ix, typename = std::enable_if_t<(std::is_integral_v<Ix> && ...)>>
    T & at(Ix... index)
    {
        return ptr[ravel_index(index...)];
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
    vector_array() = default;

    template <typename Vector, typename = std::enable_if_t<std::is_base_of_v<std::vector<T>, std::remove_cvref_t<Vector>>>>
    vector_array(Vector && v, detail::shape_handler::ShapeContainer shape) : array<T>(std::move(shape), v.data()), buffer(std::forward<Vector>(v))
    {
        if (buffer.size() != this->size) buffer.resize(this->size);
    }

    vector_array(detail::shape_handler::ShapeContainer shape) : array<T>(std::move(shape), nullptr)
    {
        buffer = std::vector<T>(this->size, T());
        array<T>::set_data(buffer.data());
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
T rectangular(T x) {return (std::abs(x) <= T(1.0)) ? T(1.0) : T();}

template <typename T>
T gaussian(T x)
{
    if (std::abs(x) <= T(1.0)) return Constants::M_1_SQRT2PI * std::exp(-std::pow(3 * x, 2) / 2);
    return T();
}

template <typename T>
T gaussian_grad(T x) {return -9 * x * gaussian(x);}

template <typename T>
T triangular(T x) {return std::max<T>(T(1.0) - std::abs(x), T());}

template <typename T>
T triangular_grad(T x)
{
    if (std::abs(x) < T(1.0)) return -signum(x);
    return T();
}

template <typename T>
T parabolic(T x) {return T(0.75) * std::max<T>(1 - std::pow(x, 2), T());}

template <typename T>
T parabolic_grad(T x)
{
    if (std::abs(x) < T(1.0)) return T(0.75) * -2 * x;
    return T();
}

template <typename T>
T biweight(T x) {return T(0.9375) * std::pow(std::max<T>(1 - std::pow(x, 2), T()), 2);}

template <typename T>
T biweight_grad(T x)
{
    if (std::abs(x) < T(1.0)) return T(0.9375) * -4 * x * (1 - std::pow(x, 2));
    return T();
}

}

template <typename T>
struct kernels
{
    enum kernel_type
    {
        biweight = 0,
        gaussian = 1,
        parabolic = 2,
        rectangular = 3,
        triangular = 4
    };

    using kernel = T (*)(T);
    using gradient = T (*)(T);

    static inline std::map<std::string, kernel_type> kernel_names =
    {
        {"biweight", kernel_type::biweight},
        {"gaussian", kernel_type::gaussian},
        {"parabolic", kernel_type::parabolic},
        {"rectangular", kernel_type::rectangular},
        {"triangular", kernel_type::triangular}
    };

    static inline std::map<kernel_type, std::pair<kernel, kernel>> registered_kernels =
    {
        {kernel_type::biweight, {detail::biweight<T>, detail::biweight_grad<T>}},
        {kernel_type::gaussian, {detail::gaussian<T>, detail::gaussian_grad<T>}},
        {kernel_type::parabolic, {detail::parabolic<T>, detail::parabolic_grad<T>}},
        {kernel_type::rectangular, {detail::rectangular<T>, nullptr}},
        {kernel_type::triangular, {detail::triangular<T>, detail::triangular_grad<T>}}
    };

    static kernel get_kernel(kernel_type k, bool throw_if_missing = true)
    {
        auto it = registered_kernels.find(k);
        if (it != registered_kernels.end()) return it->second.first;
        if (throw_if_missing)
            throw std::invalid_argument("kernel is missing for " + std::to_string(k));
        return nullptr;
    }

    static kernel get_kernel(std::string name, bool throw_if_missing = true)
    {
        auto it = kernel_names.find(name);
        if (it != kernel_names.end()) return get_kernel(it->second, throw_if_missing);
        if (throw_if_missing)
            throw std::invalid_argument("kernel is missing for " + name);
        return nullptr;
    }

    static kernel get_grad(kernel_type k, bool throw_if_missing = true)
    {
        auto it = registered_kernels.find(k);
        if (it != registered_kernels.end() && it->second.second) return it->second.second;
        if (throw_if_missing)
            throw std::invalid_argument("gradient is missing for " + std::to_string(k));
        return nullptr;
    }

    static kernel get_grad(std::string name, bool throw_if_missing = true)
    {
        auto it = kernel_names.find(name);
        if (it != kernel_names.end()) return get_grad(it->second, throw_if_missing);
        if (throw_if_missing)
            throw std::invalid_argument("gradient is missing for " + name);
        return nullptr;
    }


};

/*----------------------------------------------------------------------------*/
/*-------------- Compile-time to_array, to_tuple, and to_tie -----------------*/
/*----------------------------------------------------------------------------*/
namespace detail {
template <size_t... I>
constexpr auto integral_sequence_impl(std::index_sequence<I...>)
{
  return std::make_tuple(std::integral_constant<size_t, I>{}...);
}

template <typename T, T... I, size_t... J>
constexpr auto reverse_impl(std::integer_sequence<T, I...>, std::index_sequence<J...>)
{
    return std::integer_sequence<T, std::get<sizeof...(J) - J - 1>(std::make_tuple(I...))...>{};
}

}

template <size_t N>
constexpr auto integral_sequence()
{
    return detail::integral_sequence_impl(std::make_index_sequence<N>{});
}

template <typename T, T... I>
constexpr auto reverse_sequence(std::integer_sequence<T, I...> seq)
{
    return detail::reverse_impl(seq, std::make_index_sequence<sizeof...(I)>{});
}

template <size_t N, class Func>
constexpr decltype(auto) apply_to_sequence(Func && func)
{
    return std::apply(std::forward<Func>(func), integral_sequence<N>());
}

template <size_t N, class Container, typename T = Container::value_type>
constexpr std::array<T, N> to_array(const Container & a, size_t start)
{
    auto impl = [&a, start](auto... idxs) -> std::array<T, N> {return {{a[start + idxs]...}};};
    return apply_to_sequence<N>(impl);
}

template <size_t N, class Container>
constexpr auto to_tuple(const Container & a, size_t start)
{
    return apply_to_sequence<N>([&a, start](auto... idxs){return std::make_tuple(a[start + idxs]...);});
}

template <typename T, size_t N>
constexpr auto to_tuple(const std::array<T, N> & a)
{
    return apply_to_sequence<N>([&a](auto... idxs){return std::make_tuple(a[idxs]...);});
}

template <size_t N, class Container>
constexpr auto to_tie(Container & a, size_t start)
{
    return apply_to_sequence<N>([&a, start](auto... idxs){return std::tie(a[start + idxs]...);});
}

template <typename T, size_t N>
constexpr auto to_tie(std::array<T, N> & a)
{
    return apply_to_sequence<N>([&a](auto... idxs){return std::tie(a[idxs]...);});
}

}

#endif
