#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"

namespace cbclib {

template <class InputIt>
size_t get_size(InputIt first, InputIt last)
{
    return std::reduce(first, last, size_t(1), std::multiplies<size_t>());
}

namespace detail{

template <typename InputIt1, typename InputIt2>
typename InputIt1::value_type ravel_index_impl(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst)
{
    using value_t = typename InputIt1::value_type;
    value_t index = value_t();
    for(; cfirst != clast; cfirst++, sfirst++) index += *cfirst * static_cast<value_t>(*sfirst);
    return index;
}

template <typename InputIt, typename OutputIt, typename T>
OutputIt unravel_index_impl(InputIt sfirst, InputIt slast, T index, OutputIt cfirst)
{
    using value_t = typename InputIt::value_type;
    for(; sfirst != slast; sfirst++)
    {
        value_t stride = index / *sfirst;
        index -= stride * *sfirst;
        *cfirst++ = stride;
    }
    return cfirst;
}

template <typename T, typename = void>
struct is_input_iterator : std::false_type {};

template <typename T>
struct is_input_iterator<T,
    std::void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>
> : std::true_type {};

template <typename T>
class any_container
{
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

class shape_handler
{
public:
    size_t ndim;
    size_t size;

    using ShapeContainer = detail::any_container<size_t>;

    shape_handler(size_t ndim, size_t size, ShapeContainer strides) : ndim(ndim), size(size), strides(std::move(strides)) {}

    shape_handler(ShapeContainer shape) : ndim(std::distance(shape->begin(), shape->end()))
    {
        this->size = get_size(shape->begin(), shape->end());
        size_t stride = this->size;
        for (auto length : *shape)
        {
            stride /= length;
            this->strides.push_back(stride);
        }
    }

    template <typename CoordIter, typename = std::enable_if_t<std::is_integral_v<typename CoordIter::value_type>>>
    size_t ravel_index(CoordIter first, CoordIter last) const
    {
        return ravel_index_impl(first, last, this->strides.begin());
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
};

}

template <typename T>
class strided_iterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    strided_iterator(T * ptr, size_t stride) : ptr(ptr), stride(stride) {}

    operator bool() const
    {
        if (this->ptr) return true;
        return false;
    }

    bool operator==(const strided_iterator<T> & iter) const {return this->ptr == iter.ptr;}
    bool operator!=(const strided_iterator<T> & iter) const {return this->ptr != iter.ptr;}
    bool operator<=(const strided_iterator<T> & iter) const {return this->ptr <= iter.ptr;}
    bool operator>=(const strided_iterator<T> & iter) const {return this->ptr >= iter.ptr;}
    bool operator<(const strided_iterator<T> & iter) const {return this->ptr < iter.ptr;}
    bool operator>(const strided_iterator<T> & iter) const {return this->ptr > iter.ptr;}

    strided_iterator<T> & operator+=(const difference_type & step) {this->ptr += step * this->stride; return *this;}
    strided_iterator<T> & operator-=(const difference_type & step) {this->ptr -= step * this->stride; return *this;}
    strided_iterator<T> & operator++() {this->ptr += this->stride; return *this;}
    strided_iterator<T> & operator--() {this->ptr -= this->stride; return *this;}
    strided_iterator<T> operator++(int) {strided_iterator<T> temp = *this; ++(*this); return temp;}
    strided_iterator<T> operator--(int) {strided_iterator<T> temp = *this; --(*this); return temp;}
    strided_iterator<T> operator+(const difference_type & step) const
    {
        return strided_iterator<T>(this->ptr + step * this->stride, this->stride);
    }
    strided_iterator<T> operator-(const difference_type & step) const
    {
        return strided_iterator<T>(this->ptr - step * this->stride, this->stride);
    }

    difference_type operator-(const strided_iterator<T> & iter) const {return std::distance(iter, *this);}

    T & operator[] (size_t index) const {return this->ptr[index * this->stride];}
    T & operator*() const {return *(this->ptr);}
    T * operator->() const {return this->ptr;}
    
private:
    T * ptr;
    size_t stride;
};

template <typename T>
class array : public detail::shape_handler
{
public:
    std::vector<size_t> shape;     // Shape of the array
    T * ptr;

    using iterator = T *;
    using const_iterator = const T *;

    array(size_t ndim, size_t size, ShapeContainer strides, ShapeContainer shape, T * ptr)
        : shape_handler(ndim, size, std::move(strides)), shape(std::move(shape)), ptr(ptr) {}

    array(shape_handler handler, std::vector<size_t> shape, T * ptr)
        : shape_handler(std::move(handler)), shape(std::move(shape)), ptr(ptr) {}

    array(ShapeContainer shape, T * ptr)
        : shape_handler(shape), shape(std::move(shape)), ptr(ptr) {}

    array(const py::buffer_info & buf) : array(buf.shape, static_cast<T *>(buf.ptr)) {}

    T & operator[] (size_t index) {return this->ptr[index];}
    const T & operator[] (size_t index) const {return this->ptr[index];}
    iterator begin() {return this->ptr;}
    iterator end() {return this->ptr + this->size;}
    const_iterator begin() const {return this->ptr;}
    const_iterator end() const {return this->ptr + this->size;}

    array<T> slice(size_t index, ShapeContainer axes) const
    {
        std::sort(axes->begin(), axes->end());

        std::vector<size_t> other_shape, shape, strides;
        for (size_t i = 0; i < this->ndim; i++)
        {
            if (std::find(axes->begin(), axes->end(), i) == axes->end()) other_shape.push_back(this->shape[i]);
        }
        std::transform(axes->begin(), axes->end(), std::back_inserter(shape), [this](size_t axis){return this->shape[axis];});
        std::transform(axes->begin(), axes->end(), std::back_inserter(strides), [this](size_t axis){return this->strides[axis];});

        std::vector<size_t> coord;
        shape_handler(std::move(other_shape)).unravel_index(std::back_inserter(coord), index);
        for (auto axis : *axes) coord.insert(std::next(coord.begin(), axis), 0);

        index = this->ravel_index(coord.begin(), coord.end());

        return array<T>(shape.size(), get_size(shape.begin(), shape.end()),
                        std::move(strides), std::move(shape), this->ptr + index);
    }

    strided_iterator<T> line_begin(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<T>(ptr, this->strides[axis]);
    }

    strided_iterator<const T> line_begin(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        const_iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<const T>(ptr, this->strides[axis]);
    }

    strided_iterator<T> line_end(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<T>(ptr + size, this->strides[axis]);
    }

    strided_iterator<const T> line_end(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        const_iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<const T>(ptr + size, this->strides[axis]);
    }

protected:
    void check_index(size_t axis, size_t index)
    {
        if (axis >= this->ndim || index >= (this->size / this->shape[axis]))
            throw std::out_of_range("index " + std::to_string(index) + " is out of bound for axis "
                                    + std::to_string(axis));
    }
};

template <typename T>
class vector_array : public array<T>
{
private:
    std::vector<T> buffer;

public:
    using ShapeContainer = detail::any_container<size_t>;

    vector_array(ShapeContainer shape) : array<T>(std::move(shape), nullptr)
    {
        this->buffer = std::vector<T>(this->size, T());
        this->ptr = this->buffer.data();
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
        this->unravel_index(std::back_inserter(this->coord), this->index);
    }

    rect_iterator & operator++()
    {
        this->index++;
        this->unravel_index(this->coord.begin(), this->index);
        return *this;
    }

    rect_iterator operator++(int)
    {
        rect_iterator temp = *this;
        this->index++;
        this->unravel_index(this->coord.begin(), this->index);
        return temp;
    }

    bool is_end() const {return this->index >= this->size; }
};

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

template <class BidirIt>
class mirror_generator
{
public:
    mirror_generator(BidirIt first, BidirIt last, BidirIt iter)
        : first(first), last(last), iter(iter) {}

    BidirIt operator()()
    {
        auto tmp = this->iter;
        this->iter++;
        if (this->iter == this->last) {this->iter = tmp; this = this->reverse();}
        return this->iter;
    }

    mirror_generator<BidirIt> & reverse() const
    {
        return mirror_generator<BidirIt>(std::reverse_iterator<BidirIt>(this->last),
                                        std::reverse_iterator<BidirIt>(this->first),
                                        std::reverse_iterator<BidirIt>(this->iter));
    }

private:
    BidirIt iter;
    BidirIt first;
    BidirIt last;
};

template <class BidirIt>
class reflect_generator
{
public:
    reflect_generator(BidirIt first, BidirIt last, BidirIt iter)
        : first(first), last(last), iter(iter) {}

    BidirIt operator()()
    {
        auto tmp = this->iter;
        this->iter++;
        if (this->iter == this->last) this = this->reverse();
        return tmp;
    }

    reflect_generator<BidirIt> & reverse() const
    {
        return reflect_generator<BidirIt>(std::reverse_iterator<BidirIt>(this->last),
                                        std::reverse_iterator<BidirIt>(this->first),
                                        std::reverse_iterator<BidirIt>(this->iter));
    }

private:
    BidirIt iter;
    BidirIt first;
    BidirIt last;
};

template <class BidirIt>
class wrap_generator
{
public:
    wrap_generator(BidirIt first, BidirIt last, BidirIt iter)
        : first(first), last(last), iter(iter) {}

    BidirIt operator()()
    {
        this->iter++;
        if (this->iter == this->last) this->iter = this->first;
        return this->iter;
    }

    wrap_generator<BidirIt> & reverse() const
    {
        return wrap_generator<BidirIt>(std::reverse_iterator<BidirIt>(this->last),
                                    std::reverse_iterator<BidirIt>(this->first),
                                    std::reverse_iterator<BidirIt>(this->iter));
    }

private:
    BidirIt iter;
    BidirIt first;
    BidirIt last;
};

template <class InputIt, class OutputIt, typename T>
void extend_line(InputIt first, InputIt last, OutputIt ofirst, OutputIt olast, extend mode, const T & cval)
{
    auto isize = std::distance(first, last);
    auto osize = std::distance(ofirst, olast);
    auto dsize = osize - isize;
    auto size_before = dsize - dsize / 2;
    auto size_after = dsize - size_before;

    std::copy(first, last, std::next(ofirst, size_before));

    switch (mode)
    {
        /* kkkkkkkk|abcd|kkkkkkkk */
        case extend::constant:

            std::fill_n(ofirst, size_before, cval);
            std::fill_n(std::next(ofirst, osize - size_after), size_after, cval);
            break;

        /* aaaaaaaa|abcd|dddddddd */
        case extend::nearest:

            std::fill_n(ofirst, size_before, *first);
            std::fill_n(std::next(ofirst, osize - size_after), *std::prev(last));
            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case extend::mirror:

            std::generate_n(std::reverse_iterator<OutputIt>(std::next(ofirst, size_before)), size_before,
                            mirror_generator<InputIt>(first, last, first));
            std::generate_n(std::next(ofirst, osize - size_after), size_after,
                            mirror_generator<InputIt>(first, last, last).reverse());
            break;

        /* abcddcba|abcd|dcbaabcd */
        case extend::reflect:
            
            std::generate_n(std::reverse_iterator<OutputIt>(std::next(ofirst, size_before)), size_before,
                            reflect_generator<InputIt>(first, last, first));
            std::generate_n(std::next(ofirst, osize - size_after), size_after,
                            reflect_generator<InputIt>(first, last, last).reverse());
            break;

        /* abcdabcd|abcd|abcdabcd */
        case extend::wrap:

            std::generate_n(std::reverse_iterator<OutputIt>(std::next(ofirst, size_before)), size_before,
                            wrap_generator<InputIt>(first, last, last).reverse());
            std::generate_n(std::next(ofirst, osize - size_after), size_after,
                            wrap_generator<InputIt>(first, last, first));
            break;

        default:
            throw std::invalid_argument("extend_line: invalid extend argument");
    }
}

template <typename Container, typename T, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
std::optional<T> extend_point(const Container & coord, const array<T> & arr, const array<bool> & mask, extend mode, const T & cval)
{
    using I = typename Container::value_type;

    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == extend::constant) return std::optional<T>(cval);

    std::vector<I> close (arr.ndim, I());
    size_t dist;

    switch (mode)
    {
        /* aaaaaaaa|abcd|dddddddd */
        case extend::nearest:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.shape[n])) close[n] = arr.shape[n] - 1;
                else if (coord[n] < 0) close[n] = 0;
                else close[n] = coord[n];
            }

            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case extend::mirror:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.shape[n]))
                {
                    close[n] = arr.shape[n] - 1;
                    dist = coord[n] - arr.shape[n] + 1;

                    while (dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n];

                    while (dist-- && close[n] < static_cast<I>(arr.shape[n])) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcddcba|abcd|dcbaabcd */
        case extend::reflect:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.shape[n]))
                {
                    close[n] = arr.shape[n] - 1;
                    dist = coord[n] - arr.shape[n];

                    while (dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n] - 1;

                    while (dist-- && close[n] < static_cast<I>(arr.shape[n])) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcdabcd|abcd|abcdabcd */
        case extend::wrap:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.shape[n]))
                {
                    close[n] = 0;
                    dist = coord[n] - arr.shape[n];

                    while (dist-- && close[n] < static_cast<I>(arr.shape[n])) close[n]++;
                }
                else if (coord[n] < 0)
                {
                    close[n] = arr.shape[n] - 1;
                    dist = -coord[n] - 1;

                    while (dist-- && close[n] >= 0) close[n]--;
                }
                else close[n] = coord[n];
            }

            break;

        default:
            throw std::invalid_argument("extend_point: invalid extend argument.");
    }

    size_t index = arr.ravel_index(close.begin(), close.end());

    if (mask[index]) return std::optional<T>(arr[index]);
    else return std::nullopt;
}

/*----------------------------------------------------------------------------*/
/*------------------------------ Binary search -------------------------------*/
/*----------------------------------------------------------------------------*/
// Array search
enum class side
{
    left = 0,
    right = 1
};

/* find idx \el [0, npts], so that base[idx - 1] < key <= base[idx] */
template <class ForwardIt, typename T, class Compare>
ForwardIt searchsorted(const T & value, ForwardIt first, ForwardIt last, side s, Compare comp)
{
    auto npts = std::distance(first, last);
    auto extreme = std::next(first, npts - 1);
    if (comp(value, *first)) return first;
    if (!comp(value, *extreme)) return extreme;

    ForwardIt out;
    switch (s)
    {
        case side::left:
            out = std::lower_bound(first, last, value, comp);
            break;

        case side::right:
            out = std::next(first, std::distance(first, std::upper_bound(first, last, value, comp)) - 1);
            break;

        default:
            throw std::invalid_argument("searchsorted: invalid side argument.");
    }
    return out;
}

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
            while (comp(*i, value)) i++;
            while (comp(value, *j)) j--;
            if (i <= j) std::swap(*(i++), *(j--));
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

}

#endif