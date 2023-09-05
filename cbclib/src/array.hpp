#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"

namespace cbclib {

namespace detail{

template <typename InputIt1, typename InputIt2>
typename InputIt1::value_type ravel_index(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst)
{
    using value_t = typename InputIt1::value_type;
    value_t index = value_t();
    for(; cfirst != clast; cfirst++, sfirst++) index += *cfirst * static_cast<value_t>(*sfirst);
    return index;
}

template <typename InputIt, typename OutputIt, typename T>
OutputIt unravel_index(InputIt sfirst, InputIt slast, T index, OutputIt cfirst)
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
struct array
{
    size_t ndim;
    size_t size;                  // Number of elements
    std::vector<size_t> dims;     // Shape of the array
    std::vector<size_t> strides;
    T * data;

    using iterator = T *;
    using const_iterator = const T *;

    array(size_t ndim, size_t size, std::vector<size_t> dims, std::vector<size_t> strides, T * data)
        : ndim(ndim), size(size), dims(std::move(dims)), strides(std::move(strides)), data(data) {}

    template <
        class ShapeContainer,
        typename = std::enable_if_t<std::is_convertible_v<typename ShapeContainer::value_type, size_t>>
    >
    array(const ShapeContainer & dims, T * data)
        : ndim(std::distance(dims.begin(), dims.end())), dims(dims.begin(), dims.end()), data(data)
    {
        this->size = std::reduce(this->dims.begin(), this->dims.end(), size_t(1), [](size_t a, size_t b){return a * b;});

        size_t stride = this->size;
        for (auto dim : this->dims)
        {
            stride /= dim;
            this->strides.push_back(stride);
        }
    }

    array(const py::buffer_info & buf) : array(buf.shape, static_cast<T *>(buf.ptr)) {}

    template <typename CoordIter, typename = std::enable_if_t<std::is_integral_v<typename CoordIter::value_type>>>
    size_t ravel_index(CoordIter first, CoordIter last) const
    {
        return detail::ravel_index(first, last, this->strides.begin());
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
        return detail::unravel_index(this->strides.begin(), this->strides.end(), index, first);
    }

    T & operator[] (size_t index) {return this->data[index];}
    const T & operator[] (size_t index) const {return this->data[index];}
    iterator begin() {return this->data;}
    iterator end() {return this->data + this->size;}
    const_iterator begin() const {return this->data;}
    const_iterator end() const {return this->data + this->size;}

    strided_iterator<T> line_begin(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t size = this->dims[axis] * this->strides[axis];
        iterator ptr = this->data + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<T>(ptr, this->strides[axis]);
    }

    strided_iterator<const T> line_begin(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t size = this->dims[axis] * this->strides[axis];
        const_iterator ptr = this->data + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<const T>(ptr, this->strides[axis]);
    }

    strided_iterator<T> line_end(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t size = this->dims[axis] * this->strides[axis];
        iterator ptr = this->data + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<T>(ptr + size, this->strides[axis]);
    }

    strided_iterator<const T> line_end(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t size = this->dims[axis] * this->strides[axis];
        const_iterator ptr = this->data + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<const T>(ptr + size, this->strides[axis]);
    }

protected:
    void check_index(size_t axis, size_t index)
    {
        if (axis >= this->ndim || index >= (this->size / this->dims[axis]))
            throw std::out_of_range("index " + std::to_string(index) + " is out of bound for axis "
                                    + std::to_string(axis));
    }
};

/*----------------------------------------------------------------------------*/
/*--------------------------- Extend line modes ------------------------------*/
/*----------------------------------------------------------------------------*/
/*
    EXTEND_CONSTANT: kkkkkkkk|abcd|kkkkkkkk
    EXTEND_NEAREST:  aaaaaaaa|abcd|dddddddd
    EXTEND_MIRROR:   cbabcdcb|abcd|cbabcdcb
    EXTEND_REFLECT:  abcddcba|abcd|dcbaabcd
    EXTEND_WRAP:     abcdabcd|abcd|abcdabcd
*/
enum EXTEND_MODE
{
    EXTEND_CONSTANT = 0,
    EXTEND_NEAREST = 1,
    EXTEND_MIRROR = 2,
    EXTEND_REFLECT = 3,
    EXTEND_WRAP = 4
};

static std::unordered_map<std::string, EXTEND_MODE> const modes = {{"constant", EXTEND_CONSTANT},
                                                                   {"nearest", EXTEND_NEAREST},
                                                                   {"mirror", EXTEND_MIRROR},
                                                                   {"reflect", EXTEND_REFLECT},
                                                                   {"wrap", EXTEND_WRAP}};

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
void extend_line(InputIt first, InputIt last, OutputIt ofirst, OutputIt olast, EXTEND_MODE mode, const T & cval)
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
        case EXTEND_CONSTANT:

            std::fill_n(ofirst, size_before, cval);
            std::fill_n(std::next(ofirst, osize - size_after), size_after, cval);
            break;

        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:

            std::fill_n(ofirst, size_before, *first);
            std::fill_n(std::next(ofirst, osize - size_after), *std::prev(last));
            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:

            std::generate_n(std::reverse_iterator<OutputIt>(std::next(ofirst, size_before)), size_before,
                            mirror_generator<InputIt>(first, last, first));
            std::generate_n(std::next(ofirst, osize - size_after), size_after,
                            mirror_generator<InputIt>(first, last, last).reverse());
            break;

        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:
            
            std::generate_n(std::reverse_iterator<OutputIt>(std::next(ofirst, size_before)), size_before,
                            reflect_generator<InputIt>(first, last, first));
            std::generate_n(std::next(ofirst, osize - size_after), size_after,
                            reflect_generator<InputIt>(first, last, last).reverse());
            break;

        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:
            std::generate_n(std::reverse_iterator<OutputIt>(std::next(ofirst, size_before)), size_before,
                            wrap_generator<InputIt>(first, last, last).reverse());
            std::generate_n(std::next(ofirst, osize - size_after), size_after,
                            wrap_generator<InputIt>(first, last, first));
            break;

        default:
            throw std::invalid_argument("extend_line: invalid EXTEND_MODE argument");
    }
}

template <typename Container, typename T, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
std::optional<T> extend_point(const Container & coord, const array<T> & arr, const array<bool> & mask, EXTEND_MODE mode, const T & cval)
{
    using I = typename Container::value_type;

    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == EXTEND_CONSTANT) return std::optional<T>(cval);

    std::vector<I> close (arr.ndim, I());
    size_t dist;

    switch (mode)
    {
        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.dims[n])) close[n] = arr.dims[n] - 1;
                else if (coord[n] < 0) close[n] = 0;
                else close[n] = coord[n];
            }

            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.dims[n]))
                {
                    close[n] = arr.dims[n] - 1;
                    dist = coord[n] - arr.dims[n] + 1;

                    while (dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n];

                    while (dist-- && close[n] < static_cast<I>(arr.dims[n])) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.dims[n]))
                {
                    close[n] = arr.dims[n] - 1;
                    dist = coord[n] - arr.dims[n];

                    while (dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n] - 1;

                    while (dist-- && close[n] < static_cast<I>(arr.dims[n])) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.dims[n]))
                {
                    close[n] = 0;
                    dist = coord[n] - arr.dims[n];

                    while (dist-- && close[n] < static_cast<I>(arr.dims[n])) close[n]++;
                }
                else if (coord[n] < 0)
                {
                    close[n] = arr.dims[n] - 1;
                    dist = -coord[n] - 1;

                    while (dist-- && close[n] >= 0) close[n]--;
                }
                else close[n] = coord[n];
            }

            break;

        default:
            throw std::invalid_argument("extend_point: invalid EXTEND_MODE argument.");
    }

    size_t index = arr.ravel_index(close.begin(), close.end());

    if (mask[index]) return std::optional<T>(arr[index]);
    else return std::nullopt;
}

/*----------------------------------------------------------------------------*/
/*------------------------------ Binary search -------------------------------*/
/*----------------------------------------------------------------------------*/
// Array search
enum SEARCH_SIDE
{
    SEARCH_LEFT = 0,
    SEARCH_RIGHT = 1
};

/* find idx \el [0, npts], so that base[idx - 1] < key <= base[idx] */
template <class ForwardIt, typename T, class Compare>
ForwardIt searchsorted(const T & value, ForwardIt first, ForwardIt last, SEARCH_SIDE side, Compare comp)
{
    auto npts = std::distance(first, last);
    auto extreme = std::next(first, npts - 1);
    if (comp(value, *first)) return first;
    if (!comp(value, *extreme)) return extreme;

    ForwardIt out;
    switch (side)
    {
        case SEARCH_LEFT:
            out = std::lower_bound(first, last, value, comp);
            break;

        case SEARCH_RIGHT:
            out = std::next(first, std::distance(first, std::upper_bound(first, last, value, comp)) - 1);
            break;

        default:
            throw std::invalid_argument("searchsorted: invalid SEARCH_SIDE argument.");
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

/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangular iterator ---------------------------*/
/*----------------------------------------------------------------------------*/

template <typename T>
struct rect_iterator
{
    size_t ndim;
    size_t size;
    std::vector<T> coord;
    std::vector<size_t> strides;
    T index;

    rect_iterator(size_t ndim, size_t size, std::vector<T> coord, std::vector<size_t> strides, T index)
        : ndim(ndim), size(size), coord(std::move(coord)), strides(std::move(strides)), index(index) {}

    template <typename InputIt, typename = std::enable_if_t<std::is_same_v<typename InputIt::value_type, T>>>
    rect_iterator(InputIt first1, InputIt last1, InputIt first2) : ndim(std::distance(first1, last1)), size(1), index(0)
    {
        for (; first1 != last1; first1++, first2++)
        {
            if (*first2 < *first1) throw std::invalid_argument("the first point is larger than the second");
            this->strides = this->size;
            this->size *= *first2 - *first1;
        }
    }

    rect_iterator & operator++()
    {
        if (!this->is_end())
        {
            this->index++; detail::unravel_index(this->strides.begin(), this->strides.end(), this->index, this->coord.begin());
        }
        return *this;
    }

    bool is_end() const {return this->index >= static_cast<int>(this->size); }
};

}

#endif