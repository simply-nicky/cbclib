#ifndef MEDIAN_
#define MEDIAN_
#include "array.hpp"

namespace cbclib {

namespace detail{

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt mirror(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = mirror(*first, *min, *max);
    }
    return d_first;
}

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt reflect(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = reflect(*first, *min, *max);
    }
    return d_first;
}

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt wrap(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = wrap(*first, *min, *max);
    }
    return d_first;
}

}

template <typename RandomIt, typename Compare>
double median_1d(RandomIt first, RandomIt last, Compare comp)
{
    auto n = std::distance(first, last);
    if (n & 1) return *wirthselect(first, last, n / 2, comp);
    else
    {
        double lo = *wirthselect(first, last, n / 2 - 1, comp);
        double hi = *wirthselect(first, last, n / 2, comp);
        return (lo + hi) / 2;
    }
}

template <typename Container, typename T, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
T extend_point(const Container & coord, const array<T> & arr, extend mode, const T & cval)
{
    using I = typename Container::value_type;

    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == extend::constant) return cval;

    std::vector<I> close;
    std::vector<I> min (arr.ndim, I());

    switch (mode)
    {
        /* aaaaaaaa|abcd|dddddddd */
        case extend::nearest:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.shape[n])) close.push_back(arr.shape[n] - 1);
                else if (coord[n] < I()) close.push_back(I());
                else close.push_back(coord[n]);
            }

            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case extend::mirror:

            detail::mirror(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape.begin());

            break;

        /* abcddcba|abcd|dcbaabcd */
        case extend::reflect:

            detail::reflect(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape.begin());

            break;

        /* abcdabcd|abcd|abcdabcd */
        case extend::wrap:

            detail::wrap(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape.begin());

            break;

        default:
            throw std::invalid_argument("Invalid extend argument: " + std::to_string(static_cast<int>(mode)));
    }

    return arr.at(close.begin(), close.end());
}

template <typename T>
struct footprint
{
    size_t ndim;
    size_t npts;
    std::vector<std::vector<long>> offsets;
    std::vector<std::vector<long>> coords;
    std::vector<T> data;

    footprint(size_t ndim, size_t npts, std::vector<std::vector<long>> offsets, std::vector<std::vector<long>> coords)
        : ndim(ndim), npts(npts), offsets(std::move(offsets)), coords(std::move(coords)) {}

    footprint(const array<bool> & fmask) : ndim(fmask.ndim)
    {
        auto fiter = fmask.begin();
        for (size_t i = 0; fiter != fmask.end(); ++fiter, ++i)
        {
            if (*fiter)
            {
                std::vector<long> coord;
                fmask.unravel_index(std::back_inserter(coord), i);
                auto & offset = offsets.emplace_back();
                std::transform(coord.begin(), coord.end(), fmask.shape.begin(), std::back_inserter(offset),
                               [](long crd, size_t dim){return crd - dim / 2;});
            }
        }

        npts = offsets.size();
        coords = std::vector<std::vector<long>>(npts, std::vector<long>(ndim));
        if (npts == 0) throw std::runtime_error("zero number of points in a footprint.");
    }

    template <typename Container, typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, long>>>
    footprint & update(const Container & coord, const array<T> & arr, extend mode, const T & cval)
    {
        data.clear();

        for (size_t i = 0; i < npts; i++)
        {
            bool extend = false;

            for (size_t n = 0; n < ndim; n++)
            {
                coords[i][n] = coord[n] + offsets[i][n];
                extend |= (coords[i][n] >= static_cast<long>(arr.shape[n])) || (coords[i][n] < 0);
            }

            if (extend)
            {
                auto val = extend_point(coords[i], arr, mode, cval);
                data.push_back(val);
            }
            else data.push_back(arr.at(coords[i]));
        }

        return *this;
    }
};

}

#endif
