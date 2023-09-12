#ifndef MEDIAN_
#define MEDIAN_
#include "array.hpp"

namespace cbclib {

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
        for (auto fiter = fmask.begin(); fiter != fmask.end(); fiter++)
        {
            if (*fiter)
            {
                std::vector<long> coord;
                fmask.unravel_index(std::back_inserter(coord), std::distance(fmask.begin(), fiter));
                auto & offset = this->offsets.emplace_back();
                std::transform(coord.begin(), coord.end(), fmask.shape.begin(), std::back_inserter(offset),
                               [](long crd, size_t dim){return crd - dim / 2;});
            }
        }

        this->npts = this->offsets.size();
        this->coords = std::vector<std::vector<long>>(npts, std::vector<long>(ndim));
        if (this->npts == 0) throw std::runtime_error("zero number of points in a footprint.");
    }

    template <typename Container, typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, long>>>
    footprint & update(const Container & coord, const array<T> & arr, const array<bool> & mask, extend mode, const T & cval)
    {
        this->data.clear();

        for (size_t i = 0; i < this->npts; i++)
        {
            bool extend = false;

            for (size_t n = 0; n < this->ndim; n++)
            {
                this->coords[i][n] = coord[n] + this->offsets[i][n];
                extend |= (this->coords[i][n] >= static_cast<long>(arr.shape[n])) || (this->coords[i][n] < 0);
            }

            if (extend)
            {
                auto val = extend_point(this->coords[i], arr, mask, mode, cval);
                if (val) this->data.push_back(val.value());
            }
            else
            {
                size_t index = arr.ravel_index(this->coords[i].begin(), this->coords[i].end());
                if (mask[index]) this->data.push_back(arr[index]);
            }
        }

        return *this;
    }
};

template <typename T, typename U>
py::array_t<T> median(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                      std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                      U axis, unsigned threads);

template <typename T, typename U>
py::array_t<T> median_filter(py::array_t<T, py::array::c_style | py::array::forcecast> inp, std::optional<U> size,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> footprint,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> inp_mask,
                             std::string mode, const T & cval, unsigned threads);

template <typename T, typename U>
py::array_t<T> maximum_filter(py::array_t<T, py::array::c_style | py::array::forcecast> inp, std::optional<U> size,
                              std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> fprint,
                              std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                              std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> inp_mask,
                              std::string mode, const T & cval, unsigned threads);

template <typename T, typename U>
auto robust_mean(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                 std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                 U axis, double r0, double r1, int n_iter, double lm, unsigned threads) -> py::array_t<std::common_type_t<T, float>>;

template <typename T, typename U>
auto robust_lsq(py::array_t<T, py::array::c_style | py::array::forcecast> W,
                py::array_t<T, py::array::c_style | py::array::forcecast> y,
                std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                U axis, double r0, double r1, int n_iter, double lm, unsigned threads) -> py::array_t<std::common_type_t<T, float>>;

}

#endif