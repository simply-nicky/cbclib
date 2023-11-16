#ifndef SIGNAL_PROC_
#define SIGNAL_PROC_
#include "array.hpp"

namespace cbclib {

/*----------------------------------------------------------------------------*/
/*------------------------- Bilinear interpolation ---------------------------*/
/*----------------------------------------------------------------------------*/

/* points (integer) follow the convention:      [..., k, j, i], where {i <-> x, j <-> y, k <-> z}
   coordinates (float) follow the convention:   [x, y, z, ...]
 */

template <typename T>
T bilinear(const array<T> & arr, const std::vector<array<T>> & grid, const std::vector<T> & coord)
{
    std::vector<size_t> lbound, ubound;
    std::vector<T> dx;

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = coord.size() - 1 - n;
        // liter is GREATER OR EQUAL
        auto liter = std::lower_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // uiter is GREATER
        auto uiter = std::upper_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // lbound is LESS OR EQUAL
        lbound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), uiter) - 1, 0, grid[index].size - 1));
        // rbound is GREATER OR EQUAL
        ubound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), liter), 0, grid[index].size - 1));
    }

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = coord.size() - 1 - n;
        if (lbound[index] != ubound[index])
        {
            dx.push_back((coord[n] - grid[n][lbound[index]]) / (grid[n][ubound[index]] - grid[n][lbound[index]]));
        }
        else dx.push_back(T());
    }

    T out = T();
    std::vector<size_t> point (coord.size());

    // Iterating over a square around coord
    for (size_t i = 0; i < (1ul << coord.size()); i++)
    {
        T factor = 1.0;
        for (size_t n = 0; n < coord.size(); n++)
        {
            // If the index is odd
            if ((i >> n) & 1)
            {
                point[point.size() - 1 - n] = ubound[ubound.size() - 1 - n];
                factor *= dx[n];
            }
            else
            {
                point[point.size() - 1 - n] = lbound[lbound.size() - 1 - n];
                factor *= 1.0 - dx[n];
            }

        }

        out += factor * arr[arr.ravel_index(point.begin(), point.end())];
    }

    return out;
}

template <typename T>
py::array_t<T> binterpolate(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                            std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> grid,
                            py::array_t<T, py::array::c_style | py::array::forcecast> coords, unsigned threads);

/*----------------------------------------------------------------------------*/
/*---------------------------- Kernel regression -----------------------------*/
/*----------------------------------------------------------------------------*/

namespace detail {

template <typename T>
T gaussian(T x, T sigma) {return exp(-std::pow(x / sigma, 2) / 2) / Constants::M_1_SQRT2PI;}

template <typename T>
T triangular(T x, T sigma) {return std::max(1 - std::abs(x / sigma), T());}

template <typename T>
T parabolic(T x, T sigma) {return T(0.75) * std::max<T>(1 - std::pow(x / sigma, 2), T());}

template <typename T>
T biweight(T x, T sigma) {return 15 / 16 * std::max<T>(std::pow(1 - std::pow(x / sigma, 2), 2), T());}

}

template <typename T>
struct kernels
{
    using kernel = T (*)(T, T);
    using kernel_info = std::tuple<kernel, T>;

    static inline std::map<std::string, kernel_info> registered_kernels = {{"gaussian",   std::make_tuple(detail::gaussian<T>, 3)},
                                                                           {"triangular", std::make_tuple(detail::triangular<T>, 1)},
                                                                           {"parabolic",  std::make_tuple(detail::parabolic<T>, 1)},
                                                                           {"biweight",   std::make_tuple(detail::biweight<T>, 1)}};

    static kernel_info get_kernel(std::string name, bool throw_if_missing = true)
    {
        auto it = registered_kernels.find(name);
        if (it != registered_kernels.end()) return it->second;
        if (throw_if_missing)
            throw std::invalid_argument("kernel is missing for " + name);
        return std::make_tuple(nullptr, T());
    }
};

template <typename T>
py::array_t<T> kr_predict(py::array_t<T, py::array::c_style | py::array::forcecast> y,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x_hat, T sigma, std::string kernel,
                          std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> w, unsigned threads);

}

#endif