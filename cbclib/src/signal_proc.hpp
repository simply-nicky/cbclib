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

template <typename T>
py::array_t<T> kr_predict(py::array_t<T, py::array::c_style | py::array::forcecast> y,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x_hat, T sigma, std::string kernel,
                          std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> w, unsigned threads);

template <typename T, typename U>
py::array_t<size_t> local_maxima(py::array_t<T, py::array::c_style | py::array::forcecast> inp, U axis, unsigned threads);

}

#endif