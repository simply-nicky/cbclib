#ifndef GEOMETRY_
#define GEOMETRY_
#include "include.hpp"
#include "kd_tree.hpp"

namespace cbclib {

template<typename F, typename = std::enable_if_t<std::is_floating_point<F>::value>>
py::array_t<F> euler_angles(py::array_t<F, py::array::c_style | py::array::forcecast> rmats, unsigned threads);

template<typename F, typename = std::enable_if_t<std::is_floating_point<F>::value>>
py::array_t<F> euler_matrix(py::array_t<F, py::array::c_style | py::array::forcecast> angles, unsigned threads);

template<typename F, typename = std::enable_if_t<std::is_floating_point<F>::value>>
py::array_t<F> tilt_angles(py::array_t<F, py::array::c_style | py::array::forcecast> rmats, unsigned threads);

template<typename F, typename = std::enable_if_t<std::is_floating_point<F>::value>>
py::array_t<F> tilt_matrix(py::array_t<F, py::array::c_style | py::array::forcecast> angles, unsigned threads);

template<typename F, typename I, typename = std::enable_if_t<std::is_floating_point<F>::value>>
py::array_t<F> det_to_k(py::array_t<F, py::array::c_style | py::array::forcecast> x,
                        py::array_t<F, py::array::c_style | py::array::forcecast> y,
                        py::array_t<F, py::array::c_style | py::array::forcecast> src,
                        std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
                        unsigned threads);

template<typename F, typename I, typename = std::enable_if_t<std::is_floating_point<F>::value>>
auto k_to_det(py::array_t<F, py::array::c_style | py::array::forcecast> karr,
              py::array_t<F, py::array::c_style | py::array::forcecast> src,
              std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
              unsigned threads) -> std::tuple<py::array_t<F>, py::array_t<F>>;

template<typename F, typename I, typename = std::enable_if_t<std::is_floating_point<F>::value>>
py::array_t<F> k_to_smp(py::array_t<F, py::array::c_style | py::array::forcecast> karr,
                        py::array_t<F, py::array::c_style | py::array::forcecast> z, std::array<F, 3> src,
                        std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
                        unsigned threads);

template<typename F, typename I, typename = std::enable_if_t<std::is_floating_point<F>::value>>
auto source_lines(py::array_t<I, py::array::c_style | py::array::forcecast> hkl,
                  py::array_t<F, py::array::c_style | py::array::forcecast> basis,
                  std::array<F, 2> kmin, std::array<F, 2> kmax,
                  std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> hidxs,
                  std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> bidxs,
                  unsigned threads) -> std::tuple<py::array_t<F>, py::array_t<bool>>;

template<typename F, typename I, typename = std::enable_if_t<std::is_floating_point<F>::value>>
py::array_t<F> rotate_vec(py::array_t<F, py::array::c_style | py::array::forcecast> vecs,
                          py::array_t<F, py::array::c_style | py::array::forcecast> rmats,
                          std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
                          unsigned threads);

}

#endif