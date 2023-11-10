#include "signal_proc.hpp"

namespace cbclib {

template <typename T>
py::array_t<T> binterpolate(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                            std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> grid,
                            py::array_t<T, py::array::c_style | py::array::forcecast> coords, unsigned threads)
{
    auto ndim = grid.size();
    auto ibuf = inp.request();
    if (ndim != static_cast<size_t>(ibuf.ndim))
        throw std::invalid_argument("data number of dimensions (" + std::to_string(ibuf.ndim) + ")" +
                                    " isn't equal to the number of grid arrays (" + std::to_string(ndim) + ")");

    auto cbuf = coords.request();
    check_dimensions("coords", cbuf.ndim - 2, cbuf.shape, cbuf.size / ndim, ndim);

    std::vector<array<T>> gvec;
    for (size_t n = 0; n < ndim; n++)
    {
        auto & arr = gvec.emplace_back(grid[n].request());
        check_dimensions("grid coordinates", arr.ndim - 1, arr.shape, ibuf.shape[ndim - 1 - n]);
    }

    auto carr = array<T>(cbuf);
    auto iarr = array<T>(ibuf);
    auto out = py::array_t<T>(cbuf.shape[0]);
    auto oarr = array<T>(out.request());   

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < carr.shape[0]; i++)
    {
        e.run([&]
        {
            oarr[i] = bilinear(iarr, gvec, std::vector<T>(carr.line_begin(1, i), carr.line_end(1, i)));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T>
py::array_t<T> kr_predict(py::array_t<T, py::array::c_style | py::array::forcecast> y,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x_hat, T sigma,
                          std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> w, unsigned threads)
{
    check_optional("w", y, w, 1.0);
}

PYBIND11_MODULE(signal_proc, m)
{
    using namespace cbclib;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("binterpolate", &binterpolate<double>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);
    m.def("binterpolate", &binterpolate<float>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);

}

}