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

    threads = (threads > carr.shape[0]) ? carr.shape[0] : threads;

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
                          py::array_t<T, py::array::c_style | py::array::forcecast> x_hat, T sigma, std::string kernel,
                          std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> w, unsigned threads)
{
    check_optional("w", y, w, T(1));

    auto krn = kernels<T>::get_kernel(kernel);

    auto ybuf = y.request(), xbuf = x.request(), xhbuf = x_hat.request();
    auto ndim = xbuf.shape[xbuf.ndim - 1], npts = xbuf.size / ndim;
    check_dimensions("x_hat", xhbuf.ndim - 1, xhbuf.shape, ndim);

    if (ybuf.size != npts)
        throw std::invalid_argument("Number of x points (" + std::to_string(npts) + ") doesn't match to " + 
                                    "the number of y points (" + std::to_string(ybuf.size) + ")");

    auto xarr = array<T>(xbuf);
    auto yarr = array<T>(ybuf);
    auto warr = array<T>(w.value().request());
    auto xharr = array<T>(xhbuf);

    auto out_shape = std::vector<py::ssize_t>(xharr.shape.begin(), std::prev(xharr.shape.end()));
    auto out = py::array_t<T>(out_shape);

    auto oarr = array<T>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > oarr.size) ? oarr.size : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> idxs (npts);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(), [&xarr, ndim](size_t i1, size_t i2){return xarr[i1 * ndim] < xarr[i2 * ndim];});

        #pragma omp for
        for (size_t i = 0; i < oarr.size; i++)
        {
            e.run([&]
            {
                auto xh_vec = std::vector<T>(xharr.line_begin(xharr.ndim - 1, i), xharr.line_end(xharr.ndim - 1, i));

                auto window = idxs;

                for (size_t axis = 0; axis < static_cast<size_t>(ndim); axis++)
                {
                    auto comp_lb = [&xarr, axis, ndim](size_t index, T val){return xarr[index * ndim + axis] < val;};
                    auto comp_ub = [&xarr, axis, ndim](T val, size_t index){return val < xarr[index * ndim + axis];};

                    // begin is LESS OR EQUAL than val
                    auto begin = std::upper_bound(window.begin(), window.end(), xh_vec[axis] - sigma, comp_ub);
                    if (begin != window.begin()) begin = std::prev(begin);

                    // end is GREATER than val
                    auto end = std::lower_bound(window.begin(), window.end(), xh_vec[axis] + sigma, comp_lb);
                    if (end != window.end()) end = std::next(end);

                    if (begin >= end)
                    {
                        window.clear(); break;
                    }
                    else
                    {
                        window = std::vector<size_t>(begin, end);
                        if (axis + 1 < static_cast<size_t>(ndim))
                        {
                            auto less = [&xarr, axis, ndim](size_t i1, size_t i2){return xarr[i1 * ndim + axis + 1] < xarr[i2 * ndim + axis + 1];};
                            std::sort(window.begin(), window.end(), less);
                        }
                    }
                }

                if (window.size())
                {
                    T Y = T(), W = T();
                    for (auto index : window)
                    {
                        T dist = T();
                        for (size_t axis = 0; axis < static_cast<size_t>(ndim); axis++) dist += std::pow(xarr[index * ndim + axis] - xh_vec[axis], 2);
                        T rbf = krn(std::sqrt(dist), sigma);
                        Y += yarr[index] * warr[index] * rbf;
                        W += warr[index] * warr[index] * rbf;
                    }
                    oarr[i] = (W > T()) ? Y / W : T();
                }
                else oarr[i] = T();
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename U>
py::array_t<size_t> local_maxima(py::array_t<T, py::array::c_style | py::array::forcecast> inp, U axis, unsigned threads)
{
    auto ibuf = inp.request();

    sequence<long> seq (axis);
    seq.unwrap(ibuf.ndim);

    for (auto ax : seq)
    {
        if (ibuf.shape[ax] < 3)
            throw std::invalid_argument("The shape along axis " + std::to_string(ax) + "is below 3 (" +
                                        std::to_string(ibuf.shape[ax]) + ")");
    }

    auto iarr = array<T>(ibuf);
    size_t repeats = iarr.size / iarr.shape[seq[0]];
    
    std::vector<size_t> peaks;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> buffer;
        auto add_peak = [&buffer, &iarr](size_t index){iarr.unravel_index(std::back_inserter(buffer), index);};

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                maxima1d(iarr.line_begin(seq[0], i), iarr.line_end(seq[0], i), add_peak, iarr, seq);
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            peaks.insert(peaks.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    if (peaks.size() % iarr.ndim)
        throw std::runtime_error("peaks have invalid size of " + std::to_string(peaks.size()));

    std::array<size_t, 2> out_shape = {peaks.size() / iarr.ndim, iarr.ndim};
    return as_pyarray(std::move(peaks)).reshape(out_shape);
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

    m.def("binterpolate", &binterpolate<float>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);
    m.def("binterpolate", &binterpolate<double>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);

    m.def("kr_predict", &kr_predict<float>, py::arg("y"), py::arg("x"), py::arg("x_hat"), py::arg("sigma"), py::arg("kernel") = "gaussian", py::arg("w") = nullptr, py::arg("num_threads") = 1);
    m.def("kr_predict", &kr_predict<double>, py::arg("y"), py::arg("x"), py::arg("x_hat"), py::arg("sigma"), py::arg("kernel") = "gaussian", py::arg("w") = nullptr, py::arg("num_threads") = 1);

    m.def("local_maxima", &local_maxima<int, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<int, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);

}

}