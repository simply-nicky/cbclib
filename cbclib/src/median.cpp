#include "median.hpp"

namespace cbclib {

template <typename T>
void check_mask(const py::array_t<T, py::array::c_style | py::array::forcecast> & inp,
                std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & mask)
{
    py::buffer_info ibuf = inp.request();
    if (!mask)
    {
        mask = py::array_t<bool>(ibuf.shape);
        PyArray_FILLWBYTE(mask.value().ptr(), 1);
    }
    py::buffer_info mbuf = mask.value().request();
    if (!std::equal(mbuf.shape.begin(), mbuf.shape.end(), ibuf.shape.begin()))
        throw std::invalid_argument("mask and inp arrays must have identical shapes");
}

template <typename T, typename U>
py::array_t<T> median(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                      std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                      U axis, unsigned threads)
{
    assert(PyArray_API);

    check_mask(inp, mask);

    sequence<long> seq (axis);
    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_axes(inp);
    mask = seq.swap_axes(mask.value());

    auto ibuf = inp.request();
    auto ax = ibuf.ndim - seq.size();
    auto out_shape = std::vector<py::ssize_t>(ibuf.shape.begin(), std::next(ibuf.shape.begin(), ax));
    auto out = py::array_t<T>(out_shape);

    auto new_shape = out_shape;
    new_shape.push_back(ibuf.size / out.size());
    inp = inp.reshape(new_shape);
    mask = mask.value().reshape(new_shape);

    auto oarr = array<T>(out.request());
    auto iarr = array<T>(inp.request());
    auto marr = array<bool>(mask.value().request());

    py::gil_scoped_release release;

    threads = (threads > oarr.size) ? oarr.size : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<T> buffer;
        std::vector<size_t> idxs (iarr.shape[ax], 0);
        std::iota(idxs.begin(), idxs.end(), 0);

        #pragma omp for
        for (size_t i = 0; i < oarr.size; i++)
        {
            buffer.clear();
            auto miter = marr.line_begin(ax, i);
            auto iiter = iarr.line_begin(ax, i);

            for (auto idx : idxs) if (miter[idx]) buffer.push_back(iiter[idx]);

            if (buffer.size()) oarr[i] = *wirthmedian(buffer.begin(), buffer.end(), std::less<T>());
            else oarr[i] = T();
        }
    }

    py::gil_scoped_acquire acquire;

    return out;
}

template <typename T, typename U>
py::array_t<T> filter_preprocessor(py::array_t<T, py::array::c_style | py::array::forcecast> & inp, std::optional<U> size,
                                   std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & fprint,
                                   std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & mask,
                                   std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & inp_mask)
{
    check_mask(inp, mask);
    if (!inp_mask) inp_mask = mask.value();

    if (!size && !fprint)
        throw std::invalid_argument("size or fprint must be provided");

    auto ibuf = inp.request();
    if (!fprint)
    {
        sequence<size_t> seq (size.value(), ibuf.ndim);
        fprint = py::array_t<bool>(seq.data);
        PyArray_FILLWBYTE(fprint.value().ptr(), 1);
    }
    py::buffer_info fbuf = fprint.value().request();
    if (fbuf.ndim != ibuf.ndim)
        throw std::invalid_argument("fprint must have the same number of dimensions as the input");

    return py::array_t<T>(ibuf.shape);
}

template <typename T, typename U>
py::array_t<T> median_filter(py::array_t<T, py::array::c_style | py::array::forcecast> inp, std::optional<U> size,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> fprint,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> inp_mask,
                             std::string mode, const T & cval, unsigned threads)
{
    assert(PyArray_API);

    auto it = modes.find(mode);
    if (it == modes.end())
        throw std::invalid_argument("invalid mode argument");
    auto m = it->second;

    auto out = filter_preprocessor(inp, size, fprint, mask, inp_mask);

    auto oarr = array<T>(out.request());
    auto iarr = array<T>(inp.request());
    auto marr = array<bool>(mask.value().request());
    auto imarr = array<bool>(inp_mask.value().request());
    auto farr = array<bool>(fprint.value().request());

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        footprint<T> fpt (farr);
        std::vector<long> coord (iarr.ndim, 0);

        #pragma omp for schedule(guided)
        for (size_t i = 0; i < iarr.size; i++)
        {
            if (marr[i])
            {
                iarr.unravel_index(coord.begin(), i);
                fpt.update(coord, iarr, imarr, m, cval);

                if (fpt.data.size()) oarr[i] = *wirthmedian(fpt.data.begin(), fpt.data.end(), std::less<T>());
            }
            else oarr[i] = T();
        }
    }

    py::gil_scoped_acquire acquire;

    return out;
}

template <typename T, typename U>
py::array_t<T> maximum_filter(py::array_t<T, py::array::c_style | py::array::forcecast> inp, std::optional<U> size,
                              std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> fprint,
                              std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                              std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> inp_mask,
                              std::string mode, const T & cval, unsigned threads)
{
    assert(PyArray_API);

    auto it = modes.find(mode);
    if (it == modes.end())
        throw std::invalid_argument("invalid mode argument");
    auto m = it->second;

    auto out = filter_preprocessor(inp, size, fprint, mask, inp_mask);

    auto oarr = array<T>(out.request());
    auto iarr = array<T>(inp.request());
    auto marr = array<bool>(mask.value().request());
    auto imarr = array<bool>(inp_mask.value().request());
    auto farr = array<bool>(fprint.value().request());

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        footprint<T> fpt (farr);
        std::vector<long> coord (iarr.ndim, 0);

        #pragma omp for schedule(guided)
        for (size_t i = 0; i < iarr.size; i++)
        {
            if (marr[i])
            {
                iarr.unravel_index(coord.begin(), i);
                fpt.update(coord, iarr, imarr, m, cval);

                if (fpt.data.size()) oarr[i] = *wirthselect(fpt.data.begin(), fpt.data.end(), fpt.data.size() - 1, std::less<T>());
            }
            else oarr[i] = T();
        }
    }

    py::gil_scoped_acquire acquire;

    return out;
}

template <typename T, typename U>
auto robust_mean(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                 std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                 U axis, double r0, double r1, int n_iter, double lm, unsigned threads) -> py::array_t<std::common_type_t<T, float>>
{
    using D = std::common_type_t<T, float>;
    assert(PyArray_API);

    check_mask(inp, mask);

    sequence<long> seq (axis);
    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_axes(inp);
    mask = seq.swap_axes(mask.value());

    auto ibuf = inp.request();
    auto ax = ibuf.ndim - seq.size();
    auto out_shape = std::vector<py::ssize_t>(ibuf.shape.begin(), std::next(ibuf.shape.begin(), ax));
    auto out = py::array_t<D>(out_shape);

    auto new_shape = out_shape;
    new_shape.push_back(ibuf.size / out.size());
    inp = inp.reshape(new_shape);
    mask = mask.value().reshape(new_shape);

    auto oarr = array<D>(out.request());
    auto iarr = array<T>(inp.request());
    auto marr = array<bool>(mask.value().request());

    py::gil_scoped_release release;

    threads = (threads > oarr.size) ? oarr.size : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<T> buffer;
        std::vector<D> err (iarr.shape[ax]);
        std::vector<size_t> idxs (iarr.shape[ax]);

        size_t j0 = r0 * iarr.shape[ax], j1 = r1 * iarr.shape[ax];
        D mean;

        #pragma omp for
        for (size_t i = 0; i < oarr.size; i++)
        {
            auto iiter = iarr.line_begin(ax, i);
            auto miter = marr.line_begin(ax, i);

            auto get_err = [=, &mean](size_t idx){return miter[idx] * (iiter[idx] - mean) * (iiter[idx] - mean);};

            buffer.clear();
            std::iota(idxs.begin(), idxs.end(), 0);
            for (auto idx : idxs) if (miter[idx]) buffer.push_back(iiter[idx]);

            if (buffer.size()) mean = *wirthmedian(buffer.begin(), buffer.end(), std::less<T>());
            else mean = D();


            for (int n = 0; n < n_iter; n++)
            {
                std::iota(idxs.begin(), idxs.end(), 0);
                std::transform(idxs.begin(), idxs.end(), err.begin(), get_err);
                std::sort(idxs.begin(), idxs.end(), [&err](size_t i1, size_t i2){return err[i1] < err[i2];});

                mean = std::transform_reduce(idxs.begin() + j0, idxs.begin() + j1, D(), std::plus<D>(),
                                             [=](size_t idx){return miter[idx] * iiter[idx];}) / (j1 - j0);
            }

            std::iota(idxs.begin(), idxs.end(), 0);
            std::transform(idxs.begin(), idxs.end(), err.begin(), get_err);
            std::sort(idxs.begin(), idxs.end(), [&err](size_t i1, size_t i2){return err[i1] < err[i2];});

            D cumsum = D(); mean = D(); int count = 0;
            for (size_t j = 0; j < idxs.size(); j++)
            {
                if (lm * cumsum > j * err[idxs[j]]) {mean += miter[idxs[j]] * iiter[idxs[j]]; count ++;}
                cumsum += err[idxs[j]];
            }
            if (count) oarr[i] = mean / count;
            else oarr[i] = D();
        }
    }

    py::gil_scoped_acquire acquire;

    return out;
}

template <typename T, typename U>
auto robust_lsq(py::array_t<T, py::array::c_style | py::array::forcecast> W,
                py::array_t<T, py::array::c_style | py::array::forcecast> y,
                std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                U axis, double r0, double r1, int n_iter, double lm, unsigned threads) -> py::array_t<std::common_type_t<T, float>>
{
    using D = std::common_type_t<T, float>;
    assert(PyArray_API);

    check_mask(y, mask);

    sequence<long> seq (axis);
    seq = seq.unwrap(y.ndim());
    y = seq.swap_axes(y);
    mask = seq.swap_axes(mask.value());

    py::buffer_info Wbuf = W.request();
    py::buffer_info ybuf = y.request();
    auto ax = ybuf.ndim - seq.size();
    if (!std::equal(std::make_reverse_iterator(ybuf.shape.end()),
                    std::make_reverse_iterator(ybuf.shape.begin() + ax),
                    std::make_reverse_iterator(Wbuf.shape.end())))
        throw std::invalid_argument("W and y arrays have incompatible shapes");

    auto new_shape = std::vector<py::ssize_t>(ybuf.shape.begin(), std::next(ybuf.shape.begin(), ax));
    auto repeats = get_size(new_shape.begin(), new_shape.end());
    new_shape.push_back(ybuf.size / repeats);

    y = y.reshape(new_shape);
    mask = mask.value().reshape(new_shape);

    auto nf = Wbuf.size / new_shape[ax];
    W = W.reshape({nf, new_shape[ax]});

    auto out_shape = std::vector<py::ssize_t>(new_shape.begin(), std::prev(new_shape.end()));
    out_shape.push_back(nf);
    auto out = py::array_t<D>(out_shape);

    auto oarr = array<D>(out.request());
    auto Warr = array<T>(W.request());
    auto yarr = array<T>(y.request());
    auto marr = array<bool>(mask.value().request());

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    auto get_x = [](std::pair<T, T> p) -> D {return (p.second > T()) ? static_cast<D>(p.first) / p.second : D();};
    auto sum_pairs = [](std::pair<T, T> p1, std::pair<T, T> p2){return std::make_pair(p1.first + p2.first, p1.second + p2.second);};

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<T, T>> sums (oarr.shape[ax]);

        std::vector<D> err (yarr.shape[ax]);
        std::vector<size_t> idxs (yarr.shape[ax]);

        size_t j0 = r0 * yarr.shape[ax], j1 = r1 * yarr.shape[ax];

        #pragma omp for
        for (size_t i = 0; i < repeats; i++)
        {
            auto yiter = yarr.line_begin(ax, i);
            auto miter = marr.line_begin(ax, i);

            auto get_err = [=, &sums, &Warr](size_t idx) -> D
            {
                D err = miter[idx] * yiter[idx];
                auto Witer = Warr.line_begin(0, idx);
                for (size_t k = 0; k < sums.size(); k++) err -= Witer[k] * get_x(sums[k]);
                return miter[idx] * err * err;
            };
            auto get_pair = [=, &Warr](size_t k)
            {
                auto Witer = Warr.line_begin(Warr.ndim - 1, k);
                auto f = [=](size_t idx)
                {
                    return std::make_pair(miter[idx] * yiter[idx] * Witer[idx], Witer[idx] * Witer[idx]);
                };
                return f;
            };

            std::iota(idxs.begin(), idxs.end(), 0);
            for (size_t k = 0; k < sums.size(); k++)
            {
                sums[k] = std::transform_reduce(idxs.begin(), idxs.end(), std::pair<T, T>(), sum_pairs, get_pair(k));
            }

            for (int n = 0; n < n_iter; n++)
            {
                std::iota(idxs.begin(), idxs.end(), 0);
                std::transform(idxs.begin(), idxs.end(), err.begin(), get_err);
                std::sort(idxs.begin(), idxs.end(), [&err](size_t i1, size_t i2){return err[i1] < err[i2];});

                for (size_t k = 0; k < sums.size(); k++)
                {
                    sums[k] = std::transform_reduce(idxs.begin() + j0, idxs.begin() + j1, std::pair<T, T>(), sum_pairs, get_pair(k));
                }
            }

            std::iota(idxs.begin(), idxs.end(), 0);
            std::transform(idxs.begin(), idxs.end(), err.begin(), get_err);
            std::sort(idxs.begin(), idxs.end(), [&err](size_t i1, size_t i2){return err[i1] < err[i2];});

            D cumsum = D();
            std::fill(sums.begin(), sums.end(), std::pair<T, T>());
            for (size_t j = 0; j < idxs.size(); j++)
            {
                if (lm * cumsum > j * err[idxs[j]])
                {
                    auto Witer = Warr.line_begin(0, idxs[j]);
                    for (size_t k = 0; k < sums.size(); k++)
                    {
                        sums[k].first += miter[idxs[j]] * yiter[idxs[j]] * Witer[k];
                        sums[k].second += Witer[k] * Witer[k];
                    }
                }
                cumsum += err[idxs[j]];
            }
            
            std::transform(sums.begin(), sums.end(), oarr.line_begin(ax, i), get_x);
        }
    }

    py::gil_scoped_acquire acquire;

    return out;

}

}

PYBIND11_MODULE(median, m)
{
    using namespace cbclib;

    try
    {
        import_numpy();
    }
    catch(const py::error_already_set & e)
    {
        return;
    }

    m.def("median", &median<double, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<double, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<float, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<float, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<int, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<int, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<long, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<long, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<size_t, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<size_t, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);

    m.def("median_filter", &median_filter<double, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<double, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<float, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<float, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<int, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<int, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<long, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<long, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<size_t, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<size_t, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);

    m.def("maximum_filter", &maximum_filter<double, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<double, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<float, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<float, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<int, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<int, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<long, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<long, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<size_t, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("maximum_filter", &maximum_filter<size_t, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);

    m.def("robust_mean", &robust_mean<double, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<double, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<float, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<float, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<int, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<int, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<long, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<long, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<size_t, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<size_t, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);

    m.def("robust_lsq", &robust_lsq<double, int>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<double, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<float, int>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<float, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<int, int>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<int, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<long, int>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<long, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<size_t, int>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<size_t, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}