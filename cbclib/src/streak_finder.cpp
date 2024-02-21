#include "streak_finder.hpp"

namespace cbclib {

template <typename T>
std::array<T, 4> test_line(std::array<int, 4> pts, py::array_t<T> data, py::array_t<bool> mask, Structure srt)
{
    auto [x0, y0, x1, y1] = pts;
    Pattern<T> pattern {data, mask, srt};

    Streak<T> streak {pattern.get_pset(x0, y0)};
    streak.insert(pattern.get_pset(x1, y1));
    return streak.pixels.get_line();
}

template <typename T>
std::array<T, 4> test_grow(size_t index, Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure srt,
                           T xtol, T vmin, unsigned max_iter, unsigned lookahead)
{
    Pattern<T> pattern {data, mask, srt};
    auto seed = pattern.get_pset(peaks.points[index].x(), peaks.points[index].y());

    return pattern.get_streak(std::move(seed), peaks, xtol, vmin, max_iter, lookahead).pixels.get_line();
}

template <typename T>
std::vector<Peaks> detect_peaks(py::array_t<T> data, py::array_t<bool> mask, size_t radius, T vmin, std::optional<std::tuple<size_t, size_t>> ax, unsigned threads)
{
    sequence<size_t> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_axes(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    check_shape(darr.shape, [](const std::vector<size_t> & shape){return shape.size() < 2;});

    size_t repeats = std::reduce(darr.shape.begin(), std::prev(darr.shape.end(), 2), 1, std::multiplies());

    std::vector<Peaks> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<Peaks> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                buffer.emplace_back(darr.slice(i, axes), marr, radius, vmin);
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            result.insert(result.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

template <typename T>
auto filter_peaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
                  std::optional<std::tuple<size_t, size_t>> ax, unsigned threads)
{
    return peaks.filter(array<T>(data.request()), array<bool>(mask.request()), structure, vmin, npts);
}

template <typename T>
auto filter_peaks_vec(std::vector<Peaks> peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
                      std::optional<std::tuple<size_t, size_t>> ax, unsigned threads)
{
    sequence<size_t> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_axes(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    check_shape(darr.shape, [](const std::vector<size_t> & shape){return shape.size() < 3;});

    size_t repeats = std::reduce(darr.shape.begin(), std::prev(darr.shape.end(), 2), 1, std::multiplies());
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    std::vector<Peaks> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<Peaks> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                buffer.emplace_back(peaks[i].filter(darr.slice(i, axes), marr, structure, vmin, npts));
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            result.insert(result.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

template <typename T>
auto detect_streaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> m, Structure structure, T xtol, T vmin, T log_eps,
                    unsigned max_iter, unsigned lookahead, size_t min_size, std::optional<std::tuple<size_t, size_t>> ax, unsigned threads)
{
    py::array_t<bool> mask {m.request()};
    Pattern<T> pattern {data, mask, std::move(structure)};
    return pattern.find_streaks(std::move(peaks), xtol, vmin, log_eps, max_iter, lookahead, min_size);
}

template <typename T>
auto detect_streaks_vec(std::vector<Peaks> peaks, py::array_t<T> data, py::array_t<bool> m, Structure structure, T xtol, T vmin,
                        T log_eps, unsigned max_iter, unsigned lookahead, size_t min_size, std::optional<std::tuple<size_t, size_t>> ax,
                        unsigned threads)
{
    py::array_t<bool> mask {m.request()};
    sequence<size_t> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_axes(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    check_shape(darr.shape, [](const std::vector<size_t> & shape){return shape.size() < 3;});

    size_t repeats = std::reduce(darr.shape.begin(), std::prev(darr.shape.end(), 2), 1, std::multiplies());
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    std::vector<std::vector<std::array<T, 4>>> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::vector<std::array<T, 4>>> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                Pattern<T> pattern {darr.slice(i, axes), marr, structure};
                buffer.emplace_back(pattern.find_streaks(std::move(peaks[i]), xtol, vmin, log_eps, max_iter, lookahead, min_size));
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            result.insert(result.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

}

PYBIND11_MODULE(streak_finder, m)
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

    py::class_<Peaks>(m, "Peaks")
        .def(py::init<py::array_t<float>, py::array_t<bool>, size_t, float>(), py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"))
        .def(py::init<py::array_t<double>, py::array_t<bool>, size_t, double>(), py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"))
        .def("filter",
            [](Peaks & peaks, py::array_t<float> data, py::array_t<bool> mask, Structure s, float vmin, size_t npts)
            {
                return peaks.filter(array<float>(data.request()), array<bool>(mask.request()), s, vmin, npts);
            },
            py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def("filter",
            [](Peaks & peaks, py::array_t<double> data, py::array_t<bool> mask, Structure s, double vmin, size_t npts)
            {
                return peaks.filter(array<double>(data.request()), array<bool>(mask.request()), s, vmin, npts);
            },
            py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def("mask",
            [](Peaks & peaks, py::array_t<bool> mask)
            {
                return peaks.mask(array<bool>(mask.request()));
            },
            py::arg("mask"))
        .def("sort",
            [](Peaks & peaks, py::array_t<float> data)
            {
                return peaks.sort(array<float>(data.request()));
            }, py::arg("data"))
        .def("sort",
            [](Peaks & peaks, py::array_t<double> data)
            {
                return peaks.sort(array<double>(data.request()));
            }, py::arg("data"))
        .def_property("size", [](const Peaks & peaks){return peaks.points.size();}, nullptr, py::keep_alive<0, 1>())
        .def_property("x", [](const Peaks & peaks){return peaks.x();}, nullptr, py::keep_alive<0, 1>())
        .def_property("y", [](const Peaks & peaks){return peaks.y();}, nullptr, py::keep_alive<0, 1>())
        .def("__repr__", &Peaks::info);

    m.def("test_line", &test_line<double>, py::arg("pts"), py::arg("data"), py::arg("mask"), py::arg("structure"));
    m.def("test_grow", &test_grow<double>, py::arg("index"), py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("max_iter"), py::arg("lookahead"));

    m.def("detect_peaks", &detect_peaks<double>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_peaks", &detect_peaks<float>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("filter_peaks", &filter_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks_vec<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks_vec<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=0.0, py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=0.0, py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks_vec<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=0.0, py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks_vec<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=0.0, py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
}