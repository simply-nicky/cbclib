#include "streak_finder.hpp"

namespace cbclib {

template <typename T>
std::array<T, 4> test_line(std::array<int, 4> pts, py::array_t<T> data, py::array_t<bool> mask, Structure srt)
{
    auto [x0, y0, x1, y1] = pts;
    Pattern<T> pattern {data, mask, srt};

    Streak<T> streak {pattern.get_pset(x0, y0)};
    streak.insert(pattern.get_pset(x1, y1));
    streak.update_line(T());
    return streak.line;
}

template <typename T>
std::array<T, 4> test_grow(int x, int y, py::array_t<T> data, py::array_t<bool> mask, Structure srt,
                           T xtol, T vmin, unsigned max_iter, unsigned lookahead)
{
    Pattern<T> pattern {data, mask, srt};

    return pattern.get_streak(x, y, xtol, vmin, max_iter, lookahead).line;
}

template <typename T>
std::vector<std::array<T, 4>> detect_streaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure,
                                             T xtol, T vmin, T log_eps, unsigned max_iter, unsigned lookahead, size_t min_size)
{
    Pattern<T> pattern {data, mask, std::move(structure)};
    return pattern.find_streaks(std::move(peaks), xtol, vmin, log_eps, max_iter, lookahead, min_size);
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
                peaks.filter(array<float>(data.request()), array<bool>(mask.request()), s, vmin, npts);
                return peaks;
            },
            py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def("filter",
            [](Peaks & peaks, py::array_t<double> data, py::array_t<bool> mask, Structure s, double vmin, size_t npts)
            {
                peaks.filter(array<double>(data.request()), array<bool>(mask.request()), s, vmin, npts);
                return peaks;
            },
            py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def("mask",
            [](Peaks & peaks, py::array_t<bool> mask)
            {
                peaks.mask(array<bool>(mask.request()));
                return peaks;
            },
            py::arg("mask"))
        .def_property("points",
            [](Peaks & peaks)
            {
                std::set<Point<int>> pts;
                pts.insert(peaks.points.begin(), peaks.points.end());
                return Points{std::move(pts)};
            }, nullptr, py::keep_alive<0, 1>())
        .def("__repr__", &Peaks::info);

    m.def("test_line", &test_line<double>, py::arg("pts"), py::arg("data"), py::arg("mask"), py::arg("structure"));
    m.def("test_grow", &test_grow<double>, py::arg("x"), py::arg("y"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("max_iter"), py::arg("lookahead"));

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=0.0, py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("log_eps")=0.0, py::arg("max_iter")=100, py::arg("lookahead")=1, py::arg("min_size")=5);
}