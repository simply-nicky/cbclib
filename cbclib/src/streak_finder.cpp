#include "streak_finder.hpp"

namespace cbclib {

template <typename T>
std::array<T, 4> test(std::array<int, 4> pts, py::array_t<T> data, int radius, int rank)
{
    auto [x0, y0, x1, y1] = pts;
    Pattern<T> pattern {data, {radius, rank}};

    Streak<T> streak {pattern.get_pset(x0, y0)};
    streak.insert(pattern.get_pset(x1, y1));
    streak.update_line();
    return streak.line.to_array();
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

    py::class_<Structure>(m, "Structure", py::dynamic_attr())
        .def(py::init<int, int>())
        .def_readwrite("radius", &Structure::radius)
        .def_readwrite("rank", &Structure::rank)
        .def_readwrite("idxs", &Structure::idxs)
        .def("__repr__", &Structure::info);

    py::class_<DetState>(m, "DetState", py::dynamic_attr())
        .def(py::init<py::array_t<float>, size_t, float>())
        .def(py::init<py::array_t<double>, size_t, double>())
        .def_readwrite("peaks", &DetState::peaks)
        .def_readwrite("used", &DetState::used)
        .def("filter",
            [](DetState & state, py::array_t<float> data, Structure s, float vmin, size_t npts)
            {
                state.filter(array<float>(data.request()), s, vmin, npts);
                return state;
            })
        .def("filter",
            [](DetState & state, py::array_t<double> data, Structure s, double vmin, size_t npts)
            {
                state.filter(array<double>(data.request()), s, vmin, npts);
                return state;
            })
        .def("__repr__", &DetState::info);

    m.def("test_line", &test<double>, py::arg("pts"), py::arg("data"), py::arg("xshifts"), py::arg("yshifts"));
}