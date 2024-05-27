#include "streak_finder.hpp"
#include "zip.hpp"

namespace cbclib {

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

    if (darr.shape.size() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.shape.size()) + " < 2)", darr.shape);

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
auto filter_peaks(std::vector<Peaks> peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
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

    if (darr.shape.size() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.shape.size()) + " < 3)", darr.shape);

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
auto detect_streaks(std::vector<Peaks> peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T xtol, T vmin,
                    unsigned min_size, unsigned lookahead, unsigned nfa, std::optional<std::tuple<size_t, size_t>> ax,
                    unsigned threads)
{
    if (xtol >= structure.rank + 0.5)
        throw std::invalid_argument("xtol (" + std::to_string(xtol) + ") must be lower than the rank of the structure (" +
                                    std::to_string(structure.rank) + ")");

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

    if (darr.shape.size() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.shape.size()) + " < 2)", darr.shape);

    size_t repeats = std::reduce(darr.shape.begin(), std::prev(darr.shape.end(), 2), 1, std::multiplies());
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    std::vector<StreakFinderResult<T>> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        StreakFinder finder {structure, min_size, lookahead, nfa};
        std::vector<StreakFinderResult<T>> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                buffer.emplace_back(finder.detect_streaks(StreakFinderResult(darr.slice(i, axes), marr), darr.slice(i, axes), std::move(peaks[i]), xtol, vmin));
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

template<typename T>
void declare_streak_finder_result(py::module & m, const std::string & typestr)
{
    py::class_<StreakFinderResult<T>>(m, (std::string("StreakFinderResult") + typestr).c_str())
        .def(py::init([](py::array_t<T> data, py::array_t<bool> mask)
        {
            return StreakFinderResult<T>(array<T>(data.request()), array<bool>(mask.request()));
        }), py::arg("data"), py::arg("mask"))
        .def("probability", [](const StreakFinderResult<T> & result, py::array_t<T> data, T vmin)
        {
            return result.probability(array<T>(data.request()), vmin);
        }, py::arg("data"), py::arg("vmin"))
        .def("p_value", [](const StreakFinderResult<T> & result, int index, T xtol, T vmin, T p)
        {
            auto iter = result.streaks.find(index);
            if (iter == result.streaks.end())
                throw std::out_of_range("No streak with index " + std::to_string(index) + "in the result");
            return result.p_value(iter, xtol, vmin, p);
        }, py::arg("index"), py::arg("xtol"), py::arg("vmin"), py::arg("probability"))
        .def("get_streak", [](const StreakFinderResult<T> & result, int index)
        {
            using integer_type = typename StreakFinderResult<T>::integer_type;
            auto iter = result.streaks.find(index);
            if (iter == result.streaks.end())
                throw std::out_of_range("No streak with index " + std::to_string(index) + "in the result");

            std::set<std::tuple<integer_type, integer_type, T>> pset;
            for (auto [pt, val] : iter->second.pixels.pset) pset.emplace_hint(pset.end(), pt.x(), pt.y(), val);
            std::map<T, std::array<T, 2>> points;
            for (auto [dist, pt] : iter->second.points) points.emplace_hint(points.end(), dist, pt.to_array());
            std::map<integer_type, std::array<integer_type, 2>> centers;
            for (auto [dist, ctr] : iter->second.centers) centers.emplace_hint(centers.end(), dist, ctr.to_array());
            return std::make_tuple(pset, points, centers, iter->second.line().to_array());
        }, py::arg("index"))
        .def_property("mask", [](const StreakFinderResult<T> & result){return to_pyarray(result.mask, result.mask.shape);}, nullptr)
        .def_property("streaks", [](const StreakFinderResult<T> & result)
        {
            std::map<int, std::array<T, 4>> lines;
            for (auto [index, streak] : result.streaks) lines.emplace_hint(lines.end(), index, streak.line().to_array());
            return lines;
        }, nullptr)
        .def_readonly("idxs", &StreakFinderResult<T>::idxs)
        .def("__repr__", &StreakFinderResult<T>::info);
}

}

PYBIND11_MODULE(streak_finder, m)
{
    using namespace cbclib;
    using integer_type = typename point_t::value_type;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    py::class_<Peaks>(m, "Peaks")
        .def(py::init([](std::vector<integer_type> xvec, std::vector<integer_type> yvec)
        {
            Peaks::container_type points;
            for (auto [x, y] : zip::zip(xvec, yvec)) points.emplace_back(x, y);
            return Peaks(std::move(points));
        }), py::arg("x"), py::arg("y"))
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
        .def("find_nearest",
            [](const Peaks & peaks, integer_type x, integer_type y)
            {
                auto [iter, dist_sq] = peaks.tree.find_nearest(point_t{x, y});
                return std::make_tuple(iter->point().to_array(), std::sqrt(dist_sq));
            }, py::arg("x"), py::arg("y"))
        .def("find_nearest",
            [](const Peaks & peaks, double x, double y)
            {
                auto [iter, dist_sq] = peaks.tree.find_nearest(std::array<double, 2>{x, y});
                return std::make_tuple(iter->point().to_array(), std::sqrt(dist_sq));
            }, py::arg("x"), py::arg("y"))
        .def("find_nearest",
            [](const Peaks & peaks, float x, float y)
            {
                auto [iter, dist_sq] = peaks.tree.find_nearest(std::array<float, 2>{x, y});
                return std::make_tuple(iter->point().to_array(), std::sqrt(dist_sq));
            }, py::arg("x"), py::arg("y"))
        .def("find_range",
            [](const Peaks & peaks, integer_type x, integer_type y, double range)
            {
                auto stack = peaks.tree.find_range(point_t{x, y}, integer_type(range * range));
                std::vector<std::tuple<std::array<integer_type, 2>, double>> result;
                for (auto [iter, dist_sq] : stack) result.emplace_back(iter->point().to_array(), std::sqrt(dist_sq));
                return result;
            }, py::arg("x"), py::arg("y"), py::arg("range"))
        .def("find_range",
            [](const Peaks & peaks, double x, double y, double range)
            {
                auto stack = peaks.tree.find_range(std::array<double, 2>{x, y}, range * range);
                std::vector<std::tuple<std::array<integer_type, 2>, double>> result;
                for (auto [iter, dist_sq] : stack) result.emplace_back(iter->point().to_array(), std::sqrt(dist_sq));
                return result;
            }, py::arg("x"), py::arg("y"), py::arg("range"))
        .def("find_range",
            [](const Peaks & peaks, float x, float y, float range)
            {
                auto stack = peaks.tree.find_range(std::array<float, 2>{x, y}, range * range);
                std::vector<std::tuple<std::array<integer_type, 2>, double>> result;
                for (auto [iter, dist_sq] : stack) result.emplace_back(iter->point().to_array(), std::sqrt(dist_sq));
                return result;
            }, py::arg("x"), py::arg("y"), py::arg("range"))
        .def("mask",
            [](Peaks & peaks, py::array_t<bool> mask)
            {
                return peaks.mask([m = array<bool>(mask.request())](const point_t & point){return m.at(point.coordinate());});
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

    declare_streak_finder_result<double>(m, "Double");
    declare_streak_finder_result<float>(m, "Float");

    py::class_<StreakFinder>(m, "StreakFinder")
        .def(py::init<Structure, unsigned, unsigned, unsigned>(), py::arg("structure"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0)
        .def("detect_streaks", [](StreakFinder & finder, py::array_t<double> data, py::array_t<bool> mask, Peaks peaks, double xtol, double vmin)
        {
            array<double> darr (data.request());
            array<bool> marr (mask.request());
            return finder.detect_streaks(StreakFinderResult(darr, marr), darr, peaks, xtol, vmin);
        }, py::arg("data"), py::arg("mask"), py::arg("peaks"), py::arg("xtol"), py::arg("vmin"))
        .def("detect_streaks", [](StreakFinder & finder, py::array_t<float> data, py::array_t<bool> mask, Peaks peaks, float xtol, float vmin)
        {
            array<float> darr (data.request());
            array<bool> marr (mask.request());
            return finder.detect_streaks(StreakFinderResult(darr, marr), darr, peaks, xtol, vmin);
        }, py::arg("data"), py::arg("mask"), py::arg("peaks"), py::arg("xtol"), py::arg("vmin"))
        .def_readwrite("lookahead", &StreakFinder::lookahead)
        .def_readwrite("nfa", &StreakFinder::nfa)
        .def_readwrite("structure", &StreakFinder::structure);

    m.def("detect_peaks", &detect_peaks<double>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_peaks", &detect_peaks<float>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("filter_peaks", &filter_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
}
