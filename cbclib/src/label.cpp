#include "label.hpp"
#include "zip.hpp"

namespace cbclib {

template <typename T>
Regions label2d(const array<T> & mask, const Structure & str, size_t npts)
{
    std::vector<Points> regions;
    std::vector<unsigned char> used (mask.size, false);

    for (size_t idx = 0; idx < mask.size; idx++)
    {
        if (mask[idx] && !used[idx])
        {
            int y = mask.index_along_dim(idx, 0);
            int x = mask.index_along_dim(idx, 1);
            Points points (Points::point_type{x, y}, mask, str);

            for (auto pt : *points)
            {
                used[mask.ravel_index(pt.coordinate())] = true;
            }

            if (points->size() > npts) regions.emplace_back(std::move(points));
        }
    }

    return Regions{std::array<size_t, 2>{mask.shape[0], mask.shape[1]}, std::move(regions)};
}

template <typename T>
auto label(py::array_t<bool> mask, Structure structure, size_t npts, std::optional<std::tuple<size_t, size_t>> ax, unsigned threads)
{
    sequence<size_t> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(mask.ndim());
    }
    else axes = {mask.ndim() - 2, mask.ndim() - 1};

    mask = axes.swap_axes(mask);
    array<bool> marr {mask.request()};

    check_shape(marr.shape, [](const std::vector<size_t> & shape){return shape.size() < 2;});

    size_t repeats = std::reduce(marr.shape.begin(), std::next(marr.shape.begin(), marr.ndim - 2), 1, std::multiplies());

    std::vector<Regions> result;
    
    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<Regions> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            buffer.emplace_back(label2d(marr.slice(i, axes), structure, npts));
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

PYBIND11_MODULE(label, m)
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

    py::class_<Points>(m, "Points")
        .def(py::init([](std::vector<int> xvec, std::vector<int> yvec)
        {
            std::set<Points::point_type> points;
            for (auto [x, y] : zip::zip(xvec, yvec)) points.insert({x, y});
            return Points(std::move(points));
        }), py::arg("x"), py::arg("y"))
        .def_property("size", [](const Points & points){return points.points.size();}, nullptr, py::keep_alive<0, 1>())
        .def_property("x", [](const Points & points){return points.x();}, nullptr, py::keep_alive<0, 1>())
        .def_property("y", [](const Points & points){return points.y();}, nullptr, py::keep_alive<0, 1>())
        .def("__repr__", &Points::info);

    py::class_<Structure>(m, "Structure")
        .def(py::init<int, int>(), py::arg("radius"), py::arg("rank"))
        .def_readonly("radius", &Structure::radius)
        .def_readonly("rank", &Structure::rank)
        .def_property("points", [](const Structure & srt){return Points{srt.idxs};}, nullptr, py::keep_alive<0, 1>())
        .def("__repr__", &Structure::info);

    py::class_<Regions>(m, "Regions")
        .def(py::init([](std::array<size_t, 2> shape){return Regions(std::move(shape));}), py::arg("shape"))
        .def_readonly("shape", &Regions::shape)
        .def("__delitem__", [](Regions & regions, size_t i)
        {
                 if (i >= regions.regions.size()) throw py::index_error();
                 regions.regions.erase(std::next(regions.regions.begin(), i));
        })
        .def("__getitem__", [](const Regions & regions, size_t i)
        {
                 if (i >= regions.regions.size()) throw py::index_error();
                 return regions.regions[i];
        })
        .def("__setitem__", [](Regions & regions, size_t i, Points region)
        {
            if (i >= regions.regions.size()) throw py::index_error();
            regions.regions[i] = std::move(region);
        })
        .def("__delitem__", [](Regions & regions, const py::slice & slice)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i)
            {
                regions.regions.erase(std::next(regions.regions.begin(), start));
                start += step;
            }
        })
        .def("__getitem__", [](const Regions & regions, const py::slice & slice) -> Regions
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            Regions new_regions (regions.shape);
            for (size_t i = 0; i < slicelength; ++i)
            {
                new_regions.regions.push_back(regions.regions[start]);
                start += step;
            }
            return new_regions;
        })
        .def("__setitem__", [](Regions & regions, const py::slice & slice, const Regions & value)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i)
            {
                regions.regions[start] = value.regions[i];
                start += step;
            }
        })
        .def("__iter__", [](Regions & regions)
        {
            return py::make_iterator(regions.regions.begin(), regions.regions.end());
        }, py::keep_alive<0, 1>())
        .def("__len__", [](Regions & regions){return regions.regions.size();})
        .def("__repr__", &Regions::info)
        .def("append", [](Regions & regions, Points region){regions.regions.emplace_back(std::move(region));}, py::keep_alive<1, 2>())
        .def("filter", &Regions::filter, py::arg("structure"), py::arg("npts"))
        .def("mask", [](Regions & regions) -> py::array_t<bool>
        {
            return as_pyarray(regions.mask(), regions.shape);
        })
        .def("center_of_mass", [](Regions & regions, py::array_t<double> data)
        {
            return regions.center_of_mass(array<double>{data.request()});
        }, py::arg("data"))
        .def("center_of_mass", [](Regions & regions, py::array_t<float> data)
        {
            return regions.center_of_mass(array<float>{data.request()});
        }, py::arg("data"))
        .def("gauss_fit", [](Regions & regions, py::array_t<double> data)
        {
            return regions.gauss_fit(array<double>{data.request()});
        }, py::arg("data"))
        .def("gauss_fit", [](Regions & regions, py::array_t<float> data)
        {
            return regions.gauss_fit(array<float>{data.request()});
        }, py::arg("data"))
        .def("ellipse_fit", [](Regions & regions, py::array_t<double> data)
        {
            return regions.ellipse_fit(array<double>{data.request()});
        }, py::arg("data"))
        .def("ellipse_fit", [](Regions & regions, py::array_t<float> data)
        {
            return regions.ellipse_fit(array<float>{data.request()});
        }, py::arg("data"))
        .def("line_fit", [](Regions & regions, py::array_t<double> data)
        {
            return regions.line_fit(array<double>{data.request()});
        }, py::arg("data"))
        .def("line_fit", [](Regions & regions, py::array_t<float> data)
        {
            return regions.line_fit(array<float>{data.request()});
        }, py::arg("data"))
        .def("moments", [](Regions & regions, py::array_t<double> data)
        {
            return regions.moments(array<double>{data.request()});
        }, py::arg("data"))
        .def("moments", [](Regions & regions, py::array_t<float> data)
        {
            return regions.moments(array<float>{data.request()});
        }, py::arg("data"));

    m.def("label", &label<float>, py::arg("mask"), py::arg("structure"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label<double>, py::arg("mask"), py::arg("structure"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);

}