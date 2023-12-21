#include "image_proc.hpp"

namespace cbclib {

template <typename T>
void check_lines(array<T> & lines)
{
    if (lines.ndim == 1 && lines.size == 5) lines = lines.reshape({1, 5});
    check_dimensions("lines", lines.ndim - 2, lines.shape, lines.size / 5, 5);
}

template <typename T, typename Out>
py::array_t<Out> draw_line(py::array_t<T, py::array::c_style | py::array::forcecast> lines,
                           std::vector<size_t> shape, Out max_val, T dilation, std::string kernel, unsigned threads)
{
    assert(PyArray_API);

    auto krn = kernels<T>::get_kernel(kernel);

    check_shape(shape, [](const std::vector<size_t> & shape){return shape.size() != 2;});
    Point<size_t> ubound {shape[shape.size() - 1] - 1, shape[shape.size() - 2] - 1};

    auto larr = array<T>(lines.request());
    check_lines(larr);

    auto out = py::array_t<Out>(shape);
    PyArray_FILLWBYTE(out.ptr(), 0);

    auto oarr = array<Out>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t j = 0; j < larr.shape[0]; j++)
    {
        e.run([&]
        {
            auto liter = larr.line_begin(1, j);
            draw_bresenham(oarr, {oarr.shape[1] - 1, oarr.shape[0] - 1}, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, krn);
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename Out>
py::array_t<Out> draw_line_vec(std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> lines,
                               std::vector<size_t> shape, Out max_val, T dilation, std::string kernel, unsigned threads)
{
    assert(PyArray_API);

    auto krn = kernels<T>::get_kernel(kernel);

    check_shape(shape, [](const std::vector<size_t> & shape){return shape.size() < 3;});
    Point<size_t> ubound {shape[shape.size() - 1] - 1, shape[shape.size() - 2] - 1};

    if (get_size(shape.begin(), std::prev(shape.end(), 2)) != lines.size())
        throw std::invalid_argument("shape is incompatible with the list of lines");

    std::vector<array<T>> lvec;
    for (const auto & obj : lines)
    {
        auto & arr = lvec.emplace_back(obj.request());
        check_lines(arr);
    }

    auto out = py::array_t<Out>(shape);
    PyArray_FILLWBYTE(out.ptr(), 0);

    auto oarr = array<Out>(out.request());
    std::array<size_t, 2> axes = {oarr.ndim - 2, oarr.ndim - 1};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < lvec.size(); i++)
    {
        e.run([&]
        {
            auto frame = oarr.slice(i, axes);

            for (size_t j = 0; j < lvec[i].shape[0]; j++)
            {
                auto liter = lvec[i].line_begin(1, j);
                draw_bresenham(frame, ubound, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, krn);
            }
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename Out>
auto draw_line_table(py::array_t<T, py::array::c_style | py::array::forcecast> lines, std::optional<std::vector<size_t>> shape,
                     Out max_val, T dilation, std::string kernel, unsigned threads)
{
    assert(PyArray_API);

    auto krn = kernels<T>::get_kernel(kernel);

    auto larr = array<T>(lines.request());
    check_lines(larr);

    Point<size_t> ubound;
    if (shape)
    {
        check_shape(shape.value(), [](const std::vector<size_t> & shape){return shape.size() != 2;});
        ubound.x = shape.value()[shape.value().size() - 1] - 1;
        ubound.y = shape.value()[shape.value().size() - 2] - 1;
    }
    else {ubound.x = INT_MAX; ubound.y = INT_MAX;}

    table_t<Out> result;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t j = 0; j < larr.shape[0]; j++)
    {
        e.run([&]
        {
            auto liter = larr.line_begin(1, j);
            draw_bresenham(result, ubound, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, krn);
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    auto & [xs, ys, vals] = result;

    return std::make_tuple(as_pyarray(std::move(xs)), as_pyarray(std::move(ys)), as_pyarray(std::move(vals)));
}

template <typename T, typename Out>
auto draw_line_table_vec(std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> lines,
                         std::optional<std::vector<size_t>> shape, Out max_val, T dilation, std::string kernel, unsigned threads)
{
    assert(PyArray_API);

    auto krn = kernels<T>::get_kernel(kernel);

    Point<size_t> ubound;
    if (shape)
    {
        check_shape(shape.value(), [](const std::vector<size_t> & shape){return shape.size() < 3;});
        ubound.x = shape.value()[shape.value().size() - 1] - 1;
        ubound.y = shape.value()[shape.value().size() - 2] - 1;
    }
    else {ubound.x = INT_MAX; ubound.y = INT_MAX;}

    std::vector<array<T>> lvec;
    for (const auto & obj : lines)
    {
        auto & arr = lvec.emplace_back(obj.request());
        check_lines(arr);
    }

    std::vector<table_t<Out>> result;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<table_t<Out>> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < lvec.size(); i++)
        {
            e.run([&]
            {
                auto & table = buffer.emplace_back();

                for (size_t j = 0; j < lvec[i].shape[0]; j++)
                {
                    auto liter = lvec[i].line_begin(1, j);
                    draw_bresenham(table, ubound, {liter[0], liter[1], liter[2], liter[3]}, liter[4] + dilation, max_val, krn);
                }
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

    std::vector<std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<Out>>> out;
    for (auto & [xs, ys, vals] : result)
    {
        out.emplace_back(as_pyarray(std::move(xs)), as_pyarray(std::move(ys)), as_pyarray(std::move(vals)));
    }

    return out;
}

}

PYBIND11_MODULE(image_proc, m)
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

    m.def("draw_line_mask", &draw_line<double, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask", &draw_line_vec<double, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask", &draw_line<float, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask", &draw_line_vec<float, uint32_t>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);

    m.def("draw_line_image", &draw_line<float, float>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image", &draw_line_vec<float, float>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image", &draw_line<double, double>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image", &draw_line_vec<double, double>, py::arg("lines"), py::arg("shape"), py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);

    m.def("draw_line_table", &draw_line_table<float, float>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table_vec<float, float>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table<double, double>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table_vec<double, double>, py::arg("lines"), py::arg("shape") = std::nullopt, py::arg("max_val") = 1.0, py::arg("dilation") = 0.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
}