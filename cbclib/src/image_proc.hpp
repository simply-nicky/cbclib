#ifndef IMAGE_PROC_
#define IMAGE_PROC_
#include "array.hpp"

namespace cbclib {

template <typename T>
using table_t = std::tuple<std::vector<int>, std::vector<int>, std::vector<T>>;

namespace detail {

template <typename T, typename Out>
void draw_pixel(array<Out> & image, int x, int y, T val)
{
    if (image.is_inbound({y, x}))
    {
        auto index = image.ravel_index({y, x});
        image[index] = std::max(image[index], static_cast<Out>(val));
    }
    else throw std::runtime_error("Invalid pixel index: {" + std::to_string(y) + ", " + std::to_string(x) + "}");
}

template <typename T, typename Out>
void draw_pixel(table_t<Out> & table, int x, int y, T val)
{
    std::get<0>(table).push_back(x);
    std::get<1>(table).push_back(y);
    std::get<2>(table).push_back(static_cast<Out>(val));
}

}

template <typename Data, typename T, typename Out>
void draw_bresenham(Data & image, std::vector<size_t> * shape, const std::array<T, 4> & line, T width, Out max_val, typename kernels<T>::kernel kernel)
{
    /* create a volatile copy of the input line */
    /* the points are given by [j0, i0, j1, i1] */
    std::array<int, 4> pts = {static_cast<int>(std::round(line[1])), static_cast<int>(std::round(line[0])),
                              static_cast<int>(std::round(line[3])), static_cast<int>(std::round(line[2]))};

    /* plot an anti-aliased line of width wd */
    T dx = std::abs(line[2] - line[0]), dy = std::abs(line[3] - line[1]), wd = 0.5 * (width + 1.0);
    int sx = (pts[1] < pts[3]) ? 1 : -1, sy = (pts[0] < pts[2]) ? 1 : -1, wi = std::round(wd);
    T ed = std::sqrt(dx * dx + dy * dy);

    /* initialise line error : err1 = [(y - line[1]) * dx - (x - line[0]) * dy] / ed */
    T err1 = (pts[0] - line[1]) * dx - (pts[1] - line[0]) * dy;

    /* check if line has a non-zero length */
    if (ed == 0.0) return;

    /* initialise bound error: err2 = [(x - line[0]) * dx + (y - line[1]) * dy] / ed */
    T err2 = (pts[1] - line[0]) * dx + (pts[0] - line[1]) * dy;

    /* define image bounds */
    std::array<int, 4> bnd;

    if (pts[1] < pts[3])
    {
        auto max = (shape) ? static_cast<int>((*shape)[1]) - 1 : INT_MAX;
        bnd[1] = std::clamp(pts[1] - wi, 0, max);
        bnd[3] = std::clamp(pts[3] + wi, 0, max);

        err1 += (pts[1] - bnd[1]) * dy;
        err2 -= (pts[1] - bnd[1]) * dx;
        pts[1] = bnd[1]; pts[3] = bnd[3];
    }
    else
    {
        auto max = (shape) ? static_cast<int>((*shape)[1]) - 1 : INT_MAX;
        bnd[1] = std::clamp(pts[3] - wi, 0, max);
        bnd[3] = std::clamp(pts[1] + wi, 0, max);

        err1 += (bnd[3] - pts[1]) * dy;
        err2 -= (bnd[3] - pts[1]) * dx;
        pts[1] = bnd[3]; pts[3] = bnd[1];
    }
    if (pts[0] < pts[2])
    {
        auto max = (shape) ? static_cast<int>((*shape)[0]) - 1 : INT_MAX;
        bnd[0] = std::clamp(pts[0] - wi, 0, max);
        bnd[2] = std::clamp(pts[2] + wi, 0, max);

        err1 -= (pts[0] - bnd[0]) * dx;
        err2 -= (pts[0] - bnd[0]) * dy;
        pts[0] = bnd[0]; pts[2] = bnd[2];
    }
    else
    {
        auto max = (shape) ? static_cast<int>((*shape)[0]) - 1 : INT_MAX;
        bnd[0] = std::clamp(pts[2] - wi, 0, max);
        bnd[2] = std::clamp(pts[0] + wi, 0, max);

        err1 -= (bnd[2] - pts[0]) * dx;
        err2 -= (bnd[2] - pts[0]) * dy; 
        pts[0] = bnd[2]; pts[2] = bnd[0];
    }

    /* Main loop */
    T derr1 = T(), derr2 = T(); int dx0 = 0;
    for (int cnt = 0; cnt < dx + dy + 4 * wi; cnt++)
    {
        /* pixel loop */
        err1 += derr1; derr1 = T();
        err2 += derr2; derr2 = T();
        pts[1] += dx0; dx0 = 0;

        auto r1 = err1 / ed, r2 = std::min(err2 / ed - T(M_SQRT1_2) * wd, T()), r3 = std::max(err2 / ed - ed + T(M_SQRT1_2) * wd, T());
        auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
        detail::draw_pixel(image, pts[1], pts[0], val);

        if (2 * err1 >= -dx)
        {
            /* x step */
            T e1, e2; int y2;
            for (e1 = err1 + dx, e2 = err2 + dy, y2 = pts[0] + sy;
                 abs(e1) < ed * wd && y2 >= bnd[0] && y2 <= bnd[2];
                 e1 += dx, e2 += dy, y2 += sy)
            {
                auto r1 = e1 / ed, r2 = std::min(e2 / ed - T(M_SQRT1_2) * wd, T()), r3 = std::max(e2 / ed - ed + T(M_SQRT1_2) * wd, T());
                auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
                detail::draw_pixel(image, pts[1], y2, val);
            }
            if (pts[1] == pts[3]) break;
            derr1 -= dy; derr2 += dx; dx0 += sx;
        }
        if (2 * err1 <= dy)
        {
            /* y step */
            T e1, e2; int x2;
            for (e1 = err1 - dy, e2 = err2 + dx, x2 = pts[1] + sx;
                 abs(e1) < ed * wd && x2 >= bnd[1] && x2 <= bnd[3];
                 e1 -= dy, e2 += dx, x2 += sx)
            {
                auto r1 = e1 / ed, r2 = std::min(e2 / ed - T(M_SQRT1_2) * wd , T()), r3 = std::max(e2 / ed - ed + T(M_SQRT1_2) * wd, T());
                auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
                detail::draw_pixel(image, x2, pts[0], val);
            }
            if (pts[0] == pts[2]) break;
            derr1 += dx; derr2 += dy; pts[0] += sy;
        }
    }
}

template <typename T, typename Out>
py::array_t<Out> draw_line(py::array_t<T, py::array::c_style | py::array::forcecast> lines,
                           std::vector<size_t> shape, Out max_val, T dilation, std::string kernel, unsigned threads);

template <typename T, typename Out>
py::array_t<Out> draw_line_vec(std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> lines,
                               std::vector<size_t> shape, Out max_val, T dilation, std::string kernel, unsigned threads);

template <typename T, typename Out>
auto draw_line_table(py::array_t<T, py::array::c_style | py::array::forcecast> lines, std::optional<std::vector<size_t>> shape,
                     Out max_val, T dilation, std::string kernel, unsigned threads);

template <typename T, typename Out>
auto draw_line_table_vec(std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> lines,
                         std::optional<std::vector<size_t>> shape, Out max_val, T dilation, std::string kernel, unsigned threads);

}

#endif