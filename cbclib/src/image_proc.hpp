#ifndef IMAGE_PROC_
#define IMAGE_PROC_
#include "streak_finder.hpp"

namespace cbclib {

template <typename T>
using table_t = std::tuple<std::vector<int>, std::vector<int>, std::vector<T>>;

namespace detail {

template <typename T, typename Out>
void draw_pixel(array<Out> & image, const Point<int> & pt, T val)
{
    if (image.is_inbound({pt.y, pt.x}))
    {
        auto index = image.ravel_index({pt.y, pt.x});
        image[index] = std::max(image[index], static_cast<Out>(val));
    }
    else throw std::runtime_error("Invalid pixel index: {" + std::to_string(pt.y) + ", " + std::to_string(pt.x) + "}");
}

template <typename T, typename Out>
void draw_pixel(table_t<Out> & table, const Point<int> & pt, T val)
{
    std::get<0>(table).push_back(pt.x);
    std::get<1>(table).push_back(pt.y);
    std::get<2>(table).push_back(static_cast<Out>(val));
}

}

/*----------------------------------------------------------------------------*/
/*-------------------------- Bresenham's Algorithm ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
    Function :  plot_line_width()
    In       :  A 2d line defined by two (float) points (x0, y0, x1, y1)
                and a (float) width wd
    Out      :  A rasterized image of the line

    Reference:
        Author: Alois Zingl
        Title: A Rasterizing Algorithm for Drawing Curves
        pdf: http://members.chello.at/%7Eeasyfilter/Bresenham.pdf
        url: http://members.chello.at/~easyfilter/bresenham.html
------------------------------------------------------------------------------*/
template <typename Data, typename T, typename Out>
void draw_bresenham(Data & image, const Point<size_t> & ubound, const Line<T> & line, T width, Out max_val, typename kernels<T>::kernel kernel)
{
    /* Discrete line */
    auto pt0 = static_cast<Point<int>>(line.pt0.round());
    auto pt1 = static_cast<Point<int>>(line.pt1.round());

    T wd = (width + 1.0) / 2, length = std::sqrt(line.magnitude());
    int wi = std::ceil(wd);

    auto step = bresenham_step(line.tau, direction::forward);

    if (!length) return;

    /* Define bounds of the line plot */
    auto bnd0 = (pt0 - wi * step).clamp(Point<size_t>{}, ubound);
    auto bnd1 = (pt1 + wi * step).clamp(Point<size_t>{}, ubound);

    BhmIterator<T, int> lpix {bnd0, line.norm(), line.pt0};
    BhmIterator<T, int> epix {bnd0, line.tau, line.pt0};
    Point<int> new_step;

    auto max_cnt = Line<int>(bnd0, bnd1).perimeter();

    for (int cnt = 0; cnt < max_cnt; cnt++)
    {
        // Perform a step
        lpix.step(new_step); epix.step(new_step);
        new_step = Point<int>();

        // Draw a pixel
        auto r1 = lpix.error / length, r2 = std::min(epix.error / length, T()), r3 = std::max(epix.error / length - length, T());
        auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
        detail::draw_pixel(image, lpix.point, val);

        if (lpix.is_xnext(step))
        {
            // x step
            for (auto liter = lpix.move(Point<int>{0, step.y}), eiter = epix.move(Point<int>{0, step.y});
                 std::abs(liter.error) < length * wd && liter.point.y != bnd1.y + step.y;
                 liter.ystep(step.y), eiter.ystep(step.y))
            {
                auto r1 = liter.error / length, r2 = std::min(eiter.error / length, T()), r3 = std::max(eiter.error / length - length, T());
                auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
                detail::draw_pixel(image, liter.point, val);
            }
            if (lpix.point.x == bnd1.x) break;
            new_step.x = step.x;
        }
        if (lpix.is_ynext(step))
        {
            // y step
            for (auto liter = lpix.move(Point<int>{step.x, 0}), eiter = epix.move(Point<int>{step.x, 0});
                 std::abs(liter.error) < length * wd && liter.point.x != bnd1.x + step.x;
                 liter.xstep(step.x), eiter.xstep(step.x))
            {
                auto r1 = liter.error / length, r2 = std::min(eiter.error / length, T()), r3 = std::max(eiter.error / length - length, T());
                auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
                detail::draw_pixel(image, liter.point, val);
            }
            if (lpix.point.y == bnd1.y) break;
            new_step.y = step.y;
        }
    }
}

template <typename T, typename Out>
py::array_t<Out> draw_line_new(py::array_t<T, py::array::c_style | py::array::forcecast> lines,
                               std::vector<size_t> shape, Out max_val, T dilation, std::string kernel, unsigned threads);

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