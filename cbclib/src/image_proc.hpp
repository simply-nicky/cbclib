#ifndef IMAGE_PROC_
#define IMAGE_PROC_
#include "geometry.hpp"

namespace cbclib {

using point_t = Point<long>;

template <typename T, typename I = typename point_t::value_type>
using table_t = std::map<std::pair<I, I>, T>;

namespace detail {

template <typename T>
void draw_pixel(array<T> & image, const point_t & pt, T val)
{
    if (val)
    {
        if (image.is_inbound(pt.coordinate()))
        {
            auto index = image.ravel_index(pt.coordinate());
            image[index] += static_cast<T>(val);
        }
        else throw std::runtime_error("Invalid pixel index: {" + std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
    }
}

template <typename T>
T get_pixel(const array<T> & image, const point_t & pt)
{
    if (image.is_inbound(pt.coordinate()))
    {
        auto index = image.ravel_index(pt.coordinate());
        return image[index];
    }
    else throw std::runtime_error("Invalid pixel index: {" + std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
}

template <typename T, typename Key = typename table_t<T>::key_type>
void draw_pixel(table_t<T> & table, const Key & key , T val)
{
    if (val) table[key] = val;
}

template <typename T, typename Key = typename table_t<T>::key_type>
T get_pixel(const table_t<T> & table, const Key & key)
{
    if (table.contains(key)) return table.at(key);
    else return T();
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

// Iterator for a rasterizing algorithm for drawing lines
// See -> http://members.chello.at/~easyfilter/bresenham.html

template <typename T, typename I>
struct BhmIterator
{
    Point<I> point;     /* Current point position                                           */
    Point<T> tau;       /* Line norm, derivative of a line error function:                  */
                        /* error(x + dx, y + dy) = error(x, y) + tau.x * dx + tau.y * dy    */
    T error;            /* Current error value                                              */

    template <typename Point1, typename Point2, typename Point3>
    BhmIterator(Point1 && pt, Point2 && tau, const Point3 & origin) :
        point(std::forward<Point1>(pt)), tau(std::forward<Point2>(tau)), error(dot(pt - origin, tau)) {}

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    BhmIterator move(const Point<J> & step) const
    {
        auto pix = *this;
        pix.xstep(step.x()); pix.ystep(step.y());
        return pix;
    }

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    BhmIterator & step(const Point<J> & step)
    {
        xstep(step.x()); ystep(step.y());
        return *this;
    }

    BhmIterator & xstep(I step)
    {
        point.x() += step; error += step * tau.x();
        return *this;
    }

    BhmIterator & ystep(I step)
    {
        point.y() += step; error += step * tau.y();
        return *this;
    }

    // Increment x if:
    //      e(x + sx, y + sy) + e(x, y + sy) < 0    if sx * tau.x() > 0
    //      e(x + sx, y + sy) + e(x, y + sy) > 0    if sx * tau.x() < 0

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    bool is_xnext(const Point<J> & step) const
    {
        // std::cout << "2 * e_xy(step) = " << 2 * e_xy(step) << ", step.x() * tau.x() = " << step.x() * tau.x() << std::endl;
        if (step.x() * tau.x() == 0) return true;
        if (step.x() * tau.x() > 0) return 2 * e_xy(step) <= step.x() * tau.x();
        return 2 * e_xy(step) >= step.x() * tau.x();
    }

    // Increment y if:
    //      e(x + sx, y + sy) + e(x + sx, y) < 0    if sy * tau.y() > 0
    //      e(x + sx, y + sy) + e(x + sx, y) > 0    if sy * tau.y() < 0

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    bool is_ynext(const Point<J> & step) const
    {
        // std::cout << "2 * e_xy(step) = " << 2 * e_xy(step) << ", step.y() * tau.y() = " << step.y() * tau.y() << std::endl;
        if (step.y() * tau.y() == 0) return true;
        if (step.y() * tau.y() > 0) return 2 * e_xy(step) <= step.y() * tau.y();
        return 2 * e_xy(step) >= step.y() * tau.y();
    }

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    bool is_end(const Point<J> & start, const Point<J> & end) const
    {
        return dot(point - end, end - start) >= 0;
    }

private:

    // Return e(x + sx, y + sy)

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    T e_xy(const Point<J> & step) const
    {
        return error + step.x() * tau.x() + step.y() * tau.y();
    }
};

// Drawing direction

enum class direction
{
    forward,
    backward
};

// Choose a steping vector based on drawing direction
// tau is NOT the same as BhmIterator::tau (Line::norm()), but is equal to Line::tau

template <typename T>
Point<int> bresenham_step(const Point<T> & tau, direction dir)
{
    int xstep, ystep;
    switch (dir)
    {
        case direction::forward:
            xstep = tau.x() > T() ? 1 : -1;
            ystep = tau.y() > T() ? 1 : -1;
            break;

        case direction::backward:
            xstep = tau.x() > T() ? -1 : 1;
            ystep = tau.y() > T() ? -1 : 1;
            break;

        default:
            throw std::invalid_argument("invalid direction argument: " + std::to_string(static_cast<int>(dir)));
    }
    return {xstep, ystep};
}

template <typename T, class Func, typename I = typename point_t::value_type, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, BhmIterator<T, I>, BhmIterator<T, I>>
>>
void draw_bresenham_func(const point_t & ubound, const Line<T> & line, T width, Func && func)
{
    /* Discrete line */
    point_t pt0 (line.pt0.round());
    point_t pt1 (line.pt1.round());

    T length = amplitude(line.tau);
    int wi = std::ceil(width) + 1;

    auto step = bresenham_step(line.tau, direction::forward);

    if (!length) return;

    /* Define bounds of the line plot */
    auto bnd0 = (pt0 - wi * step).clamp(point_t{}, ubound);
    auto bnd1 = (pt1 + wi * step).clamp(point_t{}, ubound);

    BhmIterator<T, I> lpix {bnd0, line.norm(), line.pt0};
    BhmIterator<T, I> epix {bnd0, line.tau, line.pt0};
    point_t new_step;

    do
    {
        // Perform a step
        lpix.step(new_step); epix.step(new_step);
        new_step = point_t();

        // Draw a pixel
        std::forward<Func>(func)(lpix, epix);

        if (lpix.is_xnext(step))
        {
            // x step
            for (auto liter = lpix.move(point_t{0, step.y()}), eiter = epix.move(point_t{0, step.y()});
                 std::abs(liter.error) < length * width && liter.point.y() != bnd1.y() + step.y();
                 liter.ystep(step.y()), eiter.ystep(step.y()))
            {
                std::forward<Func>(func)(liter, eiter);
            }
            new_step.x() = step.x();
        }
        if (lpix.is_ynext(step))
        {
            // y step
            for (auto liter = lpix.move(point_t{step.x(), 0}), eiter = epix.move(point_t{step.x(), 0});
                 std::abs(liter.error) < length * width && liter.point.x() != bnd1.x() + step.x();
                 liter.xstep(step.x()), eiter.xstep(step.x()))
            {
                std::forward<Func>(func)(liter, eiter);
            }
            new_step.y() = step.y();
        }
    }
    while (!lpix.is_end(bnd0, bnd1));
}

}

#endif