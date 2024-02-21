#ifndef IMAGE_PROC_
#define IMAGE_PROC_
#include "array.hpp"

namespace cbclib {

// 2D Point class

template <typename T>
class Point
{
public:
    using value_type = T;

    using const_iterator = std::array<T, 2>::const_iterator;
    using iterator = std::array<T, 2>::iterator;

    using const_reference = std::array<T, 2>::const_reference;
    using reference = std::array<T, 2>::reference;

    const_iterator begin() const {return pt.begin();}
    const_iterator end() const {return pt.end();}
    iterator begin() {return pt.begin();}
    iterator end() {return pt.end();}

    const_reference operator[](size_t index) const {return pt[index];}
    reference operator[](size_t index) {return pt[index];}

    const_reference x() const {return pt[0];}
    const_reference y() const {return pt[1];}
    reference x() {return pt[0];}
    reference y() {return pt[1];}

    size_t size() const {return pt.size();}

    Point() : pt() {}
    Point(T x, T y) : pt({x, y}) {}

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    operator Point<V>() const {return {static_cast<V>(x()), static_cast<V>(y())};}

    operator std::array<T, 2>() const {return pt;}

    template <typename V>
    Point<std::common_type_t<T, V>> operator+(const Point<V> & rhs) const {return {x() + rhs.x(), y() + rhs.y()};}
    template <typename V>
    Point<std::common_type_t<T, V>> operator+(V rhs) const {return {x() + rhs, y() + rhs};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator+(V lhs, const Point<T> & rhs) {return {lhs + rhs.x(), lhs + rhs.y()};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator-(const Point<V> & rhs) const {return {x() - rhs.x(), y() - rhs.y()};}
    template <typename V>
    Point<std::common_type_t<T, V>> operator-(V rhs) const {return {x() - rhs, y() - rhs};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator-(V lhs, const Point<T> & rhs) {return {lhs - rhs.x(), lhs - rhs.y()};}


    template <typename V>
    Point<std::common_type_t<T, V>> operator*(V rhs) const {return {rhs * x(), rhs * y()};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator*(V lhs, const Point<T> & rhs) {return {lhs * rhs.x(), lhs * rhs.y()};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator/(V rhs) const {return {x() / rhs, y() / rhs};}

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator+=(const Point<V> & rhs) {x() += rhs.x(); y() += rhs.y(); return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator+=(V rhs) {x() += rhs; y() += rhs; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator-=(const Point<V> & rhs) {x() -= rhs.x(); y() -= rhs.y(); return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator-=(V rhs) {x() -= rhs; y() -= rhs; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> operator/=(V rhs) {x() /= rhs; y() /= rhs; return *this;}

    template <typename V>
    std::common_type_t<T, V> dot(const Point<V> & rhs) const {return x() * rhs.x() + y() * rhs.y();}

    bool operator<(const Point<T> & rhs) const
    {
        if (x() < rhs.x()) return true;
        if (rhs.x() < x()) return false;
        if (y() < rhs.y()) return true;
        if (rhs.y() < y()) return false;
        return false;
    }
    bool operator==(const Point<T> & rhs) const {return x() == rhs.x() && y() == rhs.y();}
    bool operator!=(const Point<T> & rhs) const {return !operator==(rhs);}

    friend std::ostream & operator<<(std::ostream & os, const Point<T> & pt)
    {
        os << "{" << pt.x() << ", " << pt.y() << "}";
        return os;
    }

    template <typename V, typename U, typename = std::enable_if_t<std::is_convertible_v<V, T> && std::is_convertible_v<U, T>>>
    Point<T> clamp(const Point<V> & lo, const Point<U> & hi) const
    {
        return {std::clamp<T>(x(), lo.x(), hi.x()), std::clamp<T>(y(), lo.y(), hi.y())};
    }

    std::array<T, 2> coordinate() const
    {
        return {y(), x()};
    }

    T magnitude() const {return x() * x() + y() * y();}
    Point<T> round() const {return {std::round(x()), std::round(y())};}

private:
    std::array<T, 2> pt;
};

template <typename Pt, typename = void>
struct is_point : std::false_type {};

template <typename Pt>
struct is_point <Pt, 
    typename std::enable_if_t<std::is_base_of_v<Point<typename Pt::value_type>, std::remove_cvref_t<Pt>>>
> : std::true_type {};

template <typename Pt>
constexpr bool is_point_v = is_point<Pt>::value;

// 2D Line class

template <typename T>
struct Line
{
    Point<T> pt0, pt1;
    Point<T> tau;

    operator std::array<T, 4>() const {return {pt0.x(), pt0.y(), pt1.x(), pt1.y()};}

    template <typename Pt0, typename Pt1, typename = std::enable_if_t<
        std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt0>> && std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt1>>
    >>
    Line(Pt0 && pt0, Pt1 && pt1) : pt0(std::forward<Pt0>(pt0)), pt1(std::forward<Pt1>(pt1)), tau(pt1 - pt0) {}

    Line(T x0, T y0, T x1, T y1) : Line(Point<T>{x0, y0}, Point<T>{x1, y1}) {}

    T magnitude() const {return tau.magnitude();}

    Point<T> norm() const {return {tau.y(), -tau.x()};}

    T perimeter() const {return std::abs(tau.x()) + std::abs(tau.y());}

    T theta() const {return std::atan(tau.y(), tau.x());}

    T distance(const Point<T> & point) const
    {
        if (magnitude())
        {
            auto r0 = pt0 - point, r1 = pt1 - point;

            if (r0.magnitude() < r1.magnitude())
            {
                auto dist = r0 - (tau.dot(r0) / magnitude()) * tau;
                return std::sqrt(dist.magnitude());
            }
            auto dist = r1 - (tau.dot(r1) / magnitude()) * tau;
            return std::sqrt(dist.magnitude());
        }
        return std::sqrt((pt0 - point).magnitude());
    }

    friend std::ostream & operator<<(std::ostream & os, const Line<T> & line)
    {
        os << "{" << line.pt0 << ", " << line.pt1 << "}";
        return os;
    }

};

template <typename T>
using table_t = std::tuple<std::vector<int>, std::vector<int>, std::vector<T>>;

namespace detail {

template <typename T, typename Out>
void draw_pixel(array<Out> & image, const Point<int> & pt, T val)
{
    if (image.is_inbound(pt.coordinate()))
    {
        auto index = image.ravel_index(pt.coordinate());
        image[index] = std::max(image[index], static_cast<Out>(val));
    }
    else throw std::runtime_error("Invalid pixel index: {" + std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
}

template <typename T, typename Out>
void draw_pixel(table_t<Out> & table, const Point<int> & pt, T val)
{
    std::get<0>(table).push_back(pt.x());
    std::get<1>(table).push_back(pt.y());
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
        point(std::forward<Point1>(pt)), tau(std::forward<Point2>(tau)), error((pt - origin).dot(tau)) {}

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
        if (step.x() * tau.x() > 0) return 2 * e_xy(step) <= step.x() * tau.x();
        return 2 * e_xy(step) >= step.x() * tau.x();
    }

    // Increment y if:
    //      e(x + sx, y + sy) + e(x + sx, y) < 0    if sy * tau.y() > 0
    //      e(x + sx, y + sy) + e(x + sx, y) > 0    if sy * tau.y() < 0

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    bool is_ynext(const Point<J> & step) const
    {
        if (step.y() * tau.y() > 0) return 2 * e_xy(step) <= step.y() * tau.y();
        return 2 * e_xy(step) >= step.y() * tau.y();
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
            for (auto liter = lpix.move(Point<int>{0, step.y()}), eiter = epix.move(Point<int>{0, step.y()});
                 std::abs(liter.error) < length * wd && liter.point.y() != bnd1.y() + step.y();
                 liter.ystep(step.y()), eiter.ystep(step.y()))
            {
                auto r1 = liter.error / length, r2 = std::min(eiter.error / length, T()), r3 = std::max(eiter.error / length - length, T());
                auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
                detail::draw_pixel(image, liter.point, val);
            }
            if (lpix.point.x() == bnd1.x()) break;
            new_step.x() = step.x();
        }
        if (lpix.is_ynext(step))
        {
            // y step
            for (auto liter = lpix.move(Point<int>{step.x(), 0}), eiter = epix.move(Point<int>{step.x(), 0});
                 std::abs(liter.error) < length * wd && liter.point.x() != bnd1.x() + step.x();
                 liter.xstep(step.x()), eiter.xstep(step.x()))
            {
                auto r1 = liter.error / length, r2 = std::min(eiter.error / length, T()), r3 = std::max(eiter.error / length - length, T());
                auto val = max_val * kernel(std::sqrt(r1 * r1 + r2 * r2 + r3 * r3), wd);
                detail::draw_pixel(image, liter.point, val);
            }
            if (lpix.point.y() == bnd1.y()) break;
            new_step.y() = step.y();
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