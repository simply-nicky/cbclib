#ifndef GEOMETRY_
#define GEOMETRY_
#include "array.hpp"

namespace cbclib {

// 2D Point class

template <typename T>
class Point
{
public:
    using value_type = T;
    using size_type = size_t;

    using const_iterator = std::array<T, 2>::const_iterator;
    using iterator = std::array<T, 2>::iterator;

    using const_reference = std::array<T, 2>::const_reference;
    using reference = std::array<T, 2>::reference;

    const_iterator begin() const {return pt.begin();}
    const_iterator end() const {return pt.end();}
    iterator begin() {return pt.begin();}
    iterator end() {return pt.end();}

    const_reference operator[](size_type index) const {return pt[index];}
    reference operator[](size_type index) {return pt[index];}

    const_reference x() const {return pt[0];}
    const_reference y() const {return pt[1];}
    reference x() {return pt[0];}
    reference y() {return pt[1];}

    size_type size() const {return pt.size();}

    Point() : pt{} {}
    Point(T x, T y) : pt{x, y} {}

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    Point(const std::array<V, 2> & point) : pt{static_cast<T>(point[0]), static_cast<T>(point[1])} {}

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    Point(const Point<V> & point) : Point(point.to_array()) {}

    Point(std::array<T, 2> && point) : pt(std::move(point)) {}

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

    bool operator<(const Point<T> & rhs) const {return pt < rhs.pt;}
    bool operator==(const Point<T> & rhs) const {return pt == rhs.pt;}
    bool operator!=(const Point<T> & rhs) const {return !operator==(rhs);}

    friend std::ostream & operator<<(std::ostream & os, const Point<T> & pt)
    {
        os << "{" << pt.x() << ", " << pt.y() << "}";
        return os;
    }

    Point<T> clamp(const Point<T> & lo, const Point<T> & hi) const
    {
        return {std::clamp(x(), lo.x(), hi.x()), std::clamp(y(), lo.y(), hi.y())};
    }

    std::array<T, 2> coordinate() const
    {
        return {y(), x()};
    }

    std::array<T, 2> & to_array() & {return pt;}
    const std::array<T, 2> & to_array() const & {return pt;}
    std::array<T, 2> && to_array() && {return std::move(pt);}

    Point<T> round() const {return {static_cast<T>(std::round(x())), static_cast<T>(std::round(y()))};}

private:
    std::array<T, 2> pt;
};

template <class Container, typename T = typename Container::value_type>
constexpr auto to_point(Container & a, size_t start)
{
    return Point<T>(a[start], a[start + 1]);
}

template <typename T, typename V, typename U = std::common_type_t<T, V>, size_t N>
constexpr U dot(const std::array<T, N> & a, const std::array<V, N> & b)
{
    return apply_to_sequence<N>([&a, &b](auto... idxs){return ((a[idxs] * b[idxs]) + ...);});
}

template <typename T, typename V, typename U = std::common_type_t<T, V>>
constexpr U dot(const Point<T> & a, const Point<V> & b)
{
    return dot(a.to_array(), b.to_array());
}

template <typename T, size_t N>
constexpr T magnitude(const std::array<T, N> & a)
{
    return apply_to_sequence<N>([&a](auto... idxs){return ((a[idxs] * a[idxs]) + ...);});
}

template <typename T>
constexpr T magnitude(const Point<T> & p)
{
    return magnitude(p.to_array());
}

template <typename T, size_t N>
constexpr auto amplitude(const std::array<T, N> & a) -> decltype(std::sqrt(std::declval<T &>()))
{
    return std::sqrt(magnitude(a));
}

template <typename T>
constexpr auto amplitude(const Point<T> & p) -> decltype(std::sqrt(std::declval<T &>()))
{
    return amplitude(p.to_array());
}

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

    Line() = default;

    template <typename Pt0, typename Pt1, typename = std::enable_if_t<
        std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt0>> && std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt1>>
    >>
    Line(Pt0 && pt0, Pt1 && pt1) : pt0(std::forward<Pt0>(pt0)), pt1(std::forward<Pt1>(pt1)), tau(pt1 - pt0) {}

    Line(T x0, T y0, T x1, T y1) : Line(Point<T>{x0, y0}, Point<T>{x1, y1}) {}

    Point<T> norm() const {return {tau.y(), -tau.x()};}

    auto theta() const {return std::atan(tau.y(), tau.x());}

    template <typename U, typename V = std::common_type_t<T, U>, typename W = decltype(amplitude(std::declval<Point<V> &>()))>
    W distance(const Point<U> & point) const
    {
        if (magnitude(tau))
        {
            auto compare_point = [](const Point<V> & a, const Point<V> & b){return magnitude(a) < magnitude(b);};
            auto r = std::min(point - pt0, pt1 - point, compare_point);

            // need to divide by magnitude(tau) : dist = amplitude(norm() * dot(norm(), r) / magnitude(norm()))
            auto r_tau = static_cast<W>(dot(tau, r)) / magnitude(tau);
            auto r_norm = static_cast<W>(dot(norm(), r)) / magnitude(norm());
            if (r_tau > 1) return amplitude(norm() * r_norm + tau * (r_tau - 1));
            if (r_tau < 0) return amplitude(norm() * r_norm + tau * r_tau);
            return amplitude(norm() * r_norm);
        }
        return amplitude(pt0 - point);
    }

    template <typename U, typename V = std::common_type_t<T, U>, typename W = decltype(amplitude(std::declval<Point<V> &>()))>
    W normal_distance(const Point<U> & point) const
    {
        if (magnitude(tau))
        {
            auto compare_point = [](const Point<V> & a, const Point<V> & b){return magnitude(a) < magnitude(b);};
            auto r = std::min(point - pt0, pt1 - point, compare_point);
            return abs(dot(norm(), r) / amplitude(norm()));
        }
        return amplitude(pt0 - point);
    }

    friend std::ostream & operator<<(std::ostream & os, const Line<T> & line)
    {
        os << "{" << line.pt0 << ", " << line.pt1 << "}";
        return os;
    }

    std::array<T, 4> to_array() const {return {pt0.x(), pt0.y(), pt1.x(), pt1.y()};}
};

}

#endif
