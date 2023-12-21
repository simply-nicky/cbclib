#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "array.hpp"
#include "signal_proc.hpp"

namespace cbclib {

namespace detail{

// Taken from the boost::hash_combine: https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
template <typename T, size_t N>
struct ArrayHasher
{
    std::size_t operator()(const std::array<T, N> & arr) const
    {
        std::size_t h = 0;

        for (auto elem : arr)
        {
            h ^= std::hash<T>{}(elem)  + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }   
};

}

template <typename T>
struct Point
{
    T x, y;

    Point() : x(), y() {}
    Point(T x, T y) : x(x), y(y) {}

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    explicit operator Point<V>() const {return {static_cast<V>(x), static_cast<V>(y)};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator+(const Point<V> & rhs) const {return {x + rhs.x, y + rhs.y};}
    template <typename V>
    Point<std::common_type_t<T, V>> operator+(V rhs) const {return {x + rhs, y + rhs};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator+(V lhs, const Point<T> & rhs) {return {lhs + rhs.x, lhs + rhs.y};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator-(const Point<V> & rhs) const {return {x - rhs.x, y - rhs.y};}
    template <typename V>
    Point<std::common_type_t<T, V>> operator-(V rhs) const {return {x - rhs, y - rhs};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator-(V lhs, const Point<T> & rhs) {return {lhs - rhs.x, lhs - rhs.y};}


    template <typename V>
    Point<std::common_type_t<T, V>> operator*(V rhs) const {return {rhs * x, rhs * y};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator*(V lhs, const Point<T> & rhs) {return {lhs * rhs.x, lhs * rhs.y};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator/(V rhs) const {return {x / rhs, y / rhs};}

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator+=(const Point<V> & rhs) {x += rhs.x; y += rhs.y; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator+=(V rhs) {x += rhs; y += rhs; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator-=(const Point<V> & rhs) {x -= rhs.x; y -= rhs.y; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator-=(V rhs) {x -= rhs; y -= rhs; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> operator/=(V rhs) {x /= rhs; y /= rhs; return *this;}

    template <typename V>
    std::common_type_t<T, V> dot(const Point<V> & rhs) const {return x * rhs.x + y * rhs.y;}

    bool operator<(const Point<T> & rhs) const
    {
        if (x < rhs.x) return true;
        if (rhs.x < x) return false;
        if (y < rhs.y) return true;
        if (rhs.y < y) return false;
        return false;
    }
    bool operator==(const Point<T> & rhs) const {return x == rhs.x && y == rhs.y;}
    bool operator!=(const Point<T> & rhs) const {return !operator==(rhs);}

    friend std::ostream & operator<<(std::ostream & os, const Point<T> & pt)
    {
        os << "{" << pt.x << ", " << pt.y << "}";
        return os;
    }

    template <typename V, typename U, typename = std::enable_if_t<std::is_convertible_v<V, T> && std::is_convertible_v<U, T>>>
    Point<T> clamp(const Point<V> & lo, const Point<U> & hi) const
    {
        return {std::clamp<T>(x, lo.x, hi.x), std::clamp<T>(y, lo.y, hi.y)};
    }

    T magnitude() const {return x * x + y * y;}
    Point<T> round() const {return {std::round(x), std::round(y)};}

    std::array<T, 2> to_array() const {return {x, y};}
};

template <typename T>
struct Line
{
    Point<T> pt0, pt1;
    Point<T> tau;

    template <typename Pt, typename = std::enable_if_t<std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt>>>>
    Line(Pt && pt0, Pt && pt1) : pt0(std::forward<Pt>(pt0)), pt1(std::forward<Pt>(pt1)), tau(pt1 - pt0) {}

    Line(T x0, T y0, T x1, T y1) : Line(Point<T>{x0, y0}, Point<T>{x1, y1}) {}

    T magnitude() const {return tau.magnitude();}

    Point<T> norm() const {return {tau.y, -tau.x};}

    T perimeter() const {return std::abs(tau.x) + std::abs(tau.y);}

    T theta() const {return std::atan(tau.y, tau.x);}

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
            auto dist = r1 - (tau.dor(r1) / magnitude()) * tau;
            return std::sqrt(dist.magnitude());
        }
        return std::sqrt((pt0 - point).magnitude());
    }

    std::array<T, 4> to_array() const {return {pt0.x, pt0.y, pt1.x, pt1.y};}

    friend std::ostream & operator<<(std::ostream & os, const Line<T> & line)
    {
        os << "{" << line.pt0 << ", " << line.pt1 << "}";
        return os;
    }

};

template <typename T, typename I>
struct BhmIterator
{
    Point<I> point;
    Point<T> tau;
    T error;

    template <typename Point1, typename Point2, typename Point3>
    BhmIterator(Point1 && pt, Point2 && tau, const Point3 & origin) :
        point(std::forward<Point1>(pt)), tau(std::forward<Point2>(tau)), error((pt - origin).dot(tau)) {}

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    BhmIterator move(const Point<J> & step) const
    {
        auto pix = *this;
        pix.xstep(step.x); pix.ystep(step.y);
        return pix;
    }

    template <typename J, typename = std::enable_if_t<std::is_convertible_v<I, J>>>
    BhmIterator & step(const Point<J> & step)
    {
        xstep(step.x); ystep(step.y);
        return *this;
    }

    BhmIterator & xstep(I step)
    {
        point.x += step; error += step * tau.x;
        return *this;
    }

    BhmIterator & ystep(I step)
    {
        point.y += step; error += step * tau.y;
        return *this;
    }

    bool is_xnext(I step) const
    {
        if (step * tau.x > 0) return 2 * error <= step * tau.x;
        return 2 * error >= step * tau.x;
    }
    bool is_ynext(I step) const
    {
        if (step * tau.y > 0) return 2 * error <= step * tau.y;
        return 2 * error >= step * tau.y;
    }
};

template <typename T>
using pset_t = std::set<std::tuple<Point<int>, T>>;

template <typename T>
struct Moments
{
    Point<T> pt0;
    T mu, mu_x, mu_y, mu_xx, mu_xy, mu_yy;

    template <typename Pt, typename = std::enable_if_t<std::is_base_of_v<Point<T>,  std::remove_cvref_t<Pt>>>>
    static Moments from_pset(Pt && pt0, const pset_t<T> & pset)
    {
        Moments m {std::forward<Pt>(pt0)};
        for (const auto & [pt, val] : pset)
        {
            m.add_point(pt, val);
        }
        return m;
    }

    static Moments from_pset(const pset_t<T> & pset)
    {
        Point<T> pt0; T mu = T();
        for (const auto & [pt, val] : pset)
        {
            pt0 += val * pt;
            mu += val;
        }

        if (mu) return Moments::from_pset(pt0 / mu, pset);
        else
        {
            Moments m;
            for (const auto & [pt, _] : pset)
            {
                m.pt0 += pt;
            }
            m.pt0 /= pset.size();
            return m;
        }
    }

    Moments operator+(const Moments & m) const
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            return {pt0, mu + new_m.mu, mu_x + new_m.mu_x, mu_y + new_m.mu_y,
                    mu_xx + new_m.mu_xx, mu_xy + new_m.mu_xy, mu_yy + new_m.mu_yy};
        }

        return {pt0, mu + m.mu, mu_x + m.mu_x, mu_y + m.mu_y,
                mu_xx + m.mu_xx, mu_xy + m.mu_xy, mu_yy + m.mu_yy};
    }

    Moments operator-(const Moments & m) const
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            return {pt0, mu - new_m.mu, mu_x - new_m.mu_x, mu_y - new_m.mu_y,
                    mu_xx - new_m.mu_xx, mu_xy - new_m.mu_xy, mu_yy - new_m.mu_yy};
        }

        return {pt0, mu - m.mu, mu_x - m.mu_x, mu_y - m.mu_y,
                mu_xx - m.mu_xx, mu_xy - m.mu_xy, mu_yy - m.mu_yy};
    }

    Moments & operator+=(const Moments & m)
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            mu += new_m.mu; mu_x += new_m.mu_x; mu_y += new_m.mu_y;
            mu_xx += new_m.mu_xx; mu_xy += new_m.mu_xy; mu_yy += new_m.mu_yy;
            return *this;
        }

        mu += m.mu; mu_x += m.mu_x; mu_y += m.mu_y;
        mu_xx += m.mu_xx; mu_xy += m.mu_xy; mu_yy += m.mu_yy;
        return *this;
    }

    Moments & operator-=(const Moments & m)
    {
        if (pt0 != m.pt0)
        {
            auto new_m = m.update_seed(pt0);
            mu -= new_m.mu; mu_x -= new_m.mu_x; mu_y -= new_m.mu_y;
            mu_xx -= new_m.mu_xx; mu_xy -= new_m.mu_xy; mu_yy -= new_m.mu_yy;
            return *this;
        }

        mu -= m.mu; mu_x -= m.mu_x; mu_y -= m.mu_y;
        mu_xx -= m.mu_xx; mu_xy -= m.mu_xy; mu_yy -= m.mu_yy;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    void add_point(const Point<V> & point, T val)
    {
        auto dist = point - pt0;

        mu += val;
        mu_x += dist.x * val;
        mu_y += dist.y * val;
        mu_xx += dist.x * dist.x * val;
        mu_xy += dist.x * dist.y * val;
        mu_yy += dist.y * dist.y * val;
    }

    Point<T> center_of_mass() const
    {
        if (mu) return {mu_x / mu + pt0.x, mu_y / mu + pt0.y};
        return pt0;
    }

    T theta() const
    {
        if (mu)
        {
            T mx = mu_x / mu;
            T my = mu_y / mu;
            T mxx = mu_xx / mu - mx * mx;
            T mxy = mu_xy / mu - mx * my;
            T myy = mu_yy / mu - my * my;

            T theta = std::atan(2 * mxy / (mxx - myy)) / 2;
            if (myy > mxx) theta += M_PI_2;
            return theta;
        }
        return T();
    }

    Moments update_seed(const Point<T> & pt) const
    {
        auto dist = pt0 - pt;
        return {pt, mu, mu_x + dist.x * mu, mu_y + dist.y * mu,
                mu_xx + 2 * dist.x * mu_x + dist.x * dist.x * mu,
                mu_xy + dist.x * mu_y + dist.y * mu_x + dist.x * dist.y * mu,
                mu_yy + 2 * dist.y * mu_y + dist.y * dist.y * mu};
    }

    friend std::ostream & operator<<(std::ostream & os, const Moments & m)
    {
        os << "{" << m.x0 << ", " << m.y0 << ", " << m.mu << ", " << m.mu_x  << ", " << m.mu_y << ", "
           << m.mu_xx << ", " << m.mu_xy << ", " << m.mu_yy << "}";
        return os;
    }
};

template <typename T>
struct Pixels
{
    Moments<T> moments;
    pset_t<T> pset;

    Pixels() = default;
    Pixels(const pset_t<T> & pset) : moments(Moments<T>::from_pset(pset)), pset(pset) {}
    Pixels(pset_t<T> && pset) : moments(Moments<T>::from_pset(pset)), pset(std::move(pset)) {}

    Pixels merge(const Pixels & pixels) const
    {
        pset_t<T> pint, pmrg;
        std::set_intersection(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pint, pint.begin()));
        std::set_union(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pmrg, pmrg.begin()));

        auto m = moments + pixels.moments - Moments<T>::from_pset(moments.pt0, pint);

        if (m.mu) return {pmrg, m.update_seed(moments.center_of_mass())};
        return {pmrg, m};
    }

    void insert(Pixels && pixels)
    {
        pset_t<T> pint;
        std::set_intersection(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pint, pint.begin()));

        moments += pixels.moments - Moments<T>::from_pset(moments.pt0, pint);

        pset.insert(std::make_move_iterator(pixels.pset.begin()), std::make_move_iterator(pixels.pset.end()));
    }

    Line<T> get_line() const
    {
        if (moments.mu)
        {
            T theta = moments.theta();
            Point<T> tau {std::cos(theta), std::sin(theta)};
            Point<T> ctr = moments.center_of_mass();

            T tmin = std::numeric_limits<T>::max(), tmax = std::numeric_limits<T>::lowest();
            for (const auto & [pt, _] : pset)
            {
                T prod = tau.dot(pt - ctr);
                if (prod < tmin) tmin = prod;
                if (prod > tmax) tmax = prod;
            }

            return {ctr + tmin * tau, ctr + tmax * tau};
        }
        return {moments.pt0, moments.pt0};
    }
};

template <typename T>
struct Pattern;

template <typename T>
struct Streak
{
    Pixels<T> pixels;
    std::vector<Line<T>> linelets;
    Line<T> line;

    Streak(const pset_t<T> & pset) : pixels(pset), line(pixels.get_line())
    {
        linelets.push_back(line);
    }

    Streak(pset_t<T> && pset) : pixels(std::move(pset)), line(pixels.get_line())
    {
        linelets.push_back(line);
    }

    // void grow(const Pattern<T> & pattern, T xtol, T vmin, unsigned max_iter, unsigned lookahead)
    // {
    //     for (unsigned i = 0; i < max_iter; i++)
    //     {
    //         Streak streak0 {pattern.get_pset(line.pt0)};

    //     }
    // }

    void insert(Streak && streak)
    {
        pixels.insert(std::move(streak.pixels));
        linelets.insert(linelets.end(), std::make_move_iterator(streak.linelets.begin()),
                                        std::make_move_iterator(streak.linelets.end()));
    }

    void update_line()
    {
        line = pixels.get_line();
    }

private:
    void grow_step(Line<T> && streak, const array<T> & data, T xtol, T vmin)
    {
        if (streak.line.magnitude())
        {
            T d0 = line.distance(streak.line.pt0);
            T d1 = line.distance(streak.line.pt1);
            if (d0 < xtol && d1 < xtol)
            {
                auto pt = static_cast<Point<int>>(streak.pixels.moments.pt0.round());
                if (data.is_inbound({pt.y, pt.x}))
                {
                    if (data[data.ravel_index({pt.y, pt.x})] > vmin) insert(std::move(streak));
                }
            }
        }
    }
};

struct Structure
{
    using struct_t = std::vector<std::array<int, 2>>;
    int radius, rank;
    struct_t idxs;

    Structure(int radius, int rank) : radius(radius), rank(rank)
    {
        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                if (std::abs(i) + std::abs(j) <= rank) idxs.push_back({i, j});
            }
        }
        std::sort(idxs.begin(), idxs.end());
    }

    std::string info() const
    {
        return "<Structure, radius = " + std::to_string(radius) + ", rank = " + std::to_string(rank) + ">";
    }
};

template <typename T>
struct Pattern
{
    array<T> data;
    Structure structure;

    template <typename Array, typename Struct,
        typename = std::enable_if_t<
            std::is_base_of_v<array<T>, std::remove_cvref_t<Array>> &&
            std::is_base_of_v<Structure, std::remove_cvref_t<Struct>>
        >
    >
    Pattern(Array && d, Struct && s) : data(std::forward<Array>(d)), structure(std::forward<Struct>(s))
    {
        if (data.ndim != 2)
        {
            throw std::invalid_argument("Pattern data array has invalid number of dimensions (" +
                                        std::to_string(data.ndim) + ")");
        }
    }

    Pattern(const py::array_t<T> & d, Structure && s) : Pattern(array<T>{d.request()}, std::move(s)) {}

    pset_t<T> get_pset(int x, int y) const
    {
        pset_t<T> pset;
        for (auto shift : structure.idxs)
        {
            int new_x = x + shift[1], new_y = y + shift[0];
            if (data.is_inbound({new_y, new_x}))
            {
                pset.emplace_hint(pset.end(), Point<int>{new_x, new_y}, data[data.ravel_index({new_y, new_x})]);
            }
        }
        return pset;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    pset_t<T> get_pset(const Point<I> & point) const
    {
        return get_pset(point.x, point.y);
    }

    Streak<T> get_streak(int x, int y) const {return get_pset(x, y);}

    std::string info() const
    {
        return "<Pattern, data.shape = (" + std::to_string(data.shape[0]) + ", " + std::to_string(data.shape[1]) +
               "), struct = " + structure.info() + ">";
    }
};

struct DetState
{
    std::list<std::array<size_t, 2>> peaks;
    std::vector<bool> used;

    template <typename ShapeContainer>
    DetState(const ShapeContainer & shape) : used(get_size(shape.begin(), shape.end()), false) {}

    template <typename T>
    DetState(const array<T> & data, size_t radius, T vmin) : DetState(data.shape)
    {
        std::array<size_t, 2> axes {0, 1};
        std::unordered_map<std::array<size_t, 2>, std::array<size_t, 2>, detail::ArrayHasher<size_t, 2>> peak_map;
        auto add_peak = [&data, &peak_map, radius, vmin](size_t index)
        {
            auto i = data.index_along_dim(index, 0);
            auto j = data.index_along_dim(index, 1);
            if (data[data.ravel_index({i, j})] > vmin)
            {
                peak_map.try_emplace(std::array<size_t, 2>{i / radius, j / radius}, std::array<size_t, 2>{i, j});
            }
        };

        for (auto axis : axes)
        {
            for (size_t i = radius / 2; i < data.shape[1 - axis]; i += radius)
            {
                maxima1d(data.line_begin(axis, i), data.line_end(axis, i), add_peak, data, axes);
            }
        }

        std::transform(std::make_move_iterator(peak_map.begin()),
                       std::make_move_iterator(peak_map.end()),
                       std::back_inserter(peaks),
                       [](std::pair<std::array<size_t, 2>, std::array<size_t, 2>> && elem){return std::move(elem.second);});
    }

    template <typename T>
    DetState(py::array_t<T> d, size_t radius, T vmin) : DetState(array<T>(d.request()), radius, vmin) {}

    template <typename T>
    void filter(const array<T> & data, const Structure & s, T vmin, size_t npts)
    {
        std::set<std::array<size_t, 2>> support;
        std::set<std::array<size_t, 2>> new_points;
        auto iter = peaks.begin();

        while (iter != peaks.end())
        {
            support.clear(); new_points.clear();

            size_t prev_size = 0;
            support.insert(*iter);

            while (support.size() != prev_size && support.size() < npts)
            {
                prev_size = support.size();

                for (const auto & point: support)
                {
                    for (const auto & shift: s.idxs)
                    {
                        int i = point[0] + shift[0], j = point[1] + shift[1];

                        if (data.is_inbound({i, j}) && data[data.ravel_index({i, j})] > vmin)
                        {
                            new_points.insert({static_cast<size_t>(i), static_cast<size_t>(j)});
                        }
                    }
                }

                support.merge(std::move(new_points)); new_points.clear();
            }

            if (support.size() < npts) iter = peaks.erase(iter);
            else ++iter;
        }
    }

    std::string info() const
    {
        return "<DetState, peaks.shape = (" + std::to_string(peaks.size()) + ", 2)>";
    }
};

template <typename T>
std::array<T, 4> test(std::array<int, 4> pts, py::array_t<T> data, int radius, int rank);

}

#endif