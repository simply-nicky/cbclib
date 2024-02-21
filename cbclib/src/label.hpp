#ifndef LABEL_H_
#define LABEL_H_
#include "array.hpp"
#include "image_proc.hpp"

namespace cbclib {

using point_t = Point<long>;

template <typename T>
using pixel_t = std::tuple<point_t, T>;

template <typename T>
using pset_t = std::set<pixel_t<T>>;

namespace detail {

template <typename Container, typename = std::enable_if_t<is_point_v<typename Container::value_type>>>
auto get_x(const Container & list)
{
    using T = Container::value_type::value_type;
    std::vector<T> x;
    std::transform(list.begin(), list.end(), std::back_inserter(x), [](const Point<T> & elem){return elem.x();});
    return x;
}

template <typename Container, typename = std::enable_if_t<is_point_v<typename Container::value_type>>>
auto get_y(const Container & list)
{
    using T = Container::value_type::value_type;
    std::vector<T> y;
    std::transform(list.begin(), list.end(), std::back_inserter(y), [](const Point<T> & elem){return elem.y();});
    return y;
}

}

template <typename T, typename Pt, typename I = std::remove_cvref_t<Pt>::value_type,
    typename = std::enable_if_t<
        std::is_base_of_v<Point<I>, std::remove_cvref_t<Pt>> && std::is_integral_v<I>
    >
>
pixel_t<T> make_pixel(Pt && point, const array<T> & data)
{
    auto index = data.ravel_index(point.coordinate());
    return std::make_tuple(std::forward<Pt>(point), data[index]);
}

// Image moments class

template <typename T>
struct CentralMoments
{
    T mu_x, mu_y, mu_xx, mu_xy, mu_yy;

    Point<T> center_of_mass(const Point<T> & origin) const
    {
        return {mu_x + origin.x(), mu_y + origin.y()};
    }

    T theta() const
    {
        T theta = std::atan(2 * mu_xy / (mu_xx - mu_yy)) / 2;
        if (mu_yy > mu_xx) theta += M_PI_2;
        return detail::modulo(theta, M_PI);
    }

    std::array<T, 3> gauss() const
    {
        T div = 2 * (mu_xx * mu_yy - mu_xy * mu_xy);
        return {mu_yy / div, mu_xx / div, -mu_xy / div};
    }

    std::array<T, 2> principal_axes() const
    {
        auto [a, b, c] = gauss();
        T p = std::sqrt((a - b) * (a - b) + 4 * c * c);
        return {std::sqrt(1 / (a + b - p)), std::sqrt(1 / (a + b + p))};
    }
};

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
        mu_x += dist.x() * val;
        mu_y += dist.y() * val;
        mu_xx += dist.x() * dist.x() * val;
        mu_xy += dist.x() * dist.y() * val;
        mu_yy += dist.y() * dist.y() * val;
    }

    CentralMoments<T> central_moments() const
    {
        if (mu)
        {
            T mx = mu_x / mu;
            T my = mu_y / mu;
            return CentralMoments<T>{mx, my, mu_xx / mu - mx * mx,
                                     mu_xy / mu - mx * my, mu_yy / mu - my * my};
        }
        return CentralMoments<T>{};
    }

    Moments update_seed(const Point<T> & pt) const
    {
        auto dist = pt0 - pt;
        return {pt, mu, mu_x + dist.x() * mu, mu_y + dist.y() * mu,
                mu_xx + 2 * dist.x() * mu_x + dist.x() * dist.x() * mu,
                mu_xy + dist.x() * mu_y + dist.y() * mu_x + dist.x() * dist.y() * mu,
                mu_yy + 2 * dist.y() * mu_y + dist.y() * dist.y() * mu};
    }

    friend std::ostream & operator<<(std::ostream & os, const Moments & m)
    {
        os << "{" << m.x0 << ", " << m.y0 << ", " << m.mu << ", " << m.mu_x  << ", " << m.mu_y << ", "
           << m.mu_xx << ", " << m.mu_xy << ", " << m.mu_yy << "}";
        return os;
    }
};

// Container of points (Python interface)
template <class Container, typename = std::enable_if_t<is_point_v<typename Container::value_type>>>
struct PointsContainer
{
    using container_type = Container;
    using point_type = Container::value_type;
    using value_type = Container::value_type::value_type;

    container_type points;

    PointsContainer() = default;

    template <typename Pts, typename = std::enable_if_t<std::is_same_v<Container, std::remove_cvref_t<Pts>>>>
    PointsContainer(Pts && p) : points(std::forward<Pts>(p)) {}

    operator container_type && () && { return std::move(points); }

    container_type & operator*() {return points;}
    const container_type & operator*() const {return points;}

    container_type * operator->() {return &(points);}
    const container_type * operator->() const {return &(points);}

    std::vector<value_type> x() const {return detail::get_x(points);}
    std::vector<value_type> y() const {return detail::get_y(points);}

    std::string info() const
    {
        return "<Points, size = " + std::to_string(points.size()) + ">";
    }
};

struct PointsList : public PointsContainer<std::vector<point_t>>
{
    using PointsContainer::PointsContainer;
};

// Connectivity structure class

struct Structure : public PointsContainer<std::set<point_t>>
{
    int radius, rank;

    Structure(int radius, int rank) : radius(radius), rank(rank)
    {
        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                if (std::abs(i) + std::abs(j) <= rank) points.emplace_hint(points.end(), point_type{j, i});
            }
        }
    }

    std::string info() const
    {
        return "<Structure, radius = " + std::to_string(radius) + ", rank = " + std::to_string(rank) + 
                         ", points = <Points, size = " + std::to_string(points.size()) + ">>";
    }
};

// Extended interface of set of points - needed for Regions

struct PointsSet : public PointsContainer<std::set<point_t>>
{
    using PointsContainer::PointsContainer;

    template <typename Func, typename = std::enable_if_t<std::is_invocable_r_v<bool, std::remove_cvref_t<Func>, point_type>>>
    PointsSet(point_type seed, Func && func, const Structure & srt)
    {
        if (std::forward<Func>(func)(seed))
        {
            std::vector<point_type> last_pixels;
            std::vector<point_type> new_pixels;

            last_pixels.push_back(seed);

            while (last_pixels.size())
            {
                for (const auto & point: last_pixels)
                {
                    for (const auto & shift: srt.points)
                    {
                        auto pt = point + shift;

                        if (std::forward<Func>(func)(pt))
                        {
                            auto [iter, is_added] = points.insert(std::move(pt));
                            if (is_added) new_pixels.push_back(*iter);
                        }
                    }
                }

                last_pixels = std::move(new_pixels);
                new_pixels.clear();
            }
        }
    }

    PointsSet(Point<size_t> seed, const array<bool> & mask, const Structure & srt) :
        PointsSet(seed, [&mask](point_type pt){return mask.is_inbound(pt.coordinate()) && mask[mask.ravel_index(pt.coordinate())];}, srt) {}

    PointsSet filter(array<bool> & mask, const Structure & srt, size_t npts) const
    {
        PointsSet result;

        for (const auto & point : points)
        {
            if (mask.is_inbound(point.coordinate()) && mask[mask.ravel_index(point.coordinate())])
            {
                PointsSet support {point, mask, srt};

                for (auto pt : *support) mask[mask.ravel_index(pt.coordinate())] = false;

                if (support->size() >= npts) result->merge(std::move(*support));
            }
        }

        return result;
    }

    template <typename Mask, typename = std::enable_if_t<std::is_base_of_v<array<bool>, std::remove_cvref_t<Mask>>>>
    Mask && mask(Mask && m) const
    {
        for (auto pt : points)
        {
            if (m.is_inbound(pt.coordinate())) m[m.ravel_index(pt.coordinate())] = true;
        }
        return std::forward<Mask>(m);
    }
};

// Set of [point, value] pairs

template <typename T>
struct Pixels
{
    Moments<T> moments;
    pset_t<T> pset;

    Pixels() = default;
    Pixels(const pset_t<T> & pset) : moments(Moments<T>::from_pset(pset)), pset(pset) {}
    Pixels(pset_t<T> && pset) : moments(Moments<T>::from_pset(pset)), pset(std::move(pset)) {}

    Pixels(const PointsSet & points, const array<T> & data)
    {
        for (auto pt : points.points)
        {
            if (data.is_inbound(pt.coordinate())) pset.insert(make_pixel(pt, data));
        }
        moments = Moments<T>::from_pset(pset);
    }

    Pixels(PointsSet && points, const array<T> & data)
    {
        std::transform(std::make_move_iterator(points.points.begin()),
                       std::make_move_iterator(points.points.end()),
                       std::inserter(pset, pset.begin()),
                       [&data](auto && point){return make_pixel(std::forward<decltype(point)>(point), data);});
        moments = Moments<T>::from_pset(pset);
    }

    Pixels merge(const Pixels & pixels) const
    {
        pset_t<T> pint, pmrg;
        std::set_intersection(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pint, pint.begin()));
        std::set_union(pset.begin(), pset.end(), pixels.pset.begin(), pixels.pset.end(), std::inserter(pmrg, pmrg.begin()));

        auto m = moments + pixels.moments - Moments<T>::from_pset(moments.pt0, pint);

        if (m.mu) return {pmrg, m.update_seed(moments.central_moments().center_of_mass(moments.pt0))};
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
            auto cm = moments.central_moments();
            T theta = cm.theta();
            Point<T> tau {std::cos(theta), std::sin(theta)};
            Point<T> ctr = cm.center_of_mass(moments.pt0);

            T tmin = std::numeric_limits<T>::max(), tmax = std::numeric_limits<T>::lowest();
            for (const auto & [pt, _] : pset)
            {
                T prod = tau.dot(pt - ctr);
                if (prod < tmin) tmin = prod;
                if (prod > tmax) tmax = prod;
            }

            if (tmin != std::numeric_limits<T>::max() && tmax != std::numeric_limits<T>::lowest())
                return {ctr + tmin * tau, ctr + tmax * tau};
        }
        return {moments.pt0, moments.pt0};
    }
};

struct Regions
{
    std::array<size_t, 2> shape;
    std::vector<PointsSet> regions;

    template <typename Shape, typename Rgn, typename I = std::remove_cvref_t<Shape>::value_type, typename = std::enable_if_t<
        std::is_same_v<std::array<I, 2>, std::remove_cvref_t<Shape>> && 
        std::is_same_v<std::vector<PointsSet>, std::remove_cvref_t<Rgn>> &&
        std::is_integral_v<I>
    >>
    Regions(Shape && s, Rgn && r) : shape(std::forward<Shape>(s)), regions(std::forward<Rgn>(r)) {}

    template <typename Shape, typename I = std::remove_cvref_t<Shape>::value_type, typename = std::enable_if_t<
        std::is_same_v<std::array<I, 2>, std::remove_cvref_t<Shape>> && std::is_integral_v<I>
    >>
    Regions(Shape && s) : Regions(std::forward<Shape>(s), std::vector<PointsSet>()) {}

    std::vector<PointsSet> & operator*() {return regions;}
    const std::vector<PointsSet> & operator*() const {return regions;}

    std::vector<PointsSet> * operator->() {return &(regions);}
    const std::vector<PointsSet> * operator->() const {return &(regions);}

    std::string info() const
    {
        return "<Regions, shape = {" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
               "}, regions = <List[PointsSet], size = " + std::to_string(regions.size()) + ">>";
    }

    Regions filter(const Structure & str, size_t npts) const
    {
        auto vec = mask();
        array<bool> mask {shape, reinterpret_cast<bool *>(vec.data())};

        Regions result {shape};
        for (const auto & region : regions)
        {
            auto new_region = region.filter(mask, str, npts);
            if (new_region.points.size()) result->emplace_back(std::move(new_region));
        }
        return result;
    }

    std::vector<unsigned char> mask() const
    {
        std::vector<unsigned char> vec (shape[0] * shape[1], false);

        array<bool> mask {shape, reinterpret_cast<bool *>(vec.data())};
        for (const auto & region : regions) mask = region.mask(mask);

        return vec;
    }

    template <typename T>
    auto center_of_mass(const array<T> & data) const
    {
        return apply(data, [](const Pixels<T> & region) -> std::array<T, 2>
        {
            return region.moments.central_moments().center_of_mass(region.moments.pt0);
        });
    }

    template <typename T>
    auto gauss_fit(const array<T> & data) const
    {
        return apply(data, [](const Pixels<T> & region)
        {
            return region.moments.central_moments().gauss();
        });
    }

    template <typename T>
    auto ellipse_fit(const array<T> & data) const
    {
        return apply(data, [](const Pixels<T> & region)
        {
            auto cm = region.moments.central_moments();
            auto [a, b] = cm.principal_axes();
            auto theta = cm.theta();
            return std::array<T, 3>{a, b, theta};
        });
    }

    template <typename T>
    auto line_fit(const array<T> & data) const
    {
        return apply(data, [](const Pixels<T> & region) -> std::array<T, 4>
        {
            return region.get_line();
        });
    }

    template <typename T>
    auto moments(const array<T> & data) const
    {
        return apply(data, [](const Pixels<T> & region)
        {
            auto cm = region.moments.central_moments();
            return std::array<T, 4>{region.moments.mu, cm.mu_xx, cm.mu_xy, cm.mu_yy};
        });
    }

private:
    template <typename T, typename Func>
    auto apply(const array<T> & data, Func && func) const
    {
        std::vector<std::invoke_result_t<Func, Pixels<T>>> results;
        for (const auto & region : regions)
        {
            results.emplace_back(std::forward<Func>(func)(Pixels<T>{region, data}));
        }

        return results;
    }
};

template <typename T>
auto label(py::array_t<bool> mask, Structure structure, size_t npts, std::optional<std::tuple<size_t, size_t>> ax, unsigned threads);

}

#endif