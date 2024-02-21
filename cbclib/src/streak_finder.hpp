#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "array.hpp"
#include "label.hpp"
#include "kd_tree.hpp"
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
            h ^= std::hash<T>{}(elem) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }   
};

template <typename T>
struct PointHasher
{
    std::size_t operator()(const Point<T> & point) const
    {
        std::size_t h = 0;

        h ^= std::hash<T>{}(point.x()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<T>{}(point.y()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Return log(binomial_tail(n, k, p))
// binomial_tail(n, k, p) = sum_{i = k}^n bincoef(n, i) * p^i * (1 - p)^{n - i}
// bincoef(n, k) = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

template <typename I, typename T>
T logbinom(I n, I k, T p)
{
    if (n == k) return n * std::log(p);

    auto term = std::exp(std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1) +
                         k * std::log(p) + (n - k) * std::log(T(1.0) - p));
    auto bin_tail = term;
    auto p_term = p / (T(1.0) - p);

    for (I i = k + 1; i < n + 1; i++)
    {
        term *= (n - i + 1) / i * p_term;
        bin_tail += term;
    }

    return std::log(bin_tail);
}

}

// Sparse 2D peaks

struct Peaks : public PointsList
{
    using tree_type = KDTree<point_type, std::nullptr_t>;

    using iterator = container_type::iterator;
    using const_iterator = container_type::const_iterator;

    tree_type tree;

    Peaks() = default;

    template <typename Pts, typename = std::enable_if_t<std::is_same_v<container_type, std::remove_cvref_t<Pts>>>>
    Peaks(Pts && pts) : PointsList(std::forward<Pts>(pts))
    {
        std::vector<std::pair<point_type, std::nullptr_t>> items;
        std::transform(points.begin(), points.end(), std::back_inserter(items), [](point_type pt){return std::make_pair(pt, nullptr);});
        tree = tree_type(std::move(items));
    }

    template <typename T>
    Peaks(const array<T> & data, const array<bool> & good, size_t radius, T vmin)
    {
        check_data(data, good);

        std::array<size_t, 2> axes {0, 1};
        std::unordered_map<point_type, point_type, detail::PointHasher<point_type::value_type>> peak_map;
        auto add_peak = [&data, &good, &peak_map, radius, vmin](size_t index)
        {
            using I = point_type::value_type;

            I y = data.index_along_dim(index, 0);
            I x = data.index_along_dim(index, 1);
            if (good[data.ravel_index({y, x})] && data[data.ravel_index({y, x})] > vmin)
            {
                peak_map.try_emplace(point_type{x / static_cast<I>(radius), y / static_cast<I>(radius)}, point_type{x, y});
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
                       std::back_inserter(points),
                       [](std::pair<point_type, point_type> && elem){return std::move(elem.second);});

        std::vector<std::pair<point_type, std::nullptr_t>> items;
        std::transform(points.begin(), points.end(), std::back_inserter(items), [](point_type pt){return std::make_pair(pt, nullptr);});
        tree = tree_type(std::move(items));
    }

    template <typename T>
    Peaks(py::array_t<T> d, py::array_t<bool> g, size_t radius, T vmin)
        : Peaks(array<T>(d.request()), array<bool>(g.request()), radius, vmin) {}

    const_iterator begin() const {return points.begin();}
    const_iterator end() const {return points.end();}
    iterator begin() {return points.begin();}
    iterator end() {return points.end();}

    iterator erase(iterator pos)
    {
        auto iter = tree.find(*pos);
        if (iter != tree.end()) tree.erase(iter);
        return points.erase(pos);
    }

    template <typename T>
    Peaks filter(const array<T> & data, const array<bool> & good, const Structure & srt, T vmin, size_t npts) const
    {
        check_data(data, good);

        container_type result;

        auto func = [&data, &good, vmin](point_type pt)
        {
            if (data.is_inbound(pt.coordinate()))
            {
                auto idx = data.ravel_index(pt.coordinate());
                return good[idx] && data[idx] > vmin;
            }
            return false;
        };

        for (const auto & point : points)
        {
            PointsSet support {point, func, srt};

            if (support->size() >= npts) result.push_back(point);
        }

        return Peaks(std::move(result));
    }

    Peaks mask(const array<bool> & good) const
    {
        container_type result;

        for (const auto & point : points)
        {
            if (good[good.ravel_index(point.coordinate())]) result.push_back(point);
        }

        return Peaks(std::move(result));
    }

    template <typename T>
    void sort(const array<T> & data) 
    {
        // Sorting peaks in descending order
        std::sort(points.begin(), points.end(), [&data](point_type a, point_type b)
        {
            return data[data.ravel_index(a.coordinate())] > data[data.ravel_index(b.coordinate())];
        });
    }

    std::string info() const
    {
        return "<Peaks, points = <Points, size = " + std::to_string(points.size()) + ">>";
    }

private:
    template <typename T>
    void check_data(const array<T> & data, const array<bool> & good) const
    {
        if (data.ndim != 2)
        {
            throw std::invalid_argument("Pattern data array has invalid number of dimensions (" +
                                        std::to_string(data.ndim) + ")");
        }
        check_equal("data and good have incompatible shapes",
                    data.shape.begin(), data.shape.end(), good.shape.begin(), good.shape.end());
    }
};

// Streak class

template <typename T>
struct Streak
{
    Pixels<T> pixels;
    std::map<T, Point<T>> centers;
    std::map<T, Point<T>> points;
    Point<T> tau, center;

    template <typename PSet, typename = std::enable_if_t<std::is_same_v<pset_t<T>, std::remove_cvref_t<PSet>>>>
    Streak(PSet && pset) : pixels(std::forward<PSet>(pset))
    {
        auto line = pixels.get_line();
        center = pixels.moments.central_moments().center_of_mass(pixels.moments.pt0);
        tau = line.tau;

        centers.emplace(make_pair(center));
        points.emplace(make_pair(line.pt0));
        points.emplace(make_pair(line.pt1));
    }

    void insert(Streak && streak)
    {
        pixels.insert(std::move(streak.pixels));
        for (auto && [_, pt] : streak.centers) centers.emplace(make_pair(std::forward<decltype(pt)>(pt)));
        for (auto && [_, pt] : streak.points) points.emplace(make_pair(std::forward<decltype(pt)>(pt)));
    }

    Line<T> central_line() const
    {
        if (centers.size()) return Line<T>{centers.begin()->second, std::prev(centers.end())->second};
        return pixels.get_line();
    }

    Line<T> line() const
    {
        if (points.size()) return Line<T>{points.begin()->second, std::prev(centers.end())->second};
        return pixels.get_line();
    }

    T line_median(T dist, T default_value) const
    {
        auto ln = line();
        std::vector<T> vals;
        for (auto && [pt, val] : pixels.pset)
        {
            if (ln.distance(pt) <= dist) vals.emplace_back(val);
        }
        if (vals.size()) return *wirthmedian(vals.begin(), vals.end(), std::less<T>());
        return default_value;
    }

    T log_nfa(unsigned min_size, T xtol, T p) const
    {
        auto ln = pixels.get_line();
        size_t k = 0;
        for (auto [_, pt]: points) if (ln.distance(pt) < xtol) k++;
        for (auto [_, pt]: centers) if (ln.distance(pt) < xtol) k++;
        return -static_cast<int>(min_size) * std::log(p) + detail::logbinom(points.size() + centers.size(), k, p);
    }

    Line<T> extend_line(const Point<T> & pt) const
    {
        if (centers.size())
        {
            if (make_pair(pt).first < centers.begin()->first) return Line<T>{pt, std::prev(centers.end())->second};
            if (make_pair(pt).first > std::prev(centers.end())->first) return Line<T>{centers.begin()->second, pt};
        }
        return central_line();
    }

private:
    template <typename Pt, typename = std::enable_if_t<std::is_same_v<Point<T>, std::remove_cvref_t<Pt>>>>
    auto make_pair(Pt point) const
    {
        return std::make_pair(tau.dot(point - center), std::forward<Pt>(point));
    }
};

template <typename T>
struct Pattern
{
    using point_type = Peaks::point_type;

    array<T> data;
    array<bool> good;
    Structure structure;

    template <typename Data, typename Good, typename Struct,
        typename = std::enable_if_t<
            std::is_base_of_v<array<T>, std::remove_cvref_t<Data>> &&
            std::is_base_of_v<array<bool>, std::remove_cvref_t<Good>> &&
            std::is_base_of_v<Structure, std::remove_cvref_t<Struct>>
        >
    >
    Pattern(Data && d, Good && g, Struct && s) :
        data(std::forward<Data>(d)), good(std::forward<Good>(g)), structure(std::forward<Struct>(s))
    {
        if (data.ndim != 2)
        {
            throw std::invalid_argument("Pattern data array has invalid number of dimensions (" +
                                        std::to_string(data.ndim) + ")");
        }
        check_equal("data and good have incompatible shapes",
                    data.shape.begin(), data.shape.end(), good.shape.begin(), good.shape.end());
    }

    template <typename Struct,
        typename = std::enable_if_t<
            std::is_base_of_v<Structure, std::remove_cvref_t<Struct>>
        >
    >
    Pattern(const py::array_t<T> & d, const py::array_t<bool> & g, Struct && s) :
        Pattern(array<T>{d.request()}, array<bool>{g.request()}, std::forward<Struct>(s)) {}

    auto find_streaks(Peaks peaks, T xtol, T vmin, T log_eps, unsigned max_iter, unsigned lookahead, unsigned min_size)
    {
        std::vector<std::array<T, 4>> lines;

        while (peaks.points.size())
        {
            auto seed = get_pset(peaks.points.begin()->x(), peaks.points.begin()->y());

            if (seed.size() >= structure.points.size())
            {
                Streak<T> streak = get_streak(std::move(seed), peaks, xtol, vmin, max_iter, lookahead);

                if (streak.log_nfa(min_size, xtol, xtol / (structure.radius + 0.5)) < log_eps)
                {
                    update(streak.pixels.pset);
                    peaks = peaks.mask(good);
                    lines.emplace_back(streak.pixels.get_line());
                }
                else peaks.erase(peaks.points.begin());
            }
            else peaks.erase(peaks.points.begin());
        }

        return lines;
    }

    Streak<T> get_streak(pset_t<T> && seed, Peaks peaks, T xtol, T vmin, unsigned max_iter, unsigned lookahead) const
    {
        Streak<T> streak {std::move(seed)};

        for (unsigned i = 0; i < max_iter; i++)
        {
            auto old_size = streak.points.size();

            streak = grow_step(std::move(streak), streak.central_line().pt0, peaks, direction::backward, xtol, vmin, lookahead);
            streak = grow_step(std::move(streak), streak.central_line().pt1, peaks, direction::forward, xtol, vmin, lookahead);

            if (streak.points.size() == old_size) break;
        }

        return streak;
    }

    pset_t<T> get_pset(int x, int y) const
    {
        pset_t<T> pset;
        for (auto shift : structure.points)
        {
            point_type pt {x + shift.x(), y + shift.y()};

            if (data.is_inbound(pt.coordinate()) && good[good.ravel_index(pt.coordinate())])
            {
                pset.emplace_hint(pset.end(), make_pixel(std::move(pt), data));
            }
        }
        return pset;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    pset_t<T> get_pset(const Point<I> & point) const
    {
        return get_pset(point.x(), point.y());
    }

    std::string info() const
    {
        return "<Pattern, data.shape = (" + std::to_string(data.shape[0]) + ", " + std::to_string(data.shape[1]) +
               "), good.shape = (" + std::to_string(good.shape[0]) + ", " + std::to_string(good.shape[1]) +
               "), struct = " + structure.info() + ">";
    }

    void update(const pset_t<T> & pset)
    {
        for (auto [pt, _] : pset)
        {
            good[good.ravel_index(pt.coordinate())] = false;
        }
    }

private:
    std::pair<bool, Streak<T>> add_streak(Streak<T> && streak, const point_type & pt, T xtol, T vmin) const
    {
        Streak<T> new_streak {get_pset(pt)};
        Line<T> new_line {streak.extend_line(new_streak.center)};

        auto is_close = [&new_line, xtol](const std::pair<T, Point<T>> & item)
        {
            return new_line.distance(item.second) < xtol;
        };
        if (std::all_of(new_streak.points.begin(), new_streak.points.end(), is_close))
        {
            if (data.is_inbound(pt.coordinate()))
            {
                if (new_streak.line_median(M_SQRT2, vmin) > vmin)
                {
                    streak.insert(std::move(new_streak));
                    return std::make_pair(true, std::move(streak));
                }
            }
        }

        return std::make_pair(false, std::move(streak));
    }

    point_type find_next_step(const Streak<T> & streak, const Point<T> & point, int max_cnt, direction dir) const
    {
        auto line = streak.line();

        BhmIterator<T, point_type::value_type> pix {point.round(), line.norm(), point};
        auto step = bresenham_step(line.tau, dir);
        point_type new_step;

        for (int i = 0, cnt = 0; i < max_cnt; i++)
        {
            pix.step(new_step);
            new_step = point_type();

            if (pix.is_xnext(step))
            {
                new_step.x() = step.x(); cnt++;
                if (cnt > max_cnt) break;
            }
            if (pix.is_ynext(step))
            {
                new_step.y() = step.y(); cnt++;
            }
        }
    
        return pix.point;
    }

    Streak<T> grow_step(Streak<T> && streak, Point<T> point, const Peaks & peaks, direction dir, T xtol, T vmin, unsigned lookahead) const
    {
        unsigned max_gap = std::sqrt(streak.line().magnitude()) / structure.radius;
        unsigned tries = 0, gap = 0;
        T threshold;

        while (tries < lookahead && gap <= max_gap)
        {
            point_type pt = find_next_step(streak, point, 2 * structure.radius + 1, dir);

            auto query = peaks.tree.find_nearest(pt);
            if (query.second < structure.radius)
            {
                pt = query.first->point();
                threshold = std::numeric_limits<T>::lowest();
            }
            else threshold = vmin;

            if (!data.is_inbound(pt.coordinate())) break;

            if (good[good.ravel_index(pt.coordinate())])
            {
                auto [is_add, new_streak] = add_streak(std::move(streak), pt, xtol, threshold);

                if (is_add) return new_streak;
                else
                {
                    streak = std::move(new_streak);
                    tries++;
                }
            }
            else gap++;

            point = std::move(pt);
        }

        return streak;
    }
};

template <typename T>
std::array<T, 4> test_line(std::array<int, 4> pts, py::array_t<T> data, py::array_t<bool> mask, Structure srt);

template <typename T>
std::array<T, 4> test_grow(size_t index, Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure srt,
                           T xtol, T vmin, unsigned max_iter, unsigned lookahead);

}

#endif