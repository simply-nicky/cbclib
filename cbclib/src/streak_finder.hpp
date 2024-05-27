#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "array.hpp"
#include "label.hpp"
#include "kd_tree.hpp"
#include "signal_proc.hpp"

namespace cbclib {

namespace detail{

// Taken from the boost::hash_combine: https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
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
        std::unordered_map<point_type, point_type, detail::PointHasher<value_type>> peak_map;
        auto add_peak = [&data, &good, &peak_map, radius, vmin](size_t index)
        {
            value_type y = data.index_along_dim(index, 0);
            value_type x = data.index_along_dim(index, 1);
            if (good.at(y, x) && data.at(y, x) > vmin)
            {
                peak_map.try_emplace(point_type{x / static_cast<value_type>(radius), y / static_cast<value_type>(radius)}, point_type{x, y});
            }
        };

        for (auto axis : axes)
        {
            for (size_t i = radius / 2; i < data.shape[1 - axis]; i += radius)
            {
                maxima_nd(data.line_begin(axis, i), data.line_end(axis, i), add_peak, data, axes, 1);
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

    Peaks mask(bool (*is_good)(point_type)) const
    {
        container_type result;

        for (const auto & point : points)
        {
            if (is_good(point)) result.push_back(point);
        }

        return Peaks(std::move(result));
    }

    template <typename Func, typename = std::enable_if_t<std::is_invocable_r_v<bool, std::remove_cvref_t<Func>, point_type>>>
    Peaks mask(Func && func) const
    {
        container_type result;

        for (const auto & point : points)
        {
            if (std::forward<Func>(func)(point)) result.push_back(point);
        }

        return Peaks(std::move(result));
    }

    template <typename T>
    void sort(const array<T> & data)
    {
        // Sorting peaks in descending order
        std::sort(points.begin(), points.end(), [&data](point_type a, point_type b)
        {
            return data.at(a.coordinate()) > data.at(b.coordinate());
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
    using point_type = point_t;
    using integer_type = point_type::value_type;

    Pixels<T> pixels;
    std::map<T, point_type> centers;
    point_type center;
    std::map<T, Point<T>> points;
    Point<T> tau;

    template <typename PSet, typename Pt, typename = std::enable_if_t<
        std::is_same_v<pset_t<T>, std::remove_cvref_t<PSet>> &&
        std::is_constructible_v<point_type, std::remove_cvref_t<Pt>>
    >>
    Streak(PSet && pset, Pt && ctr) : pixels(std::forward<PSet>(pset)), center(std::forward<Pt>(ctr))
    {
        auto line = pixels.get_line();
        tau = line.tau;

        centers.emplace(make_pair(center));
        points.emplace(make_pair(line.pt0));
        points.emplace(make_pair(line.pt1));
    }

    void insert(Streak && streak)
    {
        pixels.insert(std::move(streak.pixels));
        update();
        for (auto && [_, pt] : streak.centers) centers.emplace(make_pair(std::forward<decltype(pt)>(pt)));
        for (auto && [_, pt] : streak.points) points.emplace(make_pair(std::forward<decltype(pt)>(pt)));
    }

    Line<integer_type> central_line() const
    {
        if (!centers.size())
            throw std::runtime_error("Streak object has no centers");
        return Line<integer_type>{centers.begin()->second, std::prev(centers.end())->second};
    }

    Line<T> line() const
    {
        return pixels.get_line();
    }

    void update()
    {
        tau = line().tau;
        std::map<T, Point<T>> new_points;
        std::map<T, point_type> new_centers;
        for (auto && [_, pt]: points) new_points.emplace_hint(new_points.end(), make_pair(std::forward<decltype(pt)>(pt)));
        for (auto && [_, pt]: centers) new_centers.emplace_hint(new_centers.end(), make_pair(std::forward<decltype(pt)>(pt)));
        points = std::move(new_points);
        centers = std::move(new_centers);
    }

private:
    template <typename Pt, typename = std::enable_if_t<std::is_convertible_v<Point<T>, std::remove_cvref_t<Pt>>>>
    auto make_pair(Pt && point) const
    {
        return std::make_pair(dot(tau, point - center), std::forward<Pt>(point));
    }
};

template <typename T>
struct StreakFinderResult
{
    enum flags
    {
        bad = -1,
        not_used = 0
    };

    vector_array<int> mask;
    std::vector<size_t> idxs;
    std::map<int, Streak<T>> streaks;

    using point_type = Peaks::point_type;
    using integer_type = typename point_type::value_type;

    using streak_iterator = std::map<int, Streak<T>>::iterator;
    using const_streak_iterator = std::map<int, Streak<T>>::const_iterator;

    StreakFinderResult(const array<T> & d, const array<bool> & m)
    {
        if (m.ndim != 2)
            throw std::invalid_argument("StreakFinder mask array has invalid number of dimensions (" +
                                        std::to_string(m.ndim) + ")");
        check_equal("data and mask have incompatible shapes", d.shape.begin(), d.shape.end(), m.shape.begin(), m.shape.end());

        std::vector<int> mvec;
        for (size_t i = 0; i < m.size; i++)
        {
            if (m[i]) {mvec.push_back(flags::not_used); idxs.push_back(i);}
            else mvec.push_back(flags::bad);
        }
        mask = vector_array<int>(std::move(mvec), m.shape);

        std::sort(idxs.begin(), idxs.end(), [&d](size_t i1, size_t i2){return d[i1] < d[i2];});
    }

    streak_iterator erase(const_streak_iterator pos)
    {
        erase_mask(pos->second.pixels.pset, pos->first);
        return streaks.erase(pos);
    }

    std::pair<streak_iterator, bool> insert(std::pair<int, Streak<T>> && elem)
    {
        auto [iter, is_added] = streaks.emplace(std::move(elem));
        if (is_added)
        {
            insert_mask(iter->second.pixels.pset, iter->first);
        }
        return std::make_pair(iter, is_added);
    }

    bool is_bad(const point_type & point) const
    {
        if (mask.is_inbound(point.coordinate()))
        {
            return mask.at(point.coordinate()) == flags::bad;
        }
        return true;
    }

    bool is_free(const point_type & point) const
    {
        if (mask.is_inbound(point.coordinate()))
        {
            return mask.at(point.coordinate()) == flags::not_used;
        }
        return false;
    }

    T line_minimum(const array<T> & data, const Line<integer_type> & line, T default_value) const
    {
        if (magnitude(line.tau))
        {
            T minimum = std::numeric_limits<T>::max();

            BresenhamIterator<integer_type, true> pix {line.norm(), line};
            point_type end = BresenhamTraits<integer_type, true>::end(line);

            do
            {
                pix.step_xy();

                if (mask.is_inbound(pix.point.coordinate()))
                {
                    auto index = mask.ravel_index(pix.point.coordinate());

                    if (mask[index] != flags::bad && data[index] < minimum)
                    {
                        minimum = data[index];
                    }
                }

                if (pix.is_xnext())
                {
                    pix.x_is_next();
                }
                if (pix.is_ynext())
                {
                    pix.y_is_next();
                }
            }
            while (!pix.is_end(end));

            if (minimum == std::numeric_limits<T>::max()) return default_value;
            return minimum;
        }

        return default_value;
    }

    T probability(const array<T> & data, T vmin) const
    {
        auto index = std::distance(idxs.begin(), std::lower_bound(idxs.begin(), idxs.end(), vmin, [&data](size_t index, T val){return data[index] < val;}));
        return T(1) - T(index) / idxs.size();
    }

    T p_value(const Streak<T> & streak, T xtol, T vmin, T p) const
    {
        return p_value(streak.pixels.pset, streak.central_line(), xtol, vmin, p, flags::not_used);
    }

    T p_value(const_streak_iterator iter, T xtol, T vmin, T p) const
    {
        return p_value(iter->second.pixels.pset, iter->second.central_line(), xtol, vmin, p, iter->first);
    }

    std::string info() const
    {
        return "<StreakFinderResult, mask = <array, shape = {" + std::to_string(mask.shape[0]) + ", " + std::to_string(mask.shape[1]) +
               "}>, idxs = <list, size = " + std::to_string(idxs.size()) + ">, streaks = <dict, size = " + std::to_string(streaks.size()) + ">>";
    }

private:
    void erase_mask(const pset_t<T> & pset, int flag)
    {
        for (auto [pt, _] : pset)
        {
            if (mask.is_inbound(pt.coordinate()))
            {
                auto index = mask.ravel_index(pt.coordinate());
                if (mask[index] != flags::bad) mask[index] -= flag;
            }
        }
    }

    void insert_mask(const pset_t<T> & pset, int flag)
    {
        for (auto [pt, _] : pset)
        {
            if (mask.is_inbound(pt.coordinate()))
            {
                auto index = mask.ravel_index(pt.coordinate());
                if (mask[index] != flags::bad) mask[index] += flag;
            }
        }
    }

    template <typename U>
    T p_value(const pset_t<T> & pset, const Line<U> & line, T xtol, T vmin, T p, int flag) const
    {
        size_t n = 0, k = 0;
        for (auto [pt, val] : pset)
        {
            if (mask.is_inbound(pt.coordinate()))
            {
                if (mask.at(pt.coordinate()) == flag && line.distance(pt) < xtol)
                {
                    n++;
                    if (val > vmin) k++;
                }
            }
        }

        return detail::logbinom(n, k, p);
    }
};

struct StreakFinder
{
    using point_type = Peaks::point_type;
    using integer_type = typename point_type::value_type;

    Structure structure;
    unsigned min_size;
    unsigned lookahead;
    unsigned nfa;

    template <typename Struct, typename = std::enable_if_t<std::is_base_of_v<Structure, std::remove_cvref_t<Struct>>>>
    StreakFinder(Struct && s, unsigned ms, unsigned lad, unsigned nf) :
        structure(std::forward<Struct>(s)), min_size(ms), lookahead(lad), nfa(nf) {}

    template <typename T>
    StreakFinderResult<T> detect_streaks(StreakFinderResult<T> && result, const array<T> & data, Peaks peaks, T xtol, T vmin)
    {
        auto p = result.probability(data, vmin);
        auto log_eps = std::log(p) * min_size;

        std::map<std::pair<size_t, int>, typename StreakFinderResult<T>::streak_iterator> streaks;
        int cnt = 0;

        auto is_good = [&result](const point_type & point)
        {
            return result.is_free(point);
        };

        while (peaks.points.size())
        {
            auto seed = *peaks.points.begin();
            peaks.erase(peaks.points.begin());

            auto streak = get_streak(seed, result, data, peaks, xtol, vmin);
            if (result.p_value(streak, xtol, vmin, p) < log_eps)
            {
                auto [iter, is_added] = result.insert(std::make_pair(++cnt, std::move(streak)));

                if (is_added)
                {
                    peaks = peaks.mask(is_good);
                    streaks.emplace(std::make_pair(iter->second.pixels.pset.size(), iter->first), iter);
                }
            }
        }

        for (auto [key, iter] : streaks)
        {
            if (result.p_value(iter, xtol, vmin, p) >= log_eps) result.erase(iter);
        }

        return std::move(result);
    }

    template <typename T>
    Streak<T> get_streak(const point_type & seed, const StreakFinderResult<T> & result, const array<T> & data, Peaks peaks, T xtol, T vmin) const
    {
        Streak<T> streak {get_pset(result, data, seed), seed};

        size_t old_size = 0;
        while (streak.points.size() != old_size)
        {
            old_size = streak.points.size();

            streak = grow_streak<T, false>(std::move(streak), result, data, streak.central_line().pt0, peaks, xtol, vmin);
            streak = grow_streak<T, true>(std::move(streak), result, data, streak.central_line().pt1, peaks, xtol, vmin);
        }

        return streak;
    }

    template <typename T>
    pset_t<T> get_pset(const StreakFinderResult<T> & result, const array<T> & data, int x, int y) const
    {
        pset_t<T> pset;
        for (auto shift : structure.points)
        {
            point_type pt {x + shift.x(), y + shift.y()};

            if (!result.is_bad(pt)) pset.emplace_hint(pset.end(), make_pixel(std::move(pt), data));
        }
        return pset;
    }

    template <typename T, typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    pset_t<T> get_pset(const StreakFinderResult<T> & result, const array<T> & data, const Point<I> & point) const
    {
        return get_pset(result, data, point.x(), point.y());
    }

    std::string info() const
    {
        return "<StreakFinder, struct = " + structure.info() + ", min_size = " + std::to_string(min_size) +
               ", lookahead = " + std::to_string(lookahead) + ", nfa = " + std::to_string(nfa) + ">";
    }

private:
    template <typename T>
    std::pair<bool, Streak<T>> add_point_to_streak(Streak<T> && streak, const StreakFinderResult<T> & result, const array<T> & data, const point_type & pt, T xtol, T vmin) const
    {
        auto new_streak = streak;
        new_streak.insert(Streak<T>{get_pset(result, data, pt), pt});
        auto new_line = new_streak.central_line();

        auto is_close = [&new_line, xtol](const std::pair<T, Point<T>> & item)
        {
            return new_line.normal_distance(item.second) >= xtol;
        };
        auto num_unaligned = std::transform_reduce(new_streak.points.begin(), new_streak.points.end(), unsigned(), std::plus(), is_close);

        if (num_unaligned <= nfa && result.line_minimum(data, new_line, vmin) > vmin)
        {
            return std::make_pair(true, std::move(new_streak));
        }

        return std::make_pair(false, std::move(streak));
    }

    template <typename T, bool IsForward>
    point_type find_next_step(const Streak<T> & streak, const point_type & point, int max_cnt) const
    {
        auto line = streak.line();

        BresenhamIterator<T, IsForward> pix {line.norm(), point, point, line};

        for (int i = 0; i <= max_cnt; i++)
        {
            pix.step_xy();

            if (pix.is_xnext()) pix.x_is_next();
            if (pix.is_ynext()) pix.y_is_next();
        }

        return pix.point;
    }

    template <typename T, bool IsForward>
    Streak<T> grow_streak(Streak<T> && streak, const StreakFinderResult<T> & result, const array<T> & data, point_type point, const Peaks & peaks, T xtol, T vmin) const
    {
        unsigned tries = 0;

        while (tries <= lookahead)
        {
            point_type pt = find_next_step<T, IsForward>(streak, point, structure.rank);

            auto stack = peaks.tree.find_range(pt, structure.rank * structure.rank);
            std::sort(stack.begin(), stack.end(), [](auto a, auto b){return a.second > b.second;});
            for (auto [item, _] : stack)
            {
                if (item->point() != pt)
                {
                    if (result.is_free(item->point())) pt = item->point();
                }
            }

            if (!result.is_bad(pt))
            {
                auto [is_add, new_streak] = add_point_to_streak(std::move(streak), result, data, pt, xtol, vmin);

                if (is_add) return new_streak;
                else
                {
                    streak = std::move(new_streak);
                    tries++;
                }
            }
            else tries++;

            point = pt;
        }

        return streak;
    }
};

}

#endif
