#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "array.hpp"
#include "label.hpp"
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

        h ^= std::hash<T>{}(point.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<T>{}(point.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
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

struct Peaks
{
    std::list<Point<size_t>> points;

    template <typename T>
    Peaks(const array<T> & data, const array<bool> & good, size_t radius, T vmin)
    {
        check_data(data, good);

        std::array<size_t, 2> axes {0, 1};
        std::unordered_map<Point<size_t>, Point<size_t>, detail::PointHasher<size_t>> peak_map;
        auto add_peak = [&data, &good, &peak_map, radius, vmin](size_t index)
        {
            auto y = data.index_along_dim(index, 0);
            auto x = data.index_along_dim(index, 1);
            if (good[data.ravel_index({y, x})] && data[data.ravel_index({y, x})] > vmin)
            {
                peak_map.try_emplace(Point<size_t>{x / radius, y / radius}, Point<size_t>{x, y});
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
                       [](std::pair<Point<size_t>, Point<size_t>> && elem){return std::move(elem.second);});

        // Sorting peaks in descending order
        points.sort([&data](Point<size_t> a, Point<size_t> b)
        {
            return data[data.ravel_index(a.coordinate())] > data[data.ravel_index(b.coordinate())];
        });
    }

    template <typename T>
    Peaks(py::array_t<T> d, py::array_t<bool> g, size_t radius, T vmin)
        : Peaks(array<T>(d.request()), array<bool>(g.request()), radius, vmin) {}

    template <typename T>
    void filter(const array<T> & data, const array<bool> & good, const Structure & srt, T vmin, size_t npts)
    {
        check_data(data, good);

        std::set<Point<size_t>> support;
        auto iter = points.begin();

        while (iter != points.end())
        {
            support.clear();

            size_t prev_size = 0;
            support.insert(*iter);

            while (support.size() != prev_size && support.size() < npts)
            {
                prev_size = support.size();

                for (const auto & point : support)
                {
                    for (const auto & shift : srt.idxs)
                    {
                        auto pt = point + shift;

                        if (data.is_inbound(pt.coordinate()))
                        {
                            auto idx = data.ravel_index(pt.coordinate());
                            if (good[idx] && data[idx] > vmin) support.insert(std::move(pt));
                        }
                    }
                }
            }

            if (support.size() < npts) iter = points.erase(iter);
            else ++iter;
        }
    }

    void mask(const array<bool> & good)
    {
        auto iter = points.begin();
        while (iter != points.end())
        {
            if (!good[good.ravel_index(iter->coordinate())]) iter = points.erase(iter);
            else ++iter;
        }
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

// Final streak class

template <typename T>
struct Streak
{
    Pixels<T> pixels;
    std::vector<Line<T>> linelets;
    Line<T> line;

    Streak(const pset_t<T> & pset) : pixels(pset), line(pixels.get_line(std::numeric_limits<T>::lowest()))
    {
        linelets.push_back(line);
    }

    Streak(pset_t<T> && pset) : pixels(std::move(pset)), line(pixels.get_line(std::numeric_limits<T>::lowest()))
    {
        linelets.push_back(line);
    }

    void insert(Streak && streak)
    {
        pixels.insert(std::move(streak.pixels));
        linelets.insert(linelets.end(), std::make_move_iterator(streak.linelets.begin()),
                                        std::make_move_iterator(streak.linelets.end()));
    }

    T log_nfa(unsigned min_size, T xtol, T p) const
    {
        size_t k = 0;
        for (auto ll: linelets)
        {
            if (line.distance(ll.pt0) < xtol && line.distance(ll.pt1) < xtol) k++;
        }
        return -static_cast<int>(min_size) * std::log(p) + detail::logbinom(linelets.size(), k, p);
    }

    void update_line(T vmin)
    {
        line = pixels.get_line(vmin);
    }
};

template <typename T>
struct Pattern
{
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
            Streak<T> streak = get_streak(peaks.points.front(), xtol, vmin, max_iter, lookahead);

            if (streak.log_nfa(min_size, xtol, xtol / (structure.radius + 0.5)) < log_eps)
            {
                update(streak.pixels.pset);
                peaks.mask(good);
                lines.emplace_back(streak.line);
            }
            else peaks.points.pop_front();
        }

        return lines;
    }

    Streak<T> get_streak(int x, int y, T xtol, T vmin, unsigned max_iter, unsigned lookahead) const
    {
        Streak<T> streak {get_pset(x, y)};
        for (unsigned i = 0; i < max_iter; i++)
        {
            auto old_size = streak.linelets.size();

            streak = grow_step(std::move(streak), streak.line.pt0, direction::backward, xtol, vmin, lookahead);
            streak = grow_step(std::move(streak), streak.line.pt1, direction::forward, xtol, vmin, lookahead);

            if (streak.linelets.size() == old_size) break;

            streak.update_line(T());
        }

        streak.update_line(vmin);
        return streak;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Streak<T> get_streak(const Point<I> & point, T xtol, T vmin, unsigned max_iter, unsigned lookahead) const
    {
        return get_streak(point.x, point.y, xtol, vmin, max_iter, lookahead);
    }

    pset_t<T> get_pset(int x, int y) const
    {
        pset_t<T> pset;
        for (auto shift : structure.idxs)
        {
            Point<int> pt {x + shift.x, y + shift.y};

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
        return get_pset(point.x, point.y);
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
    std::pair<bool, Streak<T>> add_streak(Streak<T> && streak, const Point<int> & pt, T xtol, T vmin) const
    {
        Streak new_streak {get_pset(pt.x, pt.y)};
        if (new_streak.line.magnitude())
        {
            T d0 = streak.line.distance(new_streak.line.pt0);
            T d1 = streak.line.distance(new_streak.line.pt1);

            if (d0 < xtol && d1 < xtol)
            {
                Point<int> pt {new_streak.pixels.moments.pt0.round()};
                if (data.is_inbound(pt.coordinate()))
                {
                    if (data[data.ravel_index(pt.coordinate())] > vmin)
                    {
                        streak.insert(std::move(new_streak));
                        return std::make_pair(true, std::move(streak));
                    }
                }
            }
        }
        return std::make_pair(false, std::move(streak));
    }

    Point<int> find_next_step(const Streak<T> & streak, const Point<T> & point, int max_cnt, direction dir) const
    {
        BhmIterator<T, int> pix {point.round(), streak.line.norm(), point};
        auto step = bresenham_step(streak.line.tau, dir);

        Point<int> new_step;

        for (int cnt = 0; cnt <= max_cnt; cnt++)
        {
            pix.step(new_step);
            new_step = Point<int>();

            if (pix.is_xnext(step)) new_step.x = step.x;
            if (pix.is_ynext(step)) new_step.y = step.y;
        }

        return pix.point;
    }

    Streak<T> grow_step(Streak<T> && streak, Point<T> point, direction dir, T xtol, T vmin, unsigned lookahead) const
    {
        unsigned max_gap = std::sqrt(streak.line.magnitude()) / structure.radius;
        unsigned tries = 0, gap = 0;

        while (tries < lookahead && gap < max_gap)
        {
            Point<int> pt = find_next_step(streak, point, structure.radius, dir);

            if (!data.is_inbound(pt.coordinate())) break;

            if (good[good.ravel_index(pt.coordinate())])
            {
                auto [is_add, new_streak] = add_streak(std::move(streak), pt, xtol, vmin);

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
std::vector<std::array<T, 4>> detect_streaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure,
                                             T xtol, T vmin, T log_eps, unsigned max_iter, unsigned lookahead, size_t min_size);

template <typename T>
std::array<T, 4> test_line(std::array<int, 4> pts, py::array_t<T> data, py::array_t<bool> mask, Structure srt);

template <typename T>
std::array<T, 4> test_grow(int x, int y, py::array_t<T> data, py::array_t<bool> mask, Structure srt,
                           T xtol, T vmin, unsigned max_iter, unsigned lookahead);

}

#endif