/***
 * MIT License
 * Author: G Davey
 */

#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <typeinfo>

namespace zip {

template <typename Iter>
using select_access_type_for = std::conditional_t<
    std::is_same_v<Iter, std::vector<bool>::iterator> ||
    std::is_same_v<Iter, std::vector<bool>::const_iterator>,
    typename Iter::value_type,
    typename Iter::reference
>;


template <typename ... Args, std::size_t ... Index>
bool any_match_impl(const std::tuple<Args...> & lhs, const std::tuple<Args...> & rhs, std::index_sequence<Index...>)
{
    auto result = false;
    result = (... | (std::get<Index>(lhs) == std::get<Index>(rhs)));
    return result;
}

template <typename ... Args>
bool any_match(const std::tuple<Args...> & lhs, const std::tuple<Args...> & rhs)
{
    return any_match_impl(lhs, rhs, std::index_sequence_for<Args...>{});
}

template <typename ... Iters>
class zip_iterator
{
public:
    using value_type = std::tuple<select_access_type_for<Iters>...>;

    zip_iterator() = delete;

    zip_iterator(Iters && ... iters) : m_iters{std::forward<Iters>(iters)...} {}

    zip_iterator & operator++() 
    {
        std::apply([](auto && ... args){ ((args += 1), ...); }, m_iters);
        return *this;
    }

    zip_iterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    bool operator!=(const zip_iterator & other) const
    {
        return !(*this == other);
    }

    bool operator==(const zip_iterator & other) const
    {
        return any_match(m_iters, other.m_iters);
    }

    value_type operator*()
    {
        return std::apply([](auto && ... args){return value_type(*args...);}, m_iters);
    }

private:
    std::tuple<Iters...> m_iters;
};


/* std::decay needed because T is a reference, and is not a complete type */
template <typename T>
using select_iterator_for = std::conditional_t<
    std::is_const_v<std::remove_reference_t<T>>, 
    typename std::remove_cvref_t<T>::const_iterator,
    typename std::remove_cvref_t<T>::iterator>;



template <typename ... T>
class zipper
{
public:
    using zip_type = zip_iterator<select_iterator_for<T> ...>;

    template <typename ... Args>
    zipper(Args && ... args) : m_args{std::forward<Args>(args)...} {}

    zip_type begin()
    {
        return std::apply([](auto && ... args){return zip_type(std::begin(args)...);}, m_args);
    }

    zip_type end()
    {
        return std::apply([](auto && ... args){return zip_type(std::end(args)...);}, m_args);
    }

private:
    std::tuple<T ...> m_args;

};


template <typename ... T>
auto zip(T && ... t)
{
    return zipper<T ...>(std::forward<T>(t)...);
}

}