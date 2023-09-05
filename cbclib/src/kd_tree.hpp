#ifndef KD_TREE_
#define KD_TREE_
#include "array.hpp"

namespace cbclib {

namespace detail {

template<typename NodePtr>
NodePtr && min_node(NodePtr && a, NodePtr && b, NodePtr && c, int axis)
{
    if (b && b->point[axis] < a->point[axis])
    {
        if (c && c->point[axis] < b->point[axis]) return std::forward<NodePtr>(c);
        return std::forward<NodePtr>(b);
    }
    if (c && c->point[axis] < a->point[axis]) return std::forward<NodePtr>(c);
    return std::forward<NodePtr>(a);
}

template <typename NodePtr>
NodePtr && max_node(NodePtr && a, NodePtr && b, NodePtr && c, int axis)
{
    if (b && b->point[axis] > a->point[axis])
    {
        if (c && c->point[axis] > b->point[axis]) return std::forward<NodePtr>(c);
        return std::forward<NodePtr>(b);
    }
    if (c && c->point[axis] > a->point[axis]) return std::forward<NodePtr>(c);
    return std::forward<NodePtr>(a);
}

template <typename F>
F distance(const std::vector<F> & a, const std::vector<F> & b)
{
    F dist = F();
    for (size_t i = 0; i < a.size(); i++) dist += std::pow(a[i] - b[i], 2);
    return dist;
}

}

template <typename NodePtr>
NodePtr find_min(NodePtr node, int axis)
{
    if (!node) return std::move(node);

    if (node->cut_dim == axis)
    {
        if (!node->left) return std::move(node);
        else return find_min(node->left, axis);
    }
    else return detail::min_node(std::move(node), find_min(node->left, axis), find_min(node->right, axis), axis);
}

template <typename NodePtr>
NodePtr find_max(NodePtr node, int axis)
{
    if (!node) return std::move(node);

    if (node->cut_dim == axis)
    {
        if (!node->right) return std::move(node);
        else return find_max(node->right, axis);
    }
    else return detail::max_node(std::move(node), find_max(node->left, axis), find_max(node->right, axis), axis);
}

template <typename F, typename Data>
class KDNode
{
public:
    using coordinate_type = F;
    using data_type = Data;

    using Point = std::vector<F>;
    using node_ptr = std::shared_ptr<KDNode<F, Data>>;
    using data_ptr = std::shared_ptr<Data>;

    Point point;
    int cut_dim;
    data_ptr data;
    node_ptr left, right;

    template<typename Container, typename = std::enable_if_t<std::is_same_v<typename Container::value_type, F>>>
    KDNode(Container c, int dir, Data data = nullptr)
        : point(std::begin(c), std::end(c)), cut_dim(dir), data(std::make_shared<Data>(data)) {}


    bool is_left(const Point & pt) const
    {
        return pt[this->cut_dim] < this->point[this->cut_dim];
    }
};

template <typename It, typename T, typename Data, typename = void>
struct is_node_pointer : std::false_type {};

template <typename It, typename T, typename Data>
struct is_node_pointer<It, T, Data,
    typename std::enable_if_t<
        std::is_same_v<
            typename std::decay_t<It>,
            std::shared_ptr<KDNode<T, Data>>
        >
    >
> : std::true_type {};

template <typename It, typename T, typename Data>
constexpr bool is_node_pointer_v = is_node_pointer<It, T, Data>::value;

template <typename F>
class Rectangle
{
public:
    using Point = std::vector<F>;
    Point low, high;

    Rectangle(Point low, Point high) : low(std::move(low)), high(std::move(high)) {};

    template <
        class NodePtr,
        typename = std::enable_if_t<
            is_node_pointer_v<NodePtr, F, typename std::decay_t<NodePtr>::element_type::data_type>
        >
    >
    Rectangle(NodePtr node, size_t ndim) : low(ndim), high(ndim)
    {
        for (size_t i = 0; i < low.size(); i++)
        {
            this->low[i] = find_min(node, i)->point[i];
            this->high[i] = find_max(node, i)->point[i];
        }
    }

    Rectangle & update(const Point & pt)
    {
        for (size_t i = 0; i < pt.size(); i++)
        {
            this->low[i] = std::min(this->low[i], pt[i]);
            this->high[i] = std::max(this->high[i], pt[i]);
        }
        return *this;
    }

    F distance(const Point & pt) const
    {
        F dist = F();
        for (size_t i = 0; i < pt.size(); i++)
        {
            if (pt[i] < this->low[i]) dist += std::pow(this->low[i] - pt[i], 2);
            if (pt[i] > this->high[i]) dist += std::pow(pt[i] - this->high[i], 2);
        }
        return dist;
    }
};

template<typename F, typename Data = std::nullptr_t>
class KDTree
{
public:
    using node_ptr = typename std::shared_ptr<KDNode<F, Data>>;
    using rect_ptr = typename std::unique_ptr<Rectangle<F>>;
    using Point = typename std::vector<F>;

    using query_t = typename std::tuple<node_ptr, F>;
    using stack_t = typename std::vector<query_t>;

    size_t ndim;
    node_ptr root;
    rect_ptr rect;

    KDTree(size_t ndim) : ndim(ndim) {};

    template<typename Container, typename = std::enable_if_t<std::is_same_v<typename Container::value_type, F>>>
    KDTree(const std::vector<Container> & points, size_t ndim) : ndim(ndim)
    {
        std::vector<node_ptr> nodes;
        std::transform(points.begin(), points.end(), std::back_inserter(nodes),
                       [](Container c){return std::make_shared<KDNode<F, Data>>(c, 0);});
        this->root = build_tree(nodes.begin(), nodes.end(), 0);
        this->rect = std::make_unique<Rectangle<F>>(this->root, this->ndim);
    }

    template<typename Container, typename = std::enable_if_t<std::is_same_v<typename Container::value_type, F>>>
    KDTree(const std::vector<Container> & points, const std::vector<Data> & data, size_t ndim) : ndim(ndim)
    {
        std::vector<node_ptr> nodes;
        std::transform(points.begin(), points.end(), data.begin(), std::back_inserter(nodes),
                       [](Container c, Data d){return std::make_shared<KDNode<F, Data>>(c, 0, d);});
        this->root = build_tree(nodes.begin(), nodes.end(), 0);
        this->rect = std::make_unique<Rectangle<F>>(this->root, this->ndim);
    }

    KDTree & insert(Point pt, Data data = nullptr)
    {
        this->root = insert_node(this->root, pt, data, this->root->cut_dim);
        if (!this->rect) this->rect = std::make_unique<Rectangle<F>>(this->root, this->ndim);
        this->rect->update(pt);
        return *this;
    }

    KDTree & remove(const Point & pt)
    {
        this->root = remove_node(this->root, pt);
        if (this->rect)
        {
            if (this->root)
            {
                for (size_t i = 0; i < this->ndim; i++)
                {
                    if (pt[i] == this->rect->low[i]) this->rect->low[i] = find_min(this->root, i)->point[i];
                    if (pt[i] == this->rect->high[i]) this->rect->high[i] = find_max(this->root, i)->point[i];
                }
            }
            else this->rect = nullptr;
        }
        return *this;
    }

    query_t find_nearest(const Point & pt) const
    {
        return nearest_node(this->root, pt, this->rect, std::make_tuple(this->root, std::numeric_limits<F>::max()));
    }

    stack_t find_range(const Point & pt, F range) const
    {
        return find_range_node(this->root, pt, range * range, this->rect, stack_t());
    }

    friend std::ostream & operator<<(std::ostream & os, const KDTree<F, Data> & tree)
    {
        tree.print_rect(os);
        tree.print_node(os, tree.root);
        return os;
    }

protected:
    template <
        class NodeIter,
        typename = std::enable_if_t<is_node_pointer_v<typename NodeIter::value_type, F, Data>>
    >
    typename NodeIter::value_type build_tree(NodeIter first, NodeIter last, int dir) const
    {
        node_ptr node;

        if (last <= first) node = nullptr;
        else if (last == std::next(first))
        {
            node = *first;
            node->cut_dim = dir;
        }
        else
        {
            auto iter = wirthmedian(first, last, [dir](node_ptr a, node_ptr b){return a->point[dir] < b->point[dir];});

            node = *iter;
            node->cut_dim = dir;
            node->left = build_tree(first, iter, (dir + 1) % this->ndim);
            node->right = build_tree(std::next(iter), last, (dir + 1) % this->ndim);
        }

        return node;
    }

    node_ptr insert_node(node_ptr node, Point pt, Data data, int dir)
    {
        // Create new node if empty
        if (!node) return std::make_shared<KDNode<F, Data>>(pt, dir, data);
        
        if (pt[node->cut_dim] == node->point[node->cut_dim])
            throw std::runtime_error("Inserting a duplicate point");
        
        if (node->is_left(pt))
        {
            // left of splitting line
            node->left = insert_node(node->left, pt, data, (node->cut_dim + 1) % this->ndim);
        }
        else
        {
            // on or right of splitting line
            node->right = insert_node(node->right, pt, data, (node->cut_dim + 1) % this->ndim);
        }
        return node;
    }

    node_ptr remove_node(node_ptr node, const Point & pt)
    {
        if (!node) return nullptr;

        // Found the node
        if (std::equal(node->point.begin(), node->point.end(), pt.begin()))
        {
            // Take replacement from right
            if (node->right)
            {
                // Swapping the node
                auto new_node = find_min(node->right, node->cut_dim);
                node->point = new_node->point; node->data = new_node->data;

                node->right = remove_node(node->right, node->point);
            }
            // Take replacement from left
            else if (node->left)
            {
                // Swapping the nodes
                auto new_node = find_min(node->left, node->cut_dim);
                node->point = new_node->point;

                // move left subtree to right!
                node->right = remove_node(node->left, node->point);
                // left subtree is now empty
                node->left = nullptr;
            }
            // Remove this leaf
            else node = nullptr;
        }
        // Search left subtree
        else if (node->is_left(pt))
        {
            node->left = remove_node(node->left, pt);
        }
        // Search right subtree
        else node->right = remove_node(node->right, pt);

        return node;
    }

    query_t nearest_node(node_ptr node, const Point & pt, const rect_ptr & rect, query_t query) const
    {
        // Fell out of tree
        if (!node) return query;
        // This cell is too far away
        if (rect->distance(pt) >= std::pow(std::get<1>(query), 2)) return query;

        // Update if the root is closer
        F dist = detail::distance(node->point, pt);
        if (dist < std::pow(std::get<1>(query), 2)) query = std::make_tuple(node, std::sqrt(dist));

        // pt is close to left child
        rect_ptr slice = std::make_unique<Rectangle<F>>(rect->low, rect->high);
        if (node->is_left(pt))
        {
            slice->high[node->cut_dim] = node->point[node->cut_dim];
            query = nearest_node(node->left, pt, slice, query);

            slice->high[node->cut_dim] = rect->high[node->cut_dim];
            slice->low[node->cut_dim] = node->point[node->cut_dim];
            query = nearest_node(node->right, pt, slice, query);
        }
        // pt is closer to right child
        else
        {
            slice->low[node->cut_dim] = node->point[node->cut_dim];
            query = nearest_node(node->right, pt, slice, query);

            slice->low[node->cut_dim] = rect->low[node->cut_dim];
            slice->high[node->cut_dim] = node->point[node->cut_dim];
            query = nearest_node(node->left, pt, slice, query);
        }
        return query;
    }

    stack_t stack_push_node(node_ptr node, const Point & pt, stack_t stack) const
    {
        if (node->left) stack = stack_push_node(node->left, pt, stack);
        stack.emplace_back(node, std::sqrt(detail::distance(node->point, pt)));
        if (node->right) stack = stack_push_node(node->right, pt, stack);
        return stack;
    }

    stack_t find_range_node(node_ptr node, const Point & pt, F range, const rect_ptr & rect, stack_t stack) const
    {
        // Fell out of tree
        if (!node) return stack;
        // The cell doesn't overlap the query
        if (rect->distance(pt) > range) return stack;

        // The query contains the cell
        if (detail::distance(pt, rect->low) < range && detail::distance(pt, rect->high) < range)
        {
            return stack_push_node(node, pt, stack);
        }

        F dist = detail::distance(pt, node->point);
        if (dist < range) stack.emplace_back(node, std::sqrt(dist));

        // Search left subtree
        rect_ptr slice = std::make_unique<Rectangle<F>>(rect->low, rect->high);
        slice->high[node->cut_dim] = node->point[node->cut_dim];
        stack = find_range_node(node->left, pt, range, slice, stack);

        // Search right subtree
        slice->high[node->cut_dim] = rect->high[node->cut_dim];
        slice->low[node->cut_dim] = node->point[node->cut_dim];
        stack = find_range_node(node->right, pt, range, slice, stack);

        return stack;
    }

    std::ostream & print_rect(std::ostream & os) const
    {
        os << "low  : [";
        std::copy(this->rect->low.begin(), this->rect->low.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;

        os << "high : [";
        std::copy(this->rect->high.begin(), this->rect->high.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;
        return os;
    }

    std::ostream & print_node(std::ostream & os, node_ptr node, size_t level = 0) const
    {
        if (!node) return os;

        print_node(os, node->left, level + 1);

        os << std::string(level, '\t') << "(";
        std::copy(node->point.begin(), node->point.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << ")" << " axis = " << node->cut_dim << std::endl;

        print_node(os, node->right, level + 1);
        return os;
    }
};

}

#endif