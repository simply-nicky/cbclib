#ifndef KD_TREE_
#define KD_TREE_
#include "array.hpp"

namespace cbclib {

template <typename Point, typename Data>
class KDTree;

template <typename Point, typename Data>
class KDNode
{
public:
    using data_type = Data;
    using item_type = std::pair<Point, Data>;

    item_type item;
    int cut_dim;

    KDNode() = default;

    auto ndim() const -> decltype(std::declval<Point &>().size()) {return item.first.size();}

    Point & point() {return item.first;}
    const Point & point() const {return item.first;}

    Data & data() {return item.second;}
    const Data & data() const {return item.second;}

    bool is_left(const Point & pt) const
    {
        return pt[this->cut_dim] < point()[this->cut_dim];
    }

private:
    KDNode *left;
    KDNode *right;
    KDNode *parent;

    friend class KDTree<Point, Data>;

    template <typename Item, typename = std::enable_if_t<std::is_same_v<item_type, std::remove_cvref_t<Item>>>>
    KDNode(Item && item, int dir, KDNode * lt = nullptr, KDNode * rt = nullptr, KDNode * par = nullptr) :
        item(std::forward<Item>(item)), cut_dim(dir), left(lt), right(rt), parent(par) {}
};

template <typename Node, typename Point, typename Data, typename = void>
struct is_node : std::false_type {};

template <typename Node, typename Point, typename Data>
struct is_node<Node, Point, Data,
    typename std::enable_if_t<std::is_base_of_v<KDNode<Point, Data>, std::remove_cvref_t<Node>>>
> : std::true_type {};

template <typename Node, typename F, typename Data>
constexpr bool is_node_v = is_node<Node, F, Data>::value;

template<typename Point, typename Data>
class KDTree
{
public:
    using F = typename Point::value_type;
    using node_t = KDNode<Point, Data>;
    using item_type = typename node_t::item_type;

    class KDIterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = KDNode<Point, Data>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        KDIterator() : ptr(nullptr), root(nullptr) {}

        bool operator==(const KDIterator & rhs) const
        {
            return root == rhs.root && ptr == rhs.ptr;
        }

        bool operator!=(const KDIterator & rhs) const {return !operator==(rhs);}

        KDIterator & operator++()
        {
            if (ptr == nullptr)
            {
                // ++ from end(). Get the root of the tree
                ptr = root;

                // error! ++ requested for an empty tree
                while (ptr != nullptr && ptr->left != nullptr) ptr = ptr->left;

            }
            else if (ptr->right != nullptr)
            {
                // successor is the farthest left node of right subtree
                ptr = ptr->right;

                while (ptr->left != nullptr) ptr = ptr->left;
            }
            else
            {
                // have already processed the left subtree, and
                // there is no right subtree. move up the tree,
                // looking for a parent for which nodePtr is a left child,
                // stopping if the parent becomes NULL. a non-NULL parent
                // is the successor. if parent is NULL, the original node
                // was the last node inorder, and its successor
                // is the end of the list
                node_t * p = ptr->parent;
                while (p != nullptr && ptr == p->right)
                {
                    ptr = p; p = p->parent;
                }

                // if we were previously at the right-most node in
                // the tree, nodePtr = nullptr, and the iterator specifies
                // the end of the list
                ptr = p;
            }

            return *this;
        }

        KDIterator operator++(int)
        {
            auto saved = *this;
            operator++();
            return saved;
        }

        KDIterator & operator--()
        {
            if (ptr == nullptr)
            {
                // -- from end(). Get the root of the tree
                ptr = root;

                // move to the largest value in the tree,
                // which is the last node inorder
                while (ptr != nullptr && ptr->right != nullptr) ptr = ptr->right;
            }
            else if (ptr->left != nullptr)
            {
                // must have gotten here by processing all the nodes
                // on the left branch. predecessor is the farthest
                // right node of the left subtree
                ptr = ptr->left;

                while (ptr->right != nullptr) ptr = ptr->right;
            }
            else
            {
                // must have gotten here by going right and then
                // far left. move up the tree, looking for a parent
                // for which ptr is a right child, stopping if the
                // parent becomes nullptr. a non-nullptr parent is the
                // predecessor. if parent is nullptr, the original node
                // was the first node inorder, and its predecessor
                // is the end of the list
                node_t * p = ptr->parent;
                while (p != nullptr && ptr == p->left)
                {
                    ptr = p; p = p->parent;
                }

                // if we were previously at the left-most node in
                // the tree, ptr = NULL, and the iterator specifies
                // the end of the list
                ptr = p;
            }

            return *this;
        }

        KDIterator operator--(int)
        {
            auto saved = *this;
            operator--();
            return saved;
        }

        reference operator*() const {return *ptr;}
        pointer operator->() const {return ptr;}

    private:
        friend class KDTree<Point, Data>;

        const node_t * ptr;
        const node_t * root;

        KDIterator(const node_t * ptr, const node_t * root) : ptr(ptr), root(root) {}
    };

    class Rectangle
    {
    public:
        std::vector<F> low, high;

        Rectangle() = default;

        void update(const Point & pt)
        {
            for (size_t i = 0; i < pt.size(); i++)
            {
                low[i] = std::min(low[i], pt[i]);
                high[i] = std::max(high[i], pt[i]);
            }
        }

        F distance(const Point & pt) const
        {
            F dist = F();
            for (size_t i = 0; i < pt.size(); i++)
            {
                if (pt[i] < low[i]) dist += std::pow(low[i] - pt[i], 2);
                if (pt[i] > high[i]) dist += std::pow(pt[i] - high[i], 2);
            }
            return dist;
        }

    private:
        friend class KDTree<Point, Data>;

        Rectangle trim_left(node_t * node) const
        {
            Rectangle rect = *this;
            rect.high[node->cut_dim] = node->point()[node->cut_dim];
            return rect;
        }

        Rectangle trim_right(node_t * node) const
        {
            Rectangle rect = *this;
            rect.low[node->cut_dim] = node->point()[node->cut_dim];
            return rect;
        }

        Rectangle(const KDTree<Point, Data> & tree)
        {
            for (size_t i = 0; i < tree.ndim(); i++)
            {
                low.push_back(tree.find_min(i)->point()[i]);
                high.push_back(tree.find_max(i)->point()[i]);
            }
        }
    };

    using const_iterator = KDIterator;
    using iterator = const_iterator;

    using rect_t = Rectangle;

    using query_t = std::pair<const_iterator, F>;
    using stack_t = std::vector<std::pair<const_iterator, F>>;

    KDTree(std::vector<item_type> && items)
    {
        root = build_tree(std::make_move_iterator(items.begin()),
                          std::make_move_iterator(items.end()), nullptr, 0);
        if (root) rect = new rect_t{*this};
    }
    
    ~KDTree() {clear();}

    auto ndim() const -> decltype(std::declval<Point &>().size())
    {
        if (root == nullptr) return 0;
        else return root->ndim();
    }

    bool empty() const {return root == nullptr;}

    void clear()
    {
        root = clear_node(root);
        clear_rect();
    }

    const_iterator begin() const
    {
        return {begin_node(root), root};
    }

    const_iterator end() const
    {
        return {nullptr, root};
    }

    const_iterator insert(item_type && item)
    {
        const_iterator inserted;
        std::tie(root, inserted) = insert_node(root, std::move(item), root, root->cut_dim);

        if (inserted != end())
        {
            if (!rect) rect = new rect_t(*this);
            else rect->update(item.first);
        }

        return inserted;
    }

    size_t erase(const Point & pt)
    {
        size_t removed;
        std::tie(root, removed) = remove_node(root, pt);

        if (rect && removed)
        {
            if (root)
            {
                for (size_t i = 0; i < ndim(); i++)
                {
                    if (pt[i] == rect->low[i]) rect->low[i] = find_min(i)->point()[i];
                    if (pt[i] == rect->high[i]) rect->high[i] = find_max(i)->point()[i];
                }
            }
            else clear_rect();
        }

        return removed;
    }

    const_iterator erase(const_iterator pos)
    {
        if (pos != end())
        {
            erase((pos++)->point());
        }
        return pos;
    }

    const_iterator find(const Point & pt) const
    {
        return {find_node(root, pt, *rect, nullptr), root};
    }

    const_iterator find_min(int axis) const
    {
        return {find_min_node(root, axis), root};
    }

    const_iterator find_max(int axis) const
    {
        return {find_max_node(root, axis), root};
    }

    query_t find_nearest(const Point & pt) const
    {
        return nearest_node(root, pt, *rect, {const_iterator(root, root), std::numeric_limits<F>::max()});
    }

    stack_t find_range(const Point & pt, F range) const
    {
        return find_range_node(root, pt, range * range, *rect, {});
    }

    void print() const
    {
        print_node(std::cout, root);
        print_rect(std::cout);
    }

private:
    node_t * root;
    rect_t * rect;

    template <class Iter>
    node_t * build_tree(Iter first, Iter last, node_t * par, int dir)
    {
        using value_t = typename Iter::value_type;

        if (last <= first) return nullptr;
        else if (last == std::next(first))
        {
            return new node_t{*first, dir, nullptr, nullptr, par};
        }
        else
        {
            auto compare = [dir](const value_t & a, const value_t & b){return a.first[dir] < b.first[dir];};
            auto iter = wirthmedian(first, last, compare);

            node_t * node = new node_t{*iter, dir, nullptr, nullptr, par};
            node->left = build_tree(first, iter, node, (dir + 1) % node->ndim());
            node->right = build_tree(std::next(iter), last, node, (dir + 1) % node->ndim());
            return node;
        }
    }

    void clear_rect()
    {
        if (rect != nullptr)
        {
            delete rect; rect = nullptr;
        }
    }

    node_t * clear_node(node_t * node)
    {
        if (node != nullptr)
        {
            node->left = clear_node(node->left);
            node->right = clear_node(node->right);

            delete node; node = nullptr;
        }
        
        return node;
    }

    std::tuple<node_t *, const_iterator> insert_node(node_t * node, item_type && item, node_t * par, int dir)
    {
        // Create new node if empty
        if (!node)
        {
            node = new node_t{std::move(item), dir, nullptr, nullptr, par};
            return {node, const_iterator(node, root)};
        }

        // Duplicate data point, no insertion 
        if (item.first == node->point())
        {
            return {node, end()};
        }
        
        const_iterator inserted;

        if (node->is_left(item.first))
        {
            // left of splitting line
            std::tie(node->left, inserted) = insert_node(node->left, std::move(item), node, (node->cut_dim + 1) % node->ndim());
        }
        else
        {
            // on or right of splitting line
            std::tie(node->right, inserted) = insert_node(node->right, std::move(item), node, (node->cut_dim + 1) % node->ndim());
        }

        return {node, inserted};
    }

    std::tuple<node_t *, size_t> remove_node(node_t * node, const Point & pt)
    {
        // Fell out of tree
        if (!node) return {node, 0};

        size_t removed;

        // Found the node
        if (node->point() == pt)
        {
            // Take replacement from right
            if (node->right)
            {
                // Swapping the node
                node->item = find_min_node(node->right, node->cut_dim)->item;

                std::tie(node->right, removed) = remove_node(node->right, node->point());
            }
            // Take replacement from left
            else if (node->left)
            {
                // Swapping the nodes
                node->item = find_min_node(node->left, node->cut_dim)->item;

                // move left subtree to right!
                std::tie(node->right, removed) = remove_node(node->left, node->point());
                // left subtree is now empty
                node->left = nullptr;
            }
            // Remove this leaf
            else
            {
                node = clear_node(node); removed = 1;
            }
        }
        // Search left subtree
        else if (node->is_left(pt))
        {
            std::tie(node->left, removed) = remove_node(node->left, pt);
        }
        // Search right subtree
        else std::tie(node->right, removed) = remove_node(node->right, pt);

        return {node, removed};
    }

    node_t * min_node(node_t * a, node_t * b, node_t * c, int axis) const
    {
        if (b && b->point()[axis] < a->point()[axis])
        {
            if (c && c->point()[axis] < b->point()[axis]) return c;
            return b;
        }
        if (c && c->point()[axis] < a->point()[axis]) return c;
        return a;
    }

    node_t * max_node(node_t * a, node_t * b, node_t * c, int axis) const
    {
        if (b && b->point()[axis] > a->point()[axis])
        {
            if (c && c->point()[axis] > b->point()[axis]) return c;
            return b;
        }
        if (c && c->point()[axis] > a->point()[axis]) return c;
        return a;
    }

    node_t * find_min_node(node_t * node, int axis) const
    {
        // Fell out of tree
        if (!node) return node;

        if (node->cut_dim == axis)
        {
            if (!node->left) return node;
            else return find_min_node(node->left, axis);
        }
        else return min_node(node, find_min_node(node->left, axis), find_min_node(node->right, axis), axis);
    }

    node_t * find_max_node(node_t * node, int axis) const
    {
        // Fell out of tree
        if (!node) return node;

        if (node->cut_dim == axis)
        {
            if (!node->right) return node;
            else return find_max_node(node->right, axis);
        }
        else return max_node(node, find_max_node(node->left, axis), find_max_node(node->right, axis), axis);
    }

    node_t * begin_node(node_t * node) const
    {
        if (node == nullptr) return nullptr;
        if (node->left == nullptr) return node;
        return begin_node(node->left);
    }

    template <typename Point1, typename Point2>
    F distance(const Point1 & a, const Point2 & b) const
    {
        F dist = F();
        for (size_t i = 0; i < a.size(); i++) dist += std::pow(a[i] - b[i], 2);
        return dist;
    }

    node_t * find_node(node_t * node, const Point & pt, const rect_t & rect, node_t * query) const
    {
        // Fell out of tree
        if (!node) return query;
        // This cell is too far away
        if (rect.distance(pt) > F()) return query;

        // We found the node
        if (pt == node->point()) query = node;

        // pt is close to left child
        if (node->is_left(pt))
        {
            // First left then right
            query = find_node(node->left, pt, rect.trim_left(node), query);
            query = find_node(node->right, pt, rect.trim_right(node), query);
        }
        // pt is closer to right child
        else
        {
            // First right then left
            query = find_node(node->right, pt, rect.trim_right(node), query);
            query = find_node(node->left, pt, rect.trim_left(node), query);
        }

        return query;
    }

    query_t nearest_node(node_t * node, const Point & pt, const rect_t & rect, query_t && query) const
    {
        // Fell out of tree
        if (!node) return query;
        // This cell is too far away
        if (rect.distance(pt) >= std::pow(query.second, 2)) return query;

        // Update if the root is closer
        F dist = distance(node->point(), pt);
        if (dist < std::pow(query.second, 2)) query = std::make_pair(const_iterator(node, root), std::sqrt(dist));

        // pt is close to left child
        if (node->is_left(pt))
        {
            // First left then right
            query = nearest_node(node->left, pt, rect.trim_left(node), std::move(query));
            query = nearest_node(node->right, pt, rect.trim_right(node), std::move(query));
        }
        // pt is closer to right child
        else
        {
            // First right then left
            query = nearest_node(node->right, pt, rect.trim_right(node), std::move(query));
            query = nearest_node(node->left, pt, rect.trim_left(node), std::move(query));
        }

        return query;
    }

    stack_t stack_push_node(node_t * node, const Point & pt, stack_t && stack) const
    {
        if (node->left) stack = stack_push_node(node->left, pt, std::move(stack));
        stack.emplace_back(const_iterator(node, root), std::sqrt(distance(node->point(), pt)));
        if (node->right) stack = stack_push_node(node->right, pt, std::move(stack));
        return stack;
    }

    stack_t find_range_node(node_t * node, const Point & pt, F range_sq, const rect_t & rect, stack_t && stack) const
    {
        // Fell out of tree
        if (!node) return stack;
        // The cell doesn't overlap the query
        if (rect.distance(pt) > range_sq) return stack;

        // The query contains the cell
        if (distance(pt, rect.low) < range_sq && distance(pt, rect.high) < range_sq)
        {
            return stack_push_node(node, pt, std::move(stack));
        }

        F dist = distance(pt, node->point());
        // Put this item to stack
        if (dist < range_sq) stack.emplace_back(const_iterator(node, root), std::sqrt(dist));

        // Search left subtree
        stack = find_range_node(node->left, pt, range_sq, rect.trim_left(node), std::move(stack));

        // Search right subtree
        stack = find_range_node(node->right, pt, range_sq, rect.trim_right(node), std::move(stack));

        return stack;
    }

    std::ostream & print_rect(std::ostream & os) const
    {
        os << "low  : [";
        std::copy(rect->low.begin(), rect->low.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;

        os << "high : [";
        std::copy(rect->high.begin(), rect->high.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;
        return os;
    }

    std::ostream & print_node(std::ostream & os, node_t * node, size_t level = 0) const
    {
        if (!node) return os;

        print_node(os, node->left, level + 1);

        os << std::string(level, '\t') << "(";
        std::copy(node->point().begin(), node->point().end(), std::experimental::make_ostream_joiner(os, ", "));
        os << ")" << " axis = " << node->cut_dim << std::endl;

        print_node(os, node->right, level + 1);
        return os;
    }
};

template<typename F, size_t ndim>
void test_tree(size_t npts, F range);

}

#endif