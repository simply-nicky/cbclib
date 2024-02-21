#include "kd_tree.hpp"

namespace cbclib {

template<typename F, size_t ndim>
void test_tree(size_t npts, F range)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::uniform_real_distribution<F> dist {0.0, 100.0};
    std::uniform_int_distribution<int> dist_int {0, 100};

    auto gen = [&dist, &mersenne_engine](){return dist(mersenne_engine);};
    auto gen_int = [&dist_int, &mersenne_engine](){return dist_int(mersenne_engine);};
    auto gen_array = [&gen, &gen_int]()
    {
        std::array<F, ndim> arr;
        std::generate(arr.begin(), arr.end(), gen);
        return std::make_pair(arr, gen_int());
    };

    std::vector<std::pair<std::array<F, ndim>, int>> items;
    std::generate_n(std::back_inserter(items), npts, gen_array);

    std::cout << "Points generated:\n";
    for (auto item : items)
    {
        std::cout << "{";
        std::copy(item.first.begin(), item.first.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
        std::cout << "} ";
    }
    std::cout << std::endl;

    auto tree = KDTree<std::array<F, ndim>, int>(std::move(items));

    std::cout << "Points in the tree:\n";
    for (auto node : tree)
    {
        std::cout << "{";
        std::copy(node.point().begin(), node.point().end(), std::experimental::make_ostream_joiner(std::cout, ", "));
        std::cout << "} ";
    }
    std::cout << std::endl;

    std::array<F, ndim> point;
    std::generate(point.begin(), point.end(), gen);
    std::cout << "Inserting {";
    std::copy(point.begin(), point.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
    std::cout << "}\n";

    auto inserted = tree.insert(std::make_pair(point, gen_int()));
    if (inserted != tree.end())
    {
        std::cout << "{";
        std::copy(inserted->point().begin(), inserted->point().end(), std::experimental::make_ostream_joiner(std::cout, ", "));
        std::cout << "} inserted\n";
    }
    inserted = tree.insert(std::make_pair(point, gen_int()));
    std::cout << ((inserted == tree.end()) ? "inserted == end() " : "inserted != end() ") << "after inserting twice\n";

    std::cout << "Tree:\n";
    tree.print();

    auto found = tree.find(point);
    if (found != tree.end())
    {
        std::cout << "{";
        std::copy(found->point().begin(), found->point().end(), std::experimental::make_ostream_joiner(std::cout, ", "));
        std::cout << "} found\n";
    
        auto removed = tree.erase(found);
        std::cout << "erase returned {";
        std::copy(removed->point().begin(), removed->point().end(), std::experimental::make_ostream_joiner(std::cout, ", "));
        std::cout << "}\n";
    }

    std::cout << "Tree:\n";
    tree.print();

    point.fill(50.0);
    auto stack = tree.find_range(point, range);

    std::cout << "Found points in vicinity:\n";
    for (auto query : stack)
    {
        std::cout << "(";
        std::copy(query.first->point().begin(), query.first->point().end(), std::experimental::make_ostream_joiner(std::cout, ", "));
        std::cout << "), dist = " << query.second << std::endl;
    }

    tree.clear();

    KDTree<std::array<F, ndim>, int> new_tree;
    new_tree.insert(std::make_pair(point, 0));

    std::cout << "New tree:\n";
    for (auto node : new_tree)
    {
        std::cout << "{";
        std::copy(node.point().begin(), node.point().end(), std::experimental::make_ostream_joiner(std::cout, ", "));
        std::cout << "} ";
    }

    tree = new_tree;
}

}

PYBIND11_MODULE(kd_tree, m)
{
    using namespace cbclib;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("test_tree", &test_tree<double, 2>, py::arg("npts"), py::arg("range"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}