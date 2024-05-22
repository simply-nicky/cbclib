#include "geometry.hpp"
#include "zip.hpp"

namespace cbclib {

template <typename T>
using py_array_t = py::array_t<T, py::array::c_style>;

template <typename T>
auto rmat_to_euler(const std::array<T, 9> & r)
{
    T alpha, beta, gamma;

    beta = std::acos(r[8]);
    if (isclose(beta, T()))
    {
        alpha = std::atan2(-r[3], r[0]);
        gamma = T();
    }
    else if (isclose(beta, T(M_PI)))
    {
        alpha = std::atan2(r[3], r[0]);
        gamma = T();
    }
    else
    {
        alpha = std::atan2(r[6], -r[7]);
        gamma = std::atan2(r[2], r[5]);
    }
    if (alpha < T()) alpha += 2 * M_PI;
    if (gamma < T()) gamma += 2 * M_PI;

    return std::make_tuple(alpha, beta, gamma);
}

template <typename T>
auto rmat_to_euler_vjp(const std::array<T, 3> & ct_a, const std::array<T, 9> & r)
{
    std::array<T, 9> ct_r {};

    T beta = std::acos(r[8]);
    ct_r[8] = -ct_a[1] / std::sqrt(1 - r[8] * r[8]);
    if (isclose(beta, T()))
    {
        ct_r[3] = -ct_a[0] * r[0] / (r[3] * r[3] + r[0] * r[0]);
        ct_r[0] = ct_a[0] * r[3] / (r[3] * r[3] + r[0] * r[0]);
    }
    else if (isclose(beta, T(M_PI)))
    {
        ct_r[3] = ct_a[0] * r[0] / (r[3] * r[3] + r[0] * r[0]);
        ct_r[0] = -ct_a[0] * r[3] / (r[3] * r[3] + r[0] * r[0]);
    }
    else
    {
        ct_r[6] = -ct_a[0] * r[7] / (r[6] * r[6] + r[7] * r[7]);
        ct_r[7] = ct_a[0] * r[6] / (r[6] * r[6] + r[7] * r[7]);
        ct_r[2] = ct_a[2] * r[5] / (r[2] * r[2] + r[5] * r[5]);
        ct_r[5] = -ct_a[2] * r[2] / (r[2] * r[2] + r[5] * r[5]);
    }

    return to_tuple(ct_r);
}

template <typename F>
py::array_t<F> euler_angles(py_array_t<F> rmats, unsigned threads)
{
    assert(PyArray_API);

    auto rbuf = rmats.request();
    check_dimensions("rmats", rbuf.ndim - 2, rbuf.shape, 3, 3);
    auto rsize = rbuf.size / 9;

    std::vector<py::ssize_t> ashape;
    std::copy_n(rbuf.shape.begin(), rbuf.ndim - 2, std::back_inserter(ashape));
    ashape.push_back(3);

    auto angles = py::array_t<F>(ashape);

    auto rarr = array<F>(rbuf);
    auto aarr = array<F>(angles.request());

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < rsize; i++)
    {
        to_tie<3>(aarr, 3 * i) = rmat_to_euler(to_array<9>(rarr, 9 * i));
    }

    py::gil_scoped_acquire acquire;

    return angles;
}

template <typename F>
py::array_t<F> euler_angles_vjp(py_array_t<F> cotangent, py_array_t<F> rmats, unsigned threads)
{
    assert(PyArray_API);

    auto rbuf = rmats.request();
    auto ctbuf = cotangent.request();
    check_dimensions("rmats", rbuf.ndim - 2, rbuf.shape, 3, 3);
    check_equal("cotangent and rmats have incompatible shapes",
                rbuf.shape.begin(), std::next(rbuf.shape.begin(), rbuf.ndim - 2),
                ctbuf.shape.begin(), std::next(ctbuf.shape.begin(), ctbuf.ndim - 1));
    auto rsize = rbuf.size / 9;

    auto out = py::array_t<F>(rbuf.shape);

    auto rarr = array<F>(rbuf);
    auto ctarr = array<F>(ctbuf);
    auto oarr = array<F>(out.request());

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < rsize; i++)
    {
        to_tie<9>(oarr, 9 * i) = rmat_to_euler_vjp(to_array<3>(ctarr, 3 * i),
                                                   to_array<9>(rarr, 9 * i));
    }

    py::gil_scoped_acquire acquire;

    return out;
}

template <typename T>
auto euler_to_rmat(const std::array<T, 3> & a)
{
    std::array<T, 3> c {std::cos(a[0]), std::cos(a[1]), std::cos(a[2])};
    std::array<T, 3> s {std::sin(a[0]), std::sin(a[1]), std::sin(a[2])};
    return std::make_tuple( c[0] * c[2] - s[0] * s[2] * c[1],  s[0] * c[2] + c[0] * s[2] * c[1], s[2] * s[1],
                           -c[0] * s[2] - s[0] * c[2] * c[1], -s[0] * s[2] + c[0] * c[2] * c[1], c[2] * s[1],
                            s[0] * s[1],                      -c[0] * s[1],                      c[1]       );
}

template <typename T>
auto euler_to_rmat_vjp(const std::array<T, 9> & ct, const std::array<T, 3> & a)
{
    std::array<T, 3> c {std::cos(a[0]), std::cos(a[1]), std::cos(a[2])};
    std::array<T, 3> s {std::sin(a[0]), std::sin(a[1]), std::sin(a[2])};
    return std::make_tuple(-s[0] * ( ct[0] * c[2]        + ct[1] * s[2] * c[1] - ct[3] * s[2]        + ct[4] * c[2] * c[1] - ct[7] * s[1]) +
                            c[0] * (-ct[0] * s[2] * c[1] + ct[1] * c[2]        - ct[3] * c[2] * c[1] - ct[4] * s[2]        + ct[6] * s[1]),
                           -s[1] * (-ct[0] * s[0] * s[2] + ct[1] * c[0] * s[2] - ct[3] * s[0] * c[2] + ct[4] * c[0] * c[2] + ct[8]       ) +
                            c[1] * ( ct[2] * s[2]        + ct[5] * c[2]        + ct[6] * s[0]        - ct[7] * c[0]                      ),
                           -s[2] * ( ct[0] * c[0]        + ct[1] * s[0]        - ct[3] * s[0] * c[1] + ct[4] * c[0] * c[1] + ct[5] * s[1]) +
                            c[2] * (-ct[0] * s[0] * c[1] + ct[1] * c[0] * c[1] + ct[2] * s[1]        - ct[3] * c[0]        - ct[4] * s[0]));
}

template <typename F>
py::array_t<F> euler_matrix(py_array_t<F> angles, unsigned threads)
{
    assert(PyArray_API);

    auto abuf = angles.request();
    check_dimensions("angles", abuf.ndim - 1, abuf.shape, 3);
    auto asize = abuf.size / 3;

    std::vector<py::ssize_t> rshape;
    std::copy_n(abuf.shape.begin(), abuf.ndim - 1, std::back_inserter(rshape));
    rshape.push_back(3); rshape.push_back(3);

    auto rmats = py::array_t<F>(rshape);

    auto aarr = array<F>(abuf);
    auto rarr = array<F>(rmats.request());

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < asize; i++)
    {
        to_tie<9>(rarr, 9 * i) = euler_to_rmat(to_array<3>(aarr, 3 * i));
    }

    py::gil_scoped_acquire acquire;

    return rmats;
}

template <typename F>
py::array_t<F> euler_matrix_vjp(py_array_t<F> cotangents, py_array_t<F> angles, unsigned threads)
{
    assert(PyArray_API);

    auto abuf = angles.request();
    auto ctbuf = cotangents.request();
    check_dimensions("angles", abuf.ndim - 1, abuf.shape, 3);
    check_equal("cotangents and angles have incompatible shapes",
                abuf.shape.begin(), std::next(abuf.shape.begin(), abuf.ndim - 1),
                ctbuf.shape.begin(), std::next(ctbuf.shape.begin(), ctbuf.ndim - 2));
    auto asize = abuf.size / 3;

    auto out = py::array_t<F>(abuf.shape);

    auto aarr = array<F>(abuf);
    auto ctarr = array<F>(ctbuf);
    auto oarr = array<F>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < asize; i++)
    {
        e.run([&]
        {
            to_tie<3>(oarr, 3 * i) = euler_to_rmat_vjp(to_array<9>(ctarr, 9 * i), to_array<3>(aarr, 3 * i));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T>
auto rmat_to_tilts(const std::array<T, 9> & r)
{
    std::array<T, 3> a {r[7] - r[5], r[2] - r[6], r[3] - r[1]};
    auto l = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];

    return std::make_tuple(std::acos((r[0] + r[4] + r[8] - 1) / 2), std::acos(a[2] / std::sqrt(l)), std::atan2(a[1], a[0]));
}

template <typename T>
auto rmat_to_tilts_vjp(const std::array<T, 3> & ct, const std::array<T, 9> & r)
{
    std::array<T, 3> a {r[7] - r[5], r[2] - r[6], r[3] - r[1]};
    auto l = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];

    return std::make_tuple(-ct[0] / std::sqrt(4 - std::pow(r[0] + r[4] + r[8] - 1, 2)),
                            ct[1] * std::sqrt(a[0] * a[0] + a[1] * a[1]) / l,
                            ct[1] * a[1] * a[2] / (std::sqrt(a[0] * a[0] + a[1] * a[1]) * l) + ct[2] * a[0] / (a[0] * a[0] + a[1] * a[1]),
                           -ct[1] * std::sqrt(a[0] * a[0] + a[1] * a[1]) / l,
                           -ct[0] / std::sqrt(4 - std::pow(r[0] + r[4] + r[8] - 1, 2)),
                           -ct[1] * a[0] * a[2] / (std::sqrt(a[0] * a[0] + a[1] * a[1]) * l) + ct[2] * a[1] / (a[0] * a[0] + a[1] * a[1]),
                           -ct[1] * a[1] * a[2] / (std::sqrt(a[0] * a[0] + a[1] * a[1]) * l) - ct[2] * a[0] / (a[0] * a[0] + a[1] * a[1]),
                            ct[1] * a[0] * a[2] / (std::sqrt(a[0] * a[0] + a[1] * a[1]) * l) - ct[2] * a[1] / (a[0] * a[0] + a[1] * a[1]),
                           -ct[0] / std::sqrt(4 - std::pow(r[0] + r[4] + r[8] - 1, 2)));
}

template <typename F>
py::array_t<F> tilt_angles(py_array_t<F> rmats, unsigned threads)
{
    assert(PyArray_API);

    auto rbuf = rmats.request();
    check_dimensions("rmats", rbuf.ndim - 2, rbuf.shape, 3, 3);
    auto rsize = rbuf.size / 9;

    std::vector<py::ssize_t> ashape;
    std::copy_n(rbuf.shape.begin(), rbuf.ndim - 2, std::back_inserter(ashape));
    ashape.push_back(3);

    auto angles = py::array_t<F>(ashape);

    auto aarr = array<F>(angles.request());
    auto rarr = array<F>(rbuf);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < rsize; i++)
    {
        e.run([&]
        {
            to_tie<3>(aarr, 3 * i) = rmat_to_tilts(to_array<9>(rarr, 9 * i));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return angles;
}

template <typename F>
py::array_t<F> tilt_angles_vjp(py_array_t<F> cotangents, py_array_t<F> rmats, unsigned threads)
{
    assert(PyArray_API);

    auto rbuf = rmats.request();
    auto ctbuf = cotangents.request();
    check_dimensions("rmats", rbuf.ndim - 2, rbuf.shape, 3, 3);
    check_equal("cotangents and rmats have incompatible shapes",
                rbuf.shape.begin(), std::next(rbuf.shape.begin(), rbuf.ndim - 2),
                ctbuf.shape.begin(), std::next(ctbuf.shape.begin(), ctbuf.ndim - 1));
    auto rsize = rbuf.size / 9;

    auto out = py::array_t<F>(rbuf.shape);

    auto rarr = array<F>(rbuf);
    auto ctarr = array<F>(ctbuf);
    auto oarr = array<F>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < rsize; i++)
    {
        e.run([&]
        {
            to_tie<9>(oarr, 9 * i) = rmat_to_tilts_vjp(to_array<3>(ctarr, 3 * i), to_array<9>(rarr, 9 * i));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T>
auto tilts_to_rmat(const std::array<T, 3> & ang)
{
    std::array<T, 4> v {std::cos(ang[0] / 2),
                        -std::sin(ang[0] / 2) * std::sin(ang[1]) * std::cos(ang[2]),
                        -std::sin(ang[0] / 2) * std::sin(ang[1]) * std::sin(ang[2]),
                        -std::sin(ang[0] / 2) * std::cos(ang[1])};

    return std::make_tuple(v[0] * v[0] + v[1] * v[1] - v[2] * v[2] - v[3] * v[3], 2 * (v[1] * v[2] + v[0] * v[3]), 2 * (v[1] * v[3] - v[0] * v[2]),
                           2 * (v[1] * v[2] - v[0] * v[3]), v[0] * v[0] + v[2] * v[2] - v[1] * v[1] - v[3] * v[3], 2 * (v[2] * v[3] + v[0] * v[1]),
                           2 * (v[1] * v[3] + v[0] * v[2]), 2 * (v[2] * v[3] - v[0] * v[1]), v[0] * v[0] + v[3] * v[3] - v[1] * v[1] - v[2] * v[2]);
}

template <typename T>
auto tilts_to_rmat_vjp(const std::array<T, 9> & ct, const std::array<T, 3> & ang)
{
    std::array<T, 3> c {std::cos(ang[0] / 2), std::cos(ang[1]), std::cos(ang[2])};
    std::array<T, 3> s {std::sin(ang[0] / 2), std::sin(ang[1]), std::sin(ang[2])};
    std::array<T, 4> v {std::cos(ang[0] / 2),
                        -std::sin(ang[0] / 2) * std::sin(ang[1]) * std::cos(ang[2]),
                        -std::sin(ang[0] / 2) * std::sin(ang[1]) * std::sin(ang[2]),
                        -std::sin(ang[0] / 2) * std::cos(ang[1])};

    return std::make_tuple(ct[0] * 2 * v[0] * (-s[1] * s[1] * s[2] * s[2] + s[1] * s[1] - 1) * s[0] +
                           ct[1] * (-2 * v[0] * v[1] * s[1] * s[2] - 2 * v[3] * s[0] - c[1]) +
                           ct[2] * (2 * s[0] * c[0] * c[1] * c[2] + s[2] * (c[0] * c[0] - s[0] * s[0])) * s[1] +
                           ct[3] * (-2 * v[0] * v[1] * s[1] * s[2] + 2 * v[3] * s[0] + c[1]) +
                           ct[4] * 2 * v[0] * (s[1] * s[1] * s[2] * s[2] - 1) * s[0] +
                           ct[5] * (2 * s[0] * c[0] * s[2] * c[1] - c[2] * (c[0] * c[0] - s[0] * s[0])) * s[1] +
                           ct[6] * (2 * s[0] * c[0] * c[1] * c[2] - s[2] * (c[0] * c[0] - s[0] * s[0])) * s[1] +
                           ct[7] * (2 * s[0] * c[0] * s[2] * c[1] + c[2] * (c[0] * c[0] - s[0] * s[0])) * s[1] +
                           ct[8] * v[0] * (c[1] * c[1] - s[1] * s[1] - 1) * s[0],
                           ct[0] * 4 * v[1] * v[3] * c[2] +
                           ct[1] * 2 * (v[0] - 2 * v[3] * s[2] * c[2]) * s[0] * s[1] +
                           ct[2] * 2 * (v[0] * s[2] * c[1] + 2 * v[1] * s[1] + s[0] * c[2]) * s[0] +
                           ct[3] * 2 * (-v[0] - 2 * v[3] * s[2] * c[2]) * s[0] * s[1] +
                           ct[4] * 4 * v[2] * v[3] * s[2] +
                           ct[5] * 2 * (-v[0] * c[1] * c[2] + 2 * v[2] * s[1] + s[0] * s[2]) * s[0] +
                           ct[6] * 2 * (-v[0] * s[2] * c[1] + 2 * v[1] * s[1] + s[0] * c[2]) * s[0] +
                           ct[7] * 2 * (v[0] * c[1] * c[2] + 2 * v[2] * s[1] + s[0] * s[2]) * s[0] +
                           ct[8] * 4 * v[3] * s[0] * s[1],
                           ct[0] * (-4 * v[1] * v[2]) +
                           ct[1] * 2 * s[0] * s[0] * s[1] * s[1] * (c[2] * c[2] - s[2] * s[2]) +
                           ct[2] * 2 * (v[0] * c[2] + v[3] * s[2]) * s[0] * s[1] +
                           ct[3] * 2 * s[0] * s[0] * s[1] * s[1] * (c[2] * c[2] - s[2] * s[2]) +
                           ct[4] * 4 * v[1] * v[2] +
                           ct[5] * 2 * (v[0] * s[2] - v[3] * c[2]) * s[0] * s[1] +
                           ct[6] * 2 * (-v[0] * c[2] + v[3] * s[2]) * s[0] * s[1] +
                           ct[7] * 2 * (-v[0] * s[2] - v[3] * c[2]) * s[0] * s[1]);
}

template <typename F>
py::array_t<F> tilt_matrix(py_array_t<F> angles, unsigned threads)
{
    assert(PyArray_API);

    auto abuf = angles.request();
    check_dimensions("angles", abuf.ndim - 1, abuf.shape, 3);
    auto asize = abuf.size / 3;

    std::vector<py::ssize_t> rshape;
    std::copy_n(abuf.shape.begin(), abuf.ndim - 1, std::back_inserter(rshape));
    rshape.push_back(3); rshape.push_back(3);

    auto rmats = py::array_t<F>(rshape);

    auto aarr = array<F>(abuf);
    auto rarr = array<F>(rmats.request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < asize; i++)
    {
        e.run([&]
        {
            to_tie<9>(rarr, 9 * i) = tilts_to_rmat(to_array<3>(aarr, 3 * i));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return rmats;
}

template <typename F>
py::array_t<F> tilt_matrix_vjp(py_array_t<F> cotangents, py_array_t<F> angles, unsigned threads)
{
    assert(PyArray_API);

    auto abuf = angles.request();
    auto ctbuf = cotangents.request();
    check_dimensions("angles", abuf.ndim - 1, abuf.shape, 3);
    check_equal("cotangents and angles have incompatible shapes",
                abuf.shape.begin(), std::next(abuf.shape.begin(), abuf.ndim - 1),
                ctbuf.shape.begin(), std::next(ctbuf.shape.begin(), ctbuf.ndim - 2));
    auto asize = abuf.size / 3;

    auto out = py::array_t<F>(abuf.shape);

    auto aarr = array<F>(abuf);
    auto ctarr = array<F>(ctbuf);
    auto oarr = array<F>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < asize; i++)
    {
        e.run([&]
        {
            to_tie<3>(oarr, 3 * i) = tilts_to_rmat_vjp(to_array<9>(ctarr, 9 * i), to_array<3>(aarr, 3 * i));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T>
auto det2k(T x, T y, const std::array<T, 3> & src)
{
    std::array<T, 3> v {x - src[0], y - src[1], -src[2]};
    auto d = amplitude(v);
    return std::make_tuple(v[0] / d, v[1] / d, v[2] / d);
}

template <typename T>
auto det2k_vjp(const std::array<T, 3> & ct, std::tuple<T &, T &, T &> ct_src, T x, T y, const std::array<T, 3> & src)
{
    std::array<T, 3> v {x - src[0], y - src[1], -src[2]};
    auto d = amplitude(v);
    auto divisor = d * d * d;

    std::get<0>(ct_src) += ct[0] * (-v[2] * v[2] - v[1] * v[1]) / divisor +
                           ct[1] * v[0] * v[1] / divisor +
                           ct[2] * v[2] * v[0] / divisor;
    std::get<1>(ct_src) += ct[0] * v[0] * v[1] / divisor +
                           ct[1] * (-v[2] * v[2] - v[0] * v[0]) / divisor +
                           ct[2] * v[2] * v[1] / divisor;
    std::get<2>(ct_src) += ct[0] * v[2] * v[0] / divisor +
                           ct[1] * v[2] * v[1] / divisor +
                           ct[2] * (-v[0] * v[0] - v[1] * v[1]) / divisor;

    return std::make_tuple(-ct[0] * (-v[2] * v[2] - v[1] * v[1]) / divisor +
                           -ct[1] * v[0] * v[1] / divisor +
                           -ct[2] * v[2] * v[0] / divisor,
                           -ct[0] * v[0] * v[1] / divisor +
                           -ct[1] * (-v[2] * v[2] - v[0] * v[0]) / divisor +
                           -ct[2] * v[2] * v[1] / divisor);
}

template <typename F, typename I>
py::array_t<F> det_to_k(py_array_t<F> x, py_array_t<F> y, py_array_t<F> src, std::optional<py_array_t<I>> idxs, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info xbuf = x.request(), ybuf = y.request(), sbuf = src.request();
    auto size = xbuf.size;

    check_dimensions("src", sbuf.ndim - 1, sbuf.shape, 3);
    auto ssize = sbuf.size / sbuf.shape[sbuf.ndim - 1];

    if (!idxs) fill_indices("idxs", ssize, size, idxs);
    else check_indices("idxs", ssize, size, idxs);

    py::buffer_info ibuf = idxs.value().request();
    if (size != ybuf.size || size != ibuf.size)
    {
        std::ostringstream oss1, oss2, oss3;
        std::copy(xbuf.shape.begin(), xbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ybuf.shape.begin(), ybuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss3, ", "));
        throw std::invalid_argument("x, y, and idxs have incompatible shapes: {" + oss1.str() +
                                    "}, {" + oss2.str() + "}, and {" + oss3.str() + "}");
    }

    std::vector<py::ssize_t> out_shape;
    std::copy(xbuf.shape.begin(), xbuf.shape.end(), std::back_inserter(out_shape));
    out_shape.push_back(3);

    auto out = py::array_t<F>(out_shape);

    auto oarr = array<F>(out.request());
    auto xarr = array<F>(xbuf);
    auto yarr = array<F>(ybuf);
    auto iarr = array<I>(ibuf);
    auto sarr = array<F>(sbuf);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < size; i++)
    {
        e.run([&]
        {
            to_tie<3>(oarr, 3 * i) = det2k(xarr[i], yarr[i], to_array<3>(sarr, 3 * iarr[i]));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename F, typename I>
auto det_to_k_vjp(py_array_t<F> cotangents, py_array_t<F> x, py_array_t<F> y, py_array_t<F> src, std::optional<py_array_t<I>> idxs, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info xbuf = x.request(), ybuf = y.request(), sbuf = src.request();
    auto size = xbuf.size;

    check_dimensions("src", sbuf.ndim - 1, sbuf.shape, 3);
    auto ssize = sbuf.size / sbuf.shape[sbuf.ndim - 1];

    if (!idxs) fill_indices("idxs", ssize, size, idxs);
    else check_indices("idxs", ssize, size, idxs);

    py::buffer_info ibuf = idxs.value().request();
    if (size != ybuf.size || size != ibuf.size)
    {
        std::ostringstream oss1, oss2, oss3;
        std::copy(xbuf.shape.begin(), xbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ybuf.shape.begin(), ybuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss3, ", "));
        throw std::invalid_argument("x, y, and idxs have incompatible shapes: {" + oss1.str() +
                                    "}, {" + oss2.str() + "}, and {" + oss3.str() + "}");
    }

    auto ct_x = py::array_t<F>(xbuf.shape);
    auto ct_y = py::array_t<F>(ybuf.shape);
    auto ct_s = py::array_t<F>(sbuf.shape);

    fill_array(ct_s, F());

    auto ct_xarr = array<F>(ct_x.request());
    auto ct_yarr = array<F>(ct_y.request());
    auto ct_sarr = array<F>(ct_s.request());

    auto ctarr = array<F>(cotangents.request());
    auto xarr = array<F>(xbuf);
    auto yarr = array<F>(ybuf);
    auto iarr = array<I>(ibuf);
    auto sarr = array<F>(sbuf);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<F> buffer (sarr.size, F());

        #pragma omp for nowait
        for (py::ssize_t i = 0; i < size; i++)
        {
            e.run([&]
            {
                std::tie(ct_xarr[i], ct_yarr[i]) = det2k_vjp(to_array<3>(ctarr, 3 * i), to_tie<3>(buffer, 3 * iarr[i]),
                                                             xarr[i], yarr[i], to_array<3>(sarr, 3 * iarr[i]));
            });
        }

        #pragma omp critical
        std::transform(buffer.begin(), buffer.end(), ct_sarr.begin(), ct_sarr.begin(), std::plus<F>());
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(ct_x, ct_y, ct_s);
}

template <typename T>
auto k2det(const std::array<T, 3> & k, const std::array<T, 3> & src)
{
    return std::make_tuple(src[0] - k[0] / k[2] * src[2], src[1] - k[1] / k[2] * src[2]);
}

template <typename T>
auto k2det_vjp(const std::array<T, 2> & ct, std::tuple<T &, T &, T &> ct_src, const std::array<T, 3> & k, const std::array<T, 3> & src)
{
    std::get<0>(ct_src) += ct[0];
    std::get<1>(ct_src) += ct[1];
    std::get<2>(ct_src) += -ct[0] * k[0] / k[2] - ct[1] * k[1] / k[2];

    return std::make_tuple(-ct[0] * src[2] / k[2], -ct[1] * src[2] / k[2],
                           ct[0] * src[2] * k[0] / (k[2] * k[2]) +
                           ct[1] * src[2] * k[1] / (k[2] * k[2]));
}

template <typename F, typename I>
std::tuple<py::array_t<F>, py::array_t<F>> k_to_det(py_array_t<F> k, py_array_t<F> src, std::optional<py_array_t<I>> idxs, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info kbuf = k.request(), sbuf = src.request();

    check_dimensions("k", kbuf.ndim - 1, kbuf.shape, 3);
    auto size = kbuf.size / kbuf.shape[kbuf.ndim - 1];

    check_dimensions("src", sbuf.ndim - 1, sbuf.shape, 3);
    auto ssize = sbuf.size / sbuf.shape[sbuf.ndim - 1];

    if (!idxs) fill_indices("idxs", ssize, size, idxs);
    else check_indices("idxs", ssize, size, idxs);

    py::buffer_info ibuf = idxs.value().request();
    if (size != ibuf.size)
    {
        std::ostringstream oss1, oss2;
        std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument("karr and idxs have incompatible shapes: {" + oss1.str() + "}, {" + oss2.str() + "}");
    }

    std::vector<py::ssize_t> out_shape;
    std::copy_n(kbuf.shape.begin(), kbuf.ndim - 1, std::back_inserter(out_shape));

    auto x = py::array_t<F>(out_shape);
    auto y = py::array_t<F>(out_shape);

    auto xarr = array<F>(x.request());
    auto yarr = array<F>(y.request());
    auto karr = array<F>(kbuf);
    auto iarr = array<I>(ibuf);
    auto sarr = array<F>(sbuf);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < size; i++)
    {
        e.run([&]
        {
            std::tie(xarr[i], yarr[i]) = k2det(to_array<3>(karr, 3 * i), to_array<3>(sarr, 3 * iarr[i]));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(x, y);
}

template <typename F, typename I>
std::tuple<py::array_t<F>, py::array_t<F>> k_to_det_vjp(std::tuple<py_array_t<F>, py_array_t<F>> cotangents, py_array_t<F> k, py_array_t<F> src, std::optional<py_array_t<I>> idxs, unsigned threads)
{
    assert(PyArray_API);

    auto [ct_x, ct_y] = cotangents;
    py::buffer_info kbuf = k.request(), sbuf = src.request(), ct_xbuf = ct_x.request(), ct_ybuf = ct_y.request();

    check_dimensions("karr", kbuf.ndim - 1, kbuf.shape, 3);
    auto size = kbuf.size / kbuf.shape[kbuf.ndim - 1];

    check_dimensions("src", sbuf.ndim - 1, sbuf.shape, 3);
    auto ssize = sbuf.size / sbuf.shape[sbuf.ndim - 1];

    if (!idxs) fill_indices("idxs", ssize, size, idxs);
    else check_indices("idxs", ssize, size, idxs);

    py::buffer_info ibuf = idxs.value().request();
    if (size != ibuf.size)
    {
        std::ostringstream oss1, oss2;
        std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument("karr and idxs have incompatible shapes: {" + oss1.str() + "}, {" + oss2.str() + "}");
    }

    auto ct_k = py::array_t<F>(kbuf.shape);
    auto ct_src = py::array_t<F>(sbuf.shape);

    fill_array(ct_src, F());

    auto ct_karr = array<F>(ct_k.request());
    auto ct_sarr = array<F>(ct_src.request());

    auto ct_xarr = array<F>(ct_xbuf);
    auto ct_yarr = array<F>(ct_ybuf);
    auto karr = array<F>(kbuf);
    auto iarr = array<I>(ibuf);
    auto sarr = array<F>(sbuf);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<F> buffer (sarr.size, F());

        #pragma omp for nowait
        for (py::ssize_t i = 0; i < size; i++)
        {
            e.run([&]
            {
                to_tie<3>(ct_karr, 3 * i) = k2det_vjp({ct_xarr[i], ct_yarr[i]}, to_tie<3>(buffer, 3 * iarr[i]),
                                                      to_array<3>(karr, 3 * i), to_array<3>(sarr, 3 * iarr[i]));
            });
        }

        #pragma omp critical
        {
            std::transform(buffer.begin(), buffer.end(), ct_sarr.begin(), ct_sarr.begin(), std::plus<F>());
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(ct_k, ct_src);
}

template <typename T>
auto k2smp(const std::array<T, 3> & k, const std::array<T, 3> & src, T z)
{
    return std::make_tuple(src[0] + k[0] / k[2] * (z - src[2]), src[1] + k[1] / k[2] * (z - src[2]), z);
}

template <typename T>
auto k2smp_vjp(const std::array<T, 3> & ct, std::array<T, 3> & ct_src, T & ct_z, const std::array<T, 3> & k, const std::array<T, 3> & src, T z)
{
    ct_src[0] += ct[0];
    ct_src[1] += ct[1];
    ct_src[2] += -ct[0] * k[0] / k[2] - ct[1] * k[1] / k[2];

    ct_z += ct[0] * k[0] / k[2] + ct[1] * k[1] / k[2] + ct[2];

    return std::make_tuple(ct[0] * (z - src[2]) / k[2], ct[1] * (z - src[2]) / k[2],
                           ct[0] * (src[2] - z) * k[0] / (k[2] * k[2]) + ct[1] * (src[2] - z) * k[1] / (k[2] * k[2]));
}

template <typename F, typename I>
py::array_t<F> k_to_smp(py_array_t<F> k, py_array_t<F> z, std::array<F, 3> src, std::optional<py_array_t<I>> idxs, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info kbuf = k.request(), zbuf = z.request();

    check_dimensions("k", kbuf.ndim - 1, kbuf.shape, 3);
    auto size = kbuf.size / kbuf.shape[kbuf.ndim - 1];

    if (!idxs) fill_indices("idxs", zbuf.size, size, idxs);
    else check_indices("idxs", zbuf.size, size, idxs);

    py::buffer_info ibuf = idxs.value().request();
    if (size != ibuf.size)
    {
        std::ostringstream oss1, oss2;
        std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument("k and idxs have incompatible shapes: {" + oss1.str() + "}, {" + oss2.str() + "}");
    }

    std::vector<py::ssize_t> out_shape;
    std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::back_inserter(out_shape));

    auto pts = py::array_t<F>(out_shape);

    auto parr = array<F>(pts.request());
    auto karr = array<F>(kbuf);
    auto iarr = array<I>(ibuf);
    auto zarr = array<F>(zbuf);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < size; i++)
    {
        e.run([&]
        {
            to_tie<3>(parr, 3 * i) = k2smp(to_array<3>(karr, 3 * i), src, zarr[iarr[i]]);
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return pts;
}

template <typename F, typename I>
auto k_to_smp_vjp(py_array_t<F> cotangents, py_array_t<F> k, py_array_t<F> z, std::array<F, 3> src, std::optional<py_array_t<I>> idxs, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info kbuf = k.request(), zbuf = z.request();

    check_dimensions("karr", kbuf.ndim - 1, kbuf.shape, 3);
    auto size = kbuf.size / kbuf.shape[kbuf.ndim - 1];

    if (!idxs) fill_indices("idxs", zbuf.size, size, idxs);
    else check_indices("idxs", zbuf.size, size, idxs);

    py::buffer_info ibuf = idxs.value().request();
    if (size != ibuf.size)
    {
        std::ostringstream oss1, oss2;
        std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument("k and idxs have incompatible shapes: {" + oss1.str() + "}, {" + oss2.str() + "}");
    }

    auto ct_k = py::array_t<F>(kbuf.shape);
    auto ct_z = py::array_t<F>(zbuf.shape);
    std::array<F, 3> ct_src {};

    fill_array(ct_z, F());

    auto ct_karr = array<F>(ct_k.request());
    auto ct_zarr = array<F>(ct_z.request());

    auto ctarr = array<F>(cotangents.request());
    auto karr = array<F>(kbuf);
    auto iarr = array<I>(ibuf);
    auto zarr = array<F>(zbuf);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<F> buffer (zarr.size, F());
        std::array<F, 3> ct_sbuffer {};

        #pragma omp for nowait
        for (py::ssize_t i = 0; i < size; i++)
        {
            e.run([&]
            {
                to_tie<3>(ct_karr, 3 * i) = k2smp_vjp(to_array<3>(ctarr, 3 * i), ct_sbuffer, buffer[iarr[i]], to_array<3>(karr, 3 * i),
                                                      src, zarr[iarr[i]]);
            });
        }

        #pragma omp critical
        {
            std::transform(buffer.begin(), buffer.end(), ct_zarr.begin(), ct_zarr.begin(), std::plus<F>());
            std::transform(ct_sbuffer.begin(), ct_sbuffer.end(), ct_src.begin(), ct_src.begin(), std::plus<F>());
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(ct_k, ct_z, ct_src);
}

template <typename T, typename V, typename U = std::common_type_t<T, V>>
std::tuple<U, U, U> multiply(std::array<V, 3> vec, std::array<T, 9> mat)
{
    return std::make_tuple(dot(std::array<T, 3>{mat[0], mat[3], mat[6]}, vec),
                           dot(std::array<T, 3>{mat[1], mat[4], mat[7]}, vec),
                           dot(std::array<T, 3>{mat[2], mat[5], mat[8]}, vec));
}

template <typename T>
using mat_tie_t = std::tuple<T &, T &, T &, T &, T &, T &, T &, T &, T &>;

template <typename T>
using vec_tie_t = std::tuple<T &, T &, T &>;

template <typename T, typename V, typename U = std::common_type_t<T, V>>
void multiply_vjp(std::array<U, 3> ct, vec_tie_t<U> ct_v, mat_tie_t<U> ct_m, std::array<V, 3> vec, std::array<T, 9> mat)
{
    std::get<0>(ct_v) += dot(std::array<T, 3>{mat[0], mat[1], mat[2]}, ct);
    std::get<1>(ct_v) += dot(std::array<T, 3>{mat[3], mat[4], mat[5]}, ct);
    std::get<2>(ct_v) += dot(std::array<T, 3>{mat[6], mat[7], mat[8]}, ct);

    std::get<0>(ct_m) += ct[0] * vec[0];
    std::get<1>(ct_m) += ct[1] * vec[0];
    std::get<2>(ct_m) += ct[2] * vec[0];
    std::get<3>(ct_m) += ct[0] * vec[1];
    std::get<4>(ct_m) += ct[1] * vec[1];
    std::get<5>(ct_m) += ct[2] * vec[1];
    std::get<6>(ct_m) += ct[0] * vec[2];
    std::get<7>(ct_m) += ct[1] * vec[2];
    std::get<8>(ct_m) += ct[2] * vec[2];
}

template <typename I>
size_t matmul_indices(size_t vsize, size_t msize, std::optional<py_array_t<I>> & vidxs,
                      std::optional<py_array_t<I>> & midxs)
{
    if (vidxs) check_indices("vidxs", vsize, vidxs.value().size(), vidxs);
    if (midxs) check_indices("midxs", msize, midxs.value().size(), midxs);

    if (!vidxs && !midxs)
    {
        if (vsize == msize)
        {
            fill_indices("vidxs", vsize, vsize, vidxs);
            fill_indices("midxs", msize, msize, midxs);
        }
        else if (vsize == 1)
        {
            fill_indices("vidxs", vsize, msize, vidxs);
            fill_indices("midxs", msize, msize, midxs);
        }
        else if (msize == 1)
        {
            fill_indices("vidxs", vsize, vsize, vidxs);
            fill_indices("midxs", msize, vsize, midxs);
        }
        else throw std::invalid_argument("vidxs and midxs are not defined");
    }
    else if (!vidxs)
    {
        fill_indices("vidxs", vsize, midxs.value().size(), vidxs);
    }
    else if (!midxs)
    {
        fill_indices("midxs", msize, vidxs.value().size(), midxs);
    }

    if (midxs.value().size() != vidxs.value().size())
        throw std::invalid_argument("vidxs and midxs have incompatible shapes");

    auto viarr = array<I>(vidxs.value().request());
    auto miarr = array<I>(midxs.value().request());
    check_equal("vidxs and midxs have incompatible shapes",
                viarr.shape.begin(), viarr.shape.end(),
                miarr.shape.begin(), miarr.shape.end());

    return viarr.size;
}

template <typename F, typename I>
py::array_t<F> matmul(py_array_t<F> vecs, py_array_t<F> mats, std::optional<py_array_t<I>> vidxs,
                      std::optional<py_array_t<I>> midxs, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info vbuf = vecs.request(), mbuf = mats.request();

    check_dimensions("vecs", vbuf.ndim - 1, vbuf.shape, 3);
    size_t vsize = vbuf.size / vbuf.shape[vbuf.ndim - 1];

    check_dimensions("mats", mbuf.ndim - 2, mbuf.shape, 3, 3);
    size_t msize = mbuf.size / (mbuf.shape[mbuf.ndim - 1] * mbuf.shape[mbuf.ndim - 2]);

    auto size = matmul_indices(vsize, msize, vidxs, midxs);

    auto out_shape = vidxs.value().request().shape;
    out_shape.push_back(3);

    auto out = py::array_t<F>(out_shape);

    auto oarr = array<F>(out.request());
    auto varr = array<F>(vbuf);
    auto marr = array<F>(mbuf);
    auto viarr = array<I>(vidxs.value().request());
    auto miarr = array<I>(midxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < size; i++)
    {
        e.run([&]
        {
            to_tie<3>(oarr, 3 * i) = multiply(to_array<3>(varr, 3 * viarr[i]), to_array<9>(marr, 9 * miarr[i]));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename F, typename I>
auto matmul_vjp(py_array_t<F> cotangents, py_array_t<F> vecs, py_array_t<F> mats, std::optional<py_array_t<I>> vidxs,
                std::optional<py_array_t<I>> midxs, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info vbuf = vecs.request(), mbuf = mats.request();

    check_dimensions("vecs", vbuf.ndim - 1, vbuf.shape, 3);
    size_t vsize = vbuf.size / vbuf.shape[vbuf.ndim - 1];

    check_dimensions("mats", mbuf.ndim - 2, mbuf.shape, 3, 3);
    size_t msize = mbuf.size / (mbuf.shape[mbuf.ndim - 1] * mbuf.shape[mbuf.ndim - 2]);

    auto size = matmul_indices(vsize, msize, vidxs, midxs);

    auto ct_v = py::array_t<F>(vbuf.shape);
    auto ct_m = py::array_t<F>(mbuf.shape);

    fill_array(ct_v, F());
    fill_array(ct_m, F());

    auto ct_varr = array<F>(ct_v.request());
    auto ct_marr = array<F>(ct_m.request());

    auto ctarr = array<F>(cotangents.request());
    auto varr = array<F>(vbuf);
    auto marr = array<F>(mbuf);
    auto viarr = array<I>(vidxs.value().request());
    auto miarr = array<I>(midxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<F> vbuffer (varr.size, F());
        std::vector<F> mbuffer (marr.size, F());

        #pragma omp for nowait
        for (size_t i = 0; i < size; i++)
        {
            e.run([&]
            {
                multiply_vjp(to_array<3>(ctarr, 3 * i), to_tie<3>(vbuffer, 3 * viarr[i]), to_tie<9>(mbuffer, 9 * miarr[i]),
                             to_array<3>(varr, 3 * viarr[i]), to_array<9>(marr, 9 * miarr[i]));
            });
        }

        #pragma omp critical
        {
            std::transform(vbuffer.begin(), vbuffer.end(), ct_varr.begin(), ct_varr.begin(), std::plus<F>());
            std::transform(mbuffer.begin(), mbuffer.end(), ct_marr.begin(), ct_marr.begin(), std::plus<F>());
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(ct_v, ct_m);
}

/*----------------------------------------------------------------------
    Rectangular bounds consist of four lines in (x, y) plane defined as:

    r = s + t * e

    Solving an equation for the plane of source points intersecting with
    a unit sphere:

    o . q = (s + t * e) * q_xy + sqrt(1 - (s + t * e)**2) * qz

    The equation yields a quadratic equation as follows:

    a * t^2 - 2 b * t + c = 0

    where:

    a = f2 * f2 + q_z^2 * e^2
    b = f1 * f2 - q_z^2 * e . s
    c = f1 * f1 - q_z^2 * (1 - s^2)

    f1 = o . q - s . q          f2 = q . e
-----------------------------------------------------------------------*/
template <typename T>
using solution_t = std::pair<T, std::array<T, 3>>;

template <typename F>
auto find_intersection(const std::array<F, 3> & q, const Line<F> & edge)
{
    Point<F> q_xy {q[0], q[1]};
    F prod = -magnitude(q) / 2;
    F f1 = prod - dot(edge.pt0, q_xy);
    F f2 = dot(edge.tau, q_xy);

    F a = f2 * f2 + q[2] * q[2] * magnitude(edge.tau);
    F b = f1 * f2 - q[2] * q[2] * dot(edge.pt0, edge.tau);
    F c = f1 * f1 - q[2] * q[2] * (1 - magnitude(edge.pt0));

    auto get_point = [&edge](F t)
    {
        auto kxy = edge.pt0 + t * edge.tau;
        auto kz = std::sqrt(1 - magnitude(kxy));
        return std::make_pair(edge.distance(kxy), std::array<F, 3>{kxy[0], kxy[1], kz});
    };

    if (b * b > a * c)
    {
        F delta = std::sqrt(b * b - a * c);
        auto s0 = get_point((b - delta) / a), s1 = get_point((b + delta) / a);

        if (isclose(dot(s0.second, q), prod)) return s0;
        if (isclose(dot(s1.second, q), prod)) return s1;
    }
    return std::make_pair(std::numeric_limits<F>::max(), std::array<F, 3>{});
}

template <typename T>
using line_tie_t = std::tuple<T &, T &, T &, T &>;

template <typename F>
auto find_intersection_vjp(const std::array<F, 3> & ct, std::tuple<F &, F &, F &> ct_q, line_tie_t<F> ct_e, const std::array<F, 3> & kin, const std::array<F, 3> & q, const Line<F> & edge)
{
    Point<F> k_xy {kin[0], kin[1]};
    Point<F> q_xy {q[0], q[1]};

    F prod = -magnitude(q) / 2;
    F f1 = prod - dot(edge.pt0, q_xy);
    F f2 = dot(edge.tau, q_xy);

    F a = f2 * f2 + q[2] * q[2] * magnitude(edge.tau);
    F b = f1 * f2 - q[2] * q[2] * dot(edge.pt0, edge.tau);
    F c = f1 * f1 - q[2] * q[2] * (1 - magnitude(edge.pt0));

    F delta = std::sqrt(b * b - a * c);
    auto ct_t = dot(Point(ct[0], ct[1]), edge.tau) - ct[2] * dot(edge.tau, k_xy) / kin[2];
    auto t = dot(k_xy - edge.pt0, edge.tau) / magnitude(edge.tau);

    std::get<0>(ct_e) += ct[0] * (1 - t) - ct[2] * (1 - t) * k_xy[0] / kin[2];
    std::get<1>(ct_e) += ct[1] * (1 - t) - ct[2] * (1 - t) * k_xy[1] / kin[2];
    std::get<2>(ct_e) += ct[0] * t - ct[2] * t * k_xy[0] / kin[2];
    std::get<3>(ct_e) += ct[1] * t - ct[2] * t * k_xy[1] / kin[2];

    std::array<F, 3> ct_abc;
    if (isclose(t, (b - delta) / a))
    {
        ct_abc[0] = ct_t * (c / (2 * a * delta) - (b - delta) / (a * a));
        ct_abc[1] = ct_t * (1 / a - b / (a * delta));
        ct_abc[2] = ct_t / (2 * delta);
    }
    else
    {
        ct_abc[0] = ct_t * (-c / (2 * a * delta) - (b + delta) / (a * a));
        ct_abc[1] = ct_t * (1 / a + b / (a * delta));
        ct_abc[2] = -ct_t / (2 * delta);
    }

    std::get<0>(ct_e) += ct_abc[0] * (-2 * q[0] * f2 - 2 * q[2] * q[2] * edge.tau[0]) +
                         ct_abc[1] * (-q[0] * f2 - q[0] * f1 + q[2] * q[2] * (2 * edge.pt0[0] - edge.pt1[0])) +
                         ct_abc[2] * (-2 * q[0] * f1 + 2 * q[2] * q[2] * edge.pt0[0]);
    std::get<1>(ct_e) += ct_abc[0] * (-2 * q[1] * f2 - 2 * q[2] * q[2] * edge.tau[1]) +
                         ct_abc[1] * (-q[1] * f2 - q[1] * f1 + q[2] * q[2] * (2 * edge.pt0[1] - edge.pt1[1])) +
                         ct_abc[2] * (-2 * q[1] * f1 + 2 * q[2] * q[2] * edge.pt0[1]);
    std::get<2>(ct_e) += ct_abc[0] * (2 * q[0] * f2 + 2 * q[2] * q[2] * edge.tau[0]) +
                         ct_abc[1] * (q[0] * f1 - q[2] * q[2] * edge.pt0[0]);
    std::get<3>(ct_e) += ct_abc[0] * (2 * q[1] * f2 + 2 * q[2] * q[2] * edge.tau[1]) +
                         ct_abc[1] * (q[1] * f1 - q[2] * q[2] * edge.pt0[1]);

    std::get<0>(ct_q) += ct_abc[0] * 2 * edge.tau[0] * f2 +
                         ct_abc[1] * ((-q[0] - edge.pt0[0]) * f2 + edge.tau[0] * f1) +
                         ct_abc[2] * 2 * (-q[0] - edge.pt0[0]) * f1;
    std::get<1>(ct_q) += ct_abc[0] * 2 * edge.tau[1] * f2 +
                         ct_abc[1] * ((-q[1] - edge.pt0[1]) * f2 + edge.tau[1] * f1) +
                         ct_abc[2] * 2 * (-q[1] - edge.pt0[1]) * f1;
    std::get<2>(ct_q) += ct_abc[0] * 2 * q[2] * magnitude(edge.tau) +
                         ct_abc[1] * (-q[2] * f2 - 2 * q[2] * dot(edge.pt0, edge.tau)) +
                         ct_abc[2] * (-2 * q[2] * (1 - magnitude(edge.pt0)) - 2 * q[2] * f1);
}

template <typename F>
py::array_t<F> source_lines(py_array_t<F> q, std::array<F, 2> kmin, std::array<F, 2> kmax, unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info qbuf = q.request();

    check_dimensions("q", qbuf.ndim - 1, qbuf.shape, 3);
    auto qsize = qbuf.size / qbuf.shape[qbuf.ndim - 1];

    std::vector<py::ssize_t> out_shape;
    std::copy(qbuf.shape.begin(), std::prev(qbuf.shape.end()), std::back_inserter(out_shape));
    out_shape.push_back(2); out_shape.push_back(3);

    auto out = py::array_t<F>(out_shape);

    auto oarr = array<F>(out.request());
    auto qarr = array<F>(qbuf);

    thread_exception e;

    py::gil_scoped_release release;

    std::array<Line<F>, 4> edges {Line<F>(Point(kmin[0], kmin[1]), Point(kmin[0], kmax[1])),
                                  Line<F>(Point(kmin[0], kmin[1]), Point(kmax[0], kmin[1])),
                                  Line<F>(Point(kmax[0], kmin[1]), Point(kmax[0], kmax[1])),
                                  Line<F>(Point(kmin[0], kmax[1]), Point(kmax[0], kmax[1]))};

    #pragma omp parallel num_threads(threads)
    {
        std::array<solution_t<F>, 4> solutions;

        auto compare_solution = [](const solution_t<F> & a, const solution_t<F> & b){return a.first < b.first;};

        #pragma omp for
        for (py::ssize_t i = 0; i < qsize; i++)
        {
            e.run([&]
            {
                auto get_solution = [&qarr, i](const Line<F> & edge){return find_intersection(to_array<3>(qarr, 3 * i), edge);};

                std::transform(edges.begin(), edges.end(), solutions.begin(), get_solution);
                std::sort(solutions.begin(), solutions.end(), compare_solution);

                if (isclose(solutions[0].first + solutions[1].first, F()))
                {
                    to_tie<3>(oarr, 6 * i) = to_tuple(solutions[0].second);
                    to_tie<3>(oarr, 6 * i + 3) = to_tuple(solutions[1].second);
                }
                else std::fill_n(std::next(oarr.begin(), 6 * i), 6, F());
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename F>
auto source_lines_vjp(py_array_t<F> cotangents, py_array_t<F> kin, py_array_t<F> q, std::array<F, 2> kmin, std::array<F, 2> kmax,
                      unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info qbuf = q.request(), kbuf = kin.request();

    check_dimensions("q", qbuf.ndim - 1, qbuf.shape, 3);
    auto qsize = qbuf.size / qbuf.shape[qbuf.ndim - 1];

    check_dimensions("kin", kbuf.ndim - 2, kbuf.shape, 2, 3);
    check_equal("kin and q have incompatible shapes", qbuf.shape.begin(), std::prev(qbuf.shape.end()),
                kbuf.shape.begin(), std::prev(kbuf.shape.end(), 2));

    auto ct_q = py::array_t<F>(qbuf.shape);
    std::array<F, 2> ct_kmin {}, ct_kmax {};

    fill_array(ct_q, F());

    auto ct_qarr = array<F>(ct_q.request());
    auto ctarr = array<F>(cotangents.request());
    auto karr = array<F>(kbuf);
    auto qarr = array<F>(qbuf);

    thread_exception e;

    py::gil_scoped_release release;

    std::array<Line<F>, 4> edges {Line<F>(Point(kmin[0], kmin[1]), Point(kmin[0], kmax[1])),
                                  Line<F>(Point(kmin[0], kmin[1]), Point(kmax[0], kmin[1])),
                                  Line<F>(Point(kmax[0], kmin[1]), Point(kmax[0], kmax[1])),
                                  Line<F>(Point(kmin[0], kmax[1]), Point(kmax[0], kmax[1]))};

    #pragma omp parallel num_threads(threads)
    {
        std::array<F, 2> ct_k0 {}, ct_k1 {};
        std::array<line_tie_t<F>, 4> ct_edges = {std::tie(ct_k0[0], ct_k0[1], ct_k0[0], ct_k1[1]),
                                                 std::tie(ct_k0[0], ct_k0[1], ct_k1[0], ct_k0[1]),
                                                 std::tie(ct_k1[0], ct_k0[1], ct_k1[0], ct_k1[1]),
                                                 std::tie(ct_k0[0], ct_k1[1], ct_k1[0], ct_k1[1])};

        #pragma omp for nowait
        for (py::ssize_t i = 0; i < qsize; i++)
        {
            e.run([&]
            {
                for (auto [edge, ct_edge] : zip::zip(edges, ct_edges))
                {
                    for (auto idx : {6 * i, 6 * i + 3})
                    {
                        if (isclose(edge.distance(to_point(karr, idx)), F()))
                        {
                            find_intersection_vjp(to_array<3>(ctarr, idx), to_tie<3>(ct_qarr, 3 * i), ct_edge, to_array<3>(karr, idx),
                                                  to_array<3>(qarr, 3 * i), edge);
                        }
                    }
                }
            });
        }

        #pragma omp critical
        {
            std::transform(ct_k0.begin(), ct_k0.end(), ct_kmin.begin(), ct_kmin.begin(), std::plus<F>());
            std::transform(ct_k1.begin(), ct_k1.end(), ct_kmax.begin(), ct_kmax.begin(), std::plus<F>());
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(ct_q, ct_kmin, ct_kmax);
}

}

PYBIND11_MODULE(geometry, m)
{
    using namespace cbclib;
    py::options options;
    options.disable_function_signatures();

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("euler_angles", &euler_angles<float>, py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("euler_angles", &euler_angles<double>, py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("euler_angles_vjp", &euler_angles_vjp<float>, py::arg("cotangents"), py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("euler_angles_vjp", &euler_angles_vjp<double>, py::arg("cotangents"), py::arg("rmats"), py::arg("num_threads") = 1);

    m.def("euler_matrix", &euler_matrix<float>, py::arg("angles"), py::arg("num_threads") = 1);
    m.def("euler_matrix", &euler_matrix<double>, py::arg("angles"), py::arg("num_threads") = 1);
    m.def("euler_matrix_vjp", &euler_matrix_vjp<float>, py::arg("cotangents"), py::arg("angles"), py::arg("num_threads") = 1);
    m.def("euler_matrix_vjp", &euler_matrix_vjp<double>, py::arg("cotangents"), py::arg("angles"), py::arg("num_threads") = 1);

    m.def("tilt_angles", &tilt_angles<float>, py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("tilt_angles", &tilt_angles<double>, py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("tilt_angles_vjp", &tilt_angles_vjp<float>, py::arg("cotangents"), py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("tilt_angles_vjp", &tilt_angles_vjp<double>, py::arg("cotangents"), py::arg("rmats"), py::arg("num_threads") = 1);

    m.def("tilt_matrix", &tilt_matrix<float>, py::arg("angles"), py::arg("num_threads") = 1);
    m.def("tilt_matrix", &tilt_matrix<double>, py::arg("angles"), py::arg("num_threads") = 1);
    m.def("tilt_matrix_vjp", &tilt_matrix_vjp<float>, py::arg("cotangents"), py::arg("angles"), py::arg("num_threads") = 1);
    m.def("tilt_matrix_vjp", &tilt_matrix_vjp<double>, py::arg("cotangents"), py::arg("angles"), py::arg("num_threads") = 1);

    m.def("fill_indices",
        [](size_t xsize, size_t isize, std::optional<py_array_t<size_t>> idxs)
        {
            if (!idxs) fill_indices("idxs", xsize, isize, idxs);
            else check_indices("idxs", xsize, isize, idxs);
            return idxs.value();
        }, py::arg("xsize"), py::arg("isize"), py::arg("idxs") = nullptr);
    m.def("fill_indices",
        [](size_t xsize, size_t isize, std::optional<py_array_t<long>> idxs)
        {
            if (!idxs) fill_indices("idxs", xsize, isize, idxs);
            else check_indices("idxs", xsize, isize, idxs);
            return idxs.value();
        }, py::arg("xsize"), py::arg("isize"), py::arg("idxs") = nullptr);

    m.def("det_to_k", &det_to_k<float, size_t>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k", &det_to_k<double, size_t>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k", &det_to_k<float, long>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k", &det_to_k<double, long>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k_vjp", &det_to_k_vjp<float, size_t>, py::arg("cotangents"), py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k_vjp", &det_to_k_vjp<double, size_t>, py::arg("cotangents"), py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k_vjp", &det_to_k_vjp<float, long>, py::arg("cotangents"), py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k_vjp", &det_to_k_vjp<double, long>, py::arg("cotangents"), py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);

    m.def("k_to_det", &k_to_det<float, size_t>, py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det", &k_to_det<double, size_t>, py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det", &k_to_det<float, long>, py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det", &k_to_det<double, long>, py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det_vjp", &k_to_det_vjp<float, size_t>, py::arg("cotangents"), py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det_vjp", &k_to_det_vjp<double, size_t>, py::arg("cotangents"), py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det_vjp", &k_to_det_vjp<float, long>, py::arg("cotangents"), py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det_vjp", &k_to_det_vjp<double, long>, py::arg("cotangents"), py::arg("k"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);

    m.def("k_to_smp", &k_to_smp<float, size_t>, py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp", &k_to_smp<double, size_t>, py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp", &k_to_smp<float, long>, py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp", &k_to_smp<double, long>, py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp_vjp", &k_to_smp_vjp<float, size_t>, py::arg("cotangents"), py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp_vjp", &k_to_smp_vjp<double, size_t>, py::arg("cotangents"), py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp_vjp", &k_to_smp_vjp<float, long>, py::arg("cotangents"), py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp_vjp", &k_to_smp_vjp<double, long>, py::arg("cotangents"), py::arg("k"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);

    m.def("matmul_indices",
        [](size_t vsize, size_t msize, std::optional<py_array_t<size_t>> vidxs, std::optional<py_array_t<size_t>> midxs)
        {
            matmul_indices(vsize, msize, vidxs, midxs);
            return std::make_tuple(vidxs, midxs);
        }, py::arg("vsize"), py::arg("msize"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr);
    m.def("matmul_indices",
        [](size_t vsize, size_t msize, std::optional<py_array_t<long>> vidxs, std::optional<py_array_t<long>> midxs)
        {
            matmul_indices(vsize, msize, vidxs, midxs);
            return std::make_tuple(vidxs, midxs);
        }, py::arg("vsize"), py::arg("msize"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr);

    m.def("matmul", &matmul<float, size_t>, py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);
    m.def("matmul", &matmul<double, size_t>, py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);
    m.def("matmul", &matmul<float, long>, py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);
    m.def("matmul", &matmul<double, long>, py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);
    m.def("matmul_vjp", &matmul_vjp<float, size_t>, py::arg("cotangents"), py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);
    m.def("matmul_vjp", &matmul_vjp<double, size_t>, py::arg("cotangents"), py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);
    m.def("matmul_vjp", &matmul_vjp<float, long>, py::arg("cotangents"), py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);
    m.def("matmul_vjp", &matmul_vjp<double, long>, py::arg("cotangents"), py::arg("vecs"), py::arg("mats"), py::arg("vidxs") = nullptr, py::arg("midxs") = nullptr,  py::arg("num_threads") = 1);

    m.def("source_lines", &source_lines<float>, py::arg("q"), py::arg("kmin"), py::arg("kmax"), py::arg("num_threads") = 1);
    m.def("source_lines", &source_lines<double>, py::arg("q"), py::arg("kmin"), py::arg("kmax"), py::arg("num_threads") = 1);
    m.def("source_lines_vjp", &source_lines_vjp<float>, py::arg("cotangents"), py::arg("kin"), py::arg("q"), py::arg("kmin"), py::arg("kmax"), py::arg("num_threads") = 1);
    m.def("source_lines_vjp", &source_lines_vjp<double>, py::arg("cotangents"), py::arg("kin"), py::arg("q"), py::arg("kmin"), py::arg("kmax"), py::arg("num_threads") = 1);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
