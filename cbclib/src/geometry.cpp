#include "array.hpp"
#include "geometry.hpp"

namespace cbclib {

template<typename F, typename>
py::array_t<F> euler_angles(py::array_t<F, py::array::c_style | py::array::forcecast> rmats, unsigned threads)
{
    assert(PyArray_API);

    auto rbuf = rmats.request();
    check_dimensions("rmats", rbuf.ndim - 2, rbuf.shape, 3, 3);
    auto rsize = rbuf.size / 9;

    std::vector<py::ssize_t> ashape;
    std::copy_n(rbuf.shape.begin(), rbuf.ndim - 2, std::back_inserter(ashape));
    ashape.push_back(3);

    auto angles = py::array_t<F>(ashape);

    auto aptr = static_cast<F *>(angles.request().ptr);
    auto rptr = static_cast<F *>(rbuf.ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < rsize; i++)
    {
        aptr[3 * i + 1] = acos(rptr[9 * i + 8]);
        if (isclose(aptr[3 * i + 1], F()))
        {
            aptr[3 * i] = atan2(-rptr[9 * i + 3], rptr[9 * i]); aptr[3 * i + 2] = F();
        }
        else if (isclose(aptr[3 * i + 1], F(M_PI)))
        {
            aptr[3 * i] = atan2(rptr[9 * i + 3], rptr[9 * i]); aptr[3 * i + 2] = F();
        }
        else
        {
            aptr[3 * i] = atan2(rptr[9 * i + 6], -rptr[9 * i + 7]);
            aptr[3 * i + 2] = atan2(rptr[9 * i + 2], rptr[9 * i + 5]);
        }
        if (aptr[3 * i] < F()) aptr[3 * i] += 2.0 * M_PI;
        if (aptr[3 * i + 2] < F()) aptr[3 * i + 2] += 2.0 * M_PI;
    }

    py::gil_scoped_acquire acquire;

    return angles;
}

template<typename F, typename>
py::array_t<F> euler_matrix(py::array_t<F, py::array::c_style | py::array::forcecast> angles, unsigned threads)
{
    assert(PyArray_API);

    auto abuf = angles.request();
    check_dimensions("angles", abuf.ndim - 1, abuf.shape, 3);
    auto asize = abuf.size / 3;

    std::vector<py::ssize_t> rshape;
    std::copy_n(abuf.shape.begin(), abuf.ndim - 1, std::back_inserter(rshape));
    rshape.push_back(3); rshape.push_back(3);

    auto rmats = py::array_t<F>(rshape);

    auto aptr = static_cast<F *>(abuf.ptr);
    auto rptr = static_cast<F *>(rmats.request().ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < asize; i++)
    {
        auto c0 = cos(aptr[3 * i]), c1 = cos(aptr[3 * i + 1]), c2 = cos(aptr[3 * i + 2]);
        auto s0 = sin(aptr[3 * i]), s1 = sin(aptr[3 * i + 1]), s2 = sin(aptr[3 * i + 2]);

        rptr[9 * i    ] = c0 * c2 - s0 * s2 * c1;
        rptr[9 * i + 1] = s0 * c2 + c0 * s2 * c1;
        rptr[9 * i + 2] = s2 * s1;
        rptr[9 * i + 3] = -c0 * s2 - s0 * c2 * c1;
        rptr[9 * i + 4] = -s0 * s2 + c0 * c2 * c1;
        rptr[9 * i + 5] = c2 * s1;
        rptr[9 * i + 6] = s0 * s1;
        rptr[9 * i + 7] = -c0 * s1;
        rptr[9 * i + 8] = c1;
    }

    py::gil_scoped_acquire acquire;

    return rmats;
}

template<typename F, typename>
py::array_t<F> tilt_angles(py::array_t<F, py::array::c_style | py::array::forcecast> rmats, unsigned threads)
{
    assert(PyArray_API);

    auto rbuf = rmats.request();
    check_dimensions("rmats", rbuf.ndim - 2, rbuf.shape, 3, 3);
    auto rsize = rbuf.size / 9;

    std::vector<py::ssize_t> ashape;
    std::copy_n(rbuf.shape.begin(), rbuf.ndim - 2, std::back_inserter(ashape));
    ashape.push_back(3);

    auto angles = py::array_t<F>(ashape);

    auto aptr = static_cast<F *>(angles.request().ptr);
    auto rptr = static_cast<F *>(rbuf.ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < rsize; i++)
    {
        auto a0 = rptr[9 * i + 7] - rptr[9 * i + 5];
        auto a1 = rptr[9 * i + 2] - rptr[9 * i + 6];
        auto a2 = rptr[9 * i + 3] - rptr[9 * i + 1];
        auto l = sqrt(a0 * a0 + a1 * a1 + a2 * a2);

        aptr[3 * i    ] = acos((rptr[9 * i] + rptr[9 * i + 4] + rptr[9 * i + 8] - 1) / 2);
        aptr[3 * i + 1] = acos(a2 / l);
        aptr[3 * i + 2] = atan2(a1, a0);
    }

    py::gil_scoped_acquire acquire;

    return angles;
}

template<typename F, typename>
py::array_t<F> tilt_matrix(py::array_t<F, py::array::c_style | py::array::forcecast> angles, unsigned threads)
{
    assert(PyArray_API);

    auto abuf = angles.request();
    check_dimensions("angles", abuf.ndim - 1, abuf.shape, 3);
    auto asize = abuf.size / 3;

    std::vector<py::ssize_t> rshape;
    std::copy_n(abuf.shape.begin(), abuf.ndim - 1, std::back_inserter(rshape));
    rshape.push_back(3); rshape.push_back(3);

    auto rmats = py::array_t<F>(rshape);

    auto aptr = static_cast<F *>(abuf.ptr);
    auto rptr = static_cast<F *>(rmats.request().ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < asize; i++)
    {
        auto a = cos(aptr[3 * i] / 2);
        auto b = -sin(aptr[3 * i] / 2) * sin(aptr[3 * i + 1]) * cos(aptr[3 * i + 2]);
        auto c = -sin(aptr[3 * i] / 2) * sin(aptr[3 * i + 1]) * sin(aptr[3 * i + 2]);
        auto d = -sin(aptr[3 * i] / 2) * cos(aptr[3 * i + 1]);

        rptr[9 * i    ] = a * a + b * b - c * c - d * d;
        rptr[9 * i + 1] = 2 * (b * c + a * d);
        rptr[9 * i + 2] = 2 * (b * d - a * c);
        rptr[9 * i + 3] = 2 * (b * c - a * d);
        rptr[9 * i + 4] = a * a + c * c - b * b - d * d;
        rptr[9 * i + 5] = 2 * (c * d + a * b);
        rptr[9 * i + 6] = 2 * (b * d + a * c);
        rptr[9 * i + 7] = 2 * (c * d - a * b);
        rptr[9 * i + 8] = a * a + d * d - b * b - c * c;
    }

    py::gil_scoped_acquire acquire;

    return rmats;
}

template<typename F, typename I, typename>
py::array_t<F> det_to_k(py::array_t<F, py::array::c_style | py::array::forcecast> x,
                        py::array_t<F, py::array::c_style | py::array::forcecast> y,
                        py::array_t<F, py::array::c_style | py::array::forcecast> src,
                        std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
                        unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info xbuf = x.request(), ybuf = y.request(), sbuf = src.request();
    auto size = xbuf.size;

    check_dimensions("src", sbuf.ndim - 1, sbuf.shape, 3);
    auto ssize = sbuf.size / sbuf.shape[sbuf.ndim - 1];

    if (!idxs)
    {
        if (ssize == 1)
        {
            Py_intptr_t shape[1] = {size};
            auto descr = PyArray_DescrFromObject(py::dtype::of<I>().release().ptr(), nullptr);
            idxs = py::reinterpret_steal<py::array_t<I>>(PyArray_Zeros(1, shape, descr, 0));
        }
        else throw std::invalid_argument("idxs is not defined");
    }

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
    if (idxs.value().data()[ibuf.size - 1] + 1 > static_cast<I>(ssize))
        throw std::invalid_argument("invalid idxs value: " + std::to_string(idxs.value().data()[ibuf.size - 1]));

    std::vector<py::ssize_t> out_shape;
    std::copy(xbuf.shape.begin(), xbuf.shape.end(), std::back_inserter(out_shape));
    out_shape.push_back(3);

    auto out = py::array_t<F>(out_shape);

    auto optr = static_cast<F *>(out.request().ptr);
    auto xptr = static_cast<F *>(xbuf.ptr);
    auto yptr = static_cast<F *>(ybuf.ptr);
    auto iptr = static_cast<I *>(ibuf.ptr);
    auto sptr = static_cast<F *>(sbuf.ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < size; i++)
    {
        auto dist = sqrt(std::pow(xptr[i] - sptr[3 * iptr[i]], 2) + 
                         std::pow(yptr[i] - sptr[3 * iptr[i] + 1], 2) + 
                         std::pow(sptr[3 * iptr[i] + 2], 2));
        optr[3 * i    ] = (xptr[i] - sptr[3 * iptr[i]]) / dist;
        optr[3 * i + 1] = (yptr[i] - sptr[3 * iptr[i] + 1]) / dist;
        optr[3 * i + 2] = -sptr[3 * iptr[i] + 2] / dist;
    }

    py::gil_scoped_acquire acquire;

    return out;
}

template<typename F, typename I, typename>
auto k_to_det(py::array_t<F, py::array::c_style | py::array::forcecast> karr,
              py::array_t<F, py::array::c_style | py::array::forcecast> src,
              std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
              unsigned threads) -> std::tuple<py::array_t<F>, py::array_t<F>>
{
    assert(PyArray_API);

    py::buffer_info kbuf = karr.request(), sbuf = src.request();

    check_dimensions("karr", kbuf.ndim - 1, kbuf.shape, 3);
    auto size = kbuf.size / kbuf.shape[kbuf.ndim - 1];

    check_dimensions("src", sbuf.ndim - 1, sbuf.shape, 3);
    auto ssize = sbuf.size / sbuf.shape[sbuf.ndim - 1];

    if (!idxs)
    {
        if (ssize == 1)
        {
            Py_intptr_t shape[1] = {size};
            auto descr = PyArray_DescrFromObject(py::dtype::of<I>().release().ptr(), nullptr);
            idxs = py::reinterpret_steal<py::array_t<I>>(PyArray_Zeros(1, shape, descr, 0));
        }
        else throw std::invalid_argument("idxs is not defined");
    }

    py::buffer_info ibuf = idxs.value().request();
    if (size != ibuf.size)
    {
        std::ostringstream oss1, oss2;
        std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument("karr and idxs have incompatible shapes: {" + oss1.str() + "}, {" + oss2.str() + "}");
    }
    if (idxs.value().data()[ibuf.size - 1] + 1 > static_cast<I>(ssize))
        throw std::invalid_argument("invalid idxs value: " + std::to_string(idxs.value().data()[ibuf.size - 1]));

    std::vector<py::ssize_t> out_shape;
    std::copy_n(kbuf.shape.begin(), kbuf.ndim - 1, std::back_inserter(out_shape));

    auto x = py::array_t<F>(out_shape);
    auto y = py::array_t<F>(out_shape);

    auto xptr = static_cast<F *>(x.request().ptr);
    auto yptr = static_cast<F *>(y.request().ptr);
    auto kptr = static_cast<F *>(kbuf.ptr);
    auto iptr = static_cast<I *>(ibuf.ptr);
    auto sptr = static_cast<F *>(sbuf.ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < size; i++)
    {
        xptr[i] = sptr[3 * iptr[i]] - kptr[3 * i] / kptr[3 * i + 2] * sptr[3 * iptr[i] + 2];
        yptr[i] = sptr[3 * iptr[i] + 1] - kptr[3 * i + 1] / kptr[3 * i + 2] * sptr[3 * iptr[i] + 2];
    }

    py::gil_scoped_acquire acquire;

    return std::tuple(x, y);
}

template<typename F, typename I, typename>
py::array_t<F> k_to_smp(py::array_t<F, py::array::c_style | py::array::forcecast> karr,
                        py::array_t<F, py::array::c_style | py::array::forcecast> z, std::array<F, 3> src,
                        std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
                        unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info kbuf = karr.request(), zbuf = z.request();

    check_dimensions("karr", kbuf.ndim - 1, kbuf.shape, 3);
    auto size = kbuf.size / kbuf.shape[kbuf.ndim - 1];

    if (!idxs)
    {
        if (zbuf.size == 1)
        {
            Py_intptr_t shape[1] = {size};
            auto descr = PyArray_DescrFromObject(py::dtype::of<I>().release().ptr(), nullptr);
            idxs = py::reinterpret_steal<py::array_t<I>>(PyArray_Zeros(1, shape, descr, 0));
        }
        else throw std::invalid_argument("idxs is not defined");
    }

    py::buffer_info ibuf = idxs.value().request();
    if (size != ibuf.size)
    {
        std::ostringstream oss1, oss2;
        std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument("karr and idxs have incompatible shapes: {" + oss1.str() + "}, {" + oss2.str() + "}");
    }
    if (idxs.value().data()[ibuf.size - 1] + 1 > static_cast<I>(zbuf.size))
        throw std::invalid_argument("invalid idxs value: " + std::to_string(idxs.value().data()[ibuf.size - 1]));

    std::vector<py::ssize_t> out_shape;
    std::copy(kbuf.shape.begin(), kbuf.shape.end(), std::back_inserter(out_shape));

    auto pts = py::array_t<F>(out_shape);

    auto pptr = static_cast<F *>(pts.request().ptr);
    auto kptr = static_cast<F *>(kbuf.ptr);
    auto iptr = static_cast<I *>(ibuf.ptr);
    auto zptr = static_cast<F *>(zbuf.ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (py::ssize_t i = 0; i < size; i++)
    {
        pptr[3 * i    ] = src[0] + kptr[3 * i] / kptr[3 * i + 2] * (zptr[iptr[i]] - src[2]);
        pptr[3 * i + 1] = src[1] + kptr[3 * i + 1] / kptr[3 * i + 2] * (zptr[iptr[i]] - src[2]);
        pptr[3 * i + 2] = zptr[iptr[i]];
    }

    py::gil_scoped_acquire acquire;

    return pts;
}

/*----------------------------------------------------------------
    Rectangular bounds consist of four lines defined as:
    
    r = s + t * e, t \el [lim[0], lim[1]]

    Solving a quadratic equation:

    f1 = o . q - s . q          f2 = q . e

    a * t^2 - 2 b * t + c = 0
    a = f2^2 + q_z^2     b = f1 * f2
    c = f1^2 - (1 - s^2) * q_z^2
-----------------------------------------------------------------*/
template <typename F>
auto find_intersection(std::array<F, 3> q, std::array<F, 2> vec,
                       std::array<F, 2> origin, std::array<F, 2> limits) -> std::optional<std::array<F, 3>>
{
    /* vector product is given by: origin . q = - q^2 / 2 */
    F prod = -(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) / 2;
    F f1 = prod - origin[0] * q[0] - origin[1] * q[1];
    F f2 = vec[0] * q[0] + vec[1] * q[1];

    F a = f2 * f2 + q[2] * q[2], b = f1 * f2; 
    F c = f1 * f1 - (1 - origin[0] * origin[0] - origin[1] * origin[1]) * q[2] * q[2];

    auto get_point = [&](F t)
    {
        auto kx = origin[0] + t * vec[0];
        auto ky = origin[1] + t * vec[1];
        auto kz = sqrt(1 - kx * kx - ky * ky);
        return std::array<F, 3>{kx, ky, kz};
    };
    auto check_point = [&](const std::array<F, 3> & pt, F t)
    {
        return isclose(pt[0] * q[0] + pt[1] * q[1] + pt[2] * q[2], prod) && t >= limits[0] && t <= limits[1];
    };

    if (b * b > a * c)
    {
        F delta = sqrt(b * b - a * c);
        auto pt1 = get_point((b - delta) / a), pt2 = get_point((b + delta) / a);

        if (check_point(pt1, (b - delta) / a)) return std::make_optional(pt1);
        if (check_point(pt2, (b + delta) / a)) return std::make_optional(pt2);
    }
    return std::nullopt;
}

template<typename F, typename I, typename>
auto source_lines(py::array_t<I, py::array::c_style | py::array::forcecast> hkl,
                  py::array_t<F, py::array::c_style | py::array::forcecast> basis,
                  std::array<F, 2> kmin, std::array<F, 2> kmax,
                  std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> hidxs,
                  std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> bidxs,
                  unsigned threads) -> std::tuple<py::array_t<F>, py::array_t<bool>>
{
    assert(PyArray_API);

    py::buffer_info hbuf = hkl.request(), bbuf = basis.request();

    check_dimensions("hkl", hbuf.ndim - 1, hbuf.shape, 3);
    auto hsize = hbuf.size / hbuf.shape[hbuf.ndim - 1];

    check_dimensions("basis", bbuf.ndim - 2, bbuf.shape, 3, 3);
    auto bsize = bbuf.size / (bbuf.shape[bbuf.ndim - 1] * bbuf.shape[bbuf.ndim - 2]);

    if (!hidxs && !bidxs)
    {
        hidxs = py::array_t<I>(hsize * bsize);
        bidxs = py::array_t<I>(hsize * bsize);

        auto hiptr = static_cast<I *>(hidxs.value().request().ptr);
        auto biptr = static_cast<I *>(bidxs.value().request().ptr);
        for (py::ssize_t i = 0; i < hsize * bsize; i++)
        {
            hiptr[i] = i % hsize; biptr[i] = i / hsize;
        }
    }
    else if (!hidxs)
    {
        if (hsize == 1)
        {
            Py_intptr_t shape[1] = {hsize};
            auto descr = PyArray_DescrFromObject(py::dtype::of<I>().release().ptr(), nullptr);
            hidxs = py::reinterpret_steal<py::array_t<I>>(PyArray_Zeros(1, shape, descr, 0));
        }
        else throw std::invalid_argument("hidxs is not defined");
    }
    else if (!bidxs)
    {
        if (bsize == 1)
        {
            Py_intptr_t shape[1] = {bsize};
            auto descr = PyArray_DescrFromObject(py::dtype::of<I>().release().ptr(), nullptr);
            bidxs = py::reinterpret_steal<py::array_t<I>>(PyArray_Zeros(1, shape, descr, 0));
        }
        else throw std::invalid_argument("bidxs is not defined");
    }

    auto hibuf = hidxs.value().request(), bibuf = bidxs.value().request();
    check_equal("hidxs and bidxs have incompatible shapes",
                hibuf.shape.begin(), hibuf.shape.end(),
                bibuf.shape.begin(), bibuf.shape.end());

    std::vector<py::ssize_t> out_shape;
    std::copy(hibuf.shape.begin(), hibuf.shape.end(), std::back_inserter(out_shape));
    out_shape.push_back(2); out_shape.push_back(3);

    auto out = py::array_t<F>(out_shape);
    auto mask = py::array_t<bool>(hibuf.shape);

    auto optr = static_cast<F *>(out.request().ptr);
    auto mptr = static_cast<bool *>(mask.request().ptr);
    auto hptr = static_cast<I *>(hbuf.ptr);
    auto hiptr = static_cast<I *>(hibuf.ptr);
    auto bptr = static_cast<F *>(bbuf.ptr);
    auto biptr = static_cast<I *>(bibuf.ptr);

    thread_exception e;

    py::gil_scoped_release release;

    auto NA = sqrt(kmax[0] * kmax[0] + kmax[1] * kmax[1]);

    #pragma omp parallel num_threads(threads)
    {
        std::array<F, 3> q;
        std::array<std::optional<std::array<F, 3>>, 4> solutions;

        #pragma omp for
        for (py::ssize_t n = 0; n < hsize * bsize; n++)
        {
            e.run([&]
            {
                q[0] = hptr[3 * hiptr[n]    ] * bptr[9 * biptr[n]    ]
                     + hptr[3 * hiptr[n] + 1] * bptr[9 * biptr[n] + 3]
                     + hptr[3 * hiptr[n] + 2] * bptr[9 * biptr[n] + 6];
                q[1] = hptr[3 * hiptr[n]    ] * bptr[9 * biptr[n] + 1]
                     + hptr[3 * hiptr[n] + 1] * bptr[9 * biptr[n] + 4]
                     + hptr[3 * hiptr[n] + 2] * bptr[9 * biptr[n] + 7];
                q[2] = hptr[3 * hiptr[n]    ] * bptr[9 * biptr[n] + 2]
                     + hptr[3 * hiptr[n] + 1] * bptr[9 * biptr[n] + 5]
                     + hptr[3 * hiptr[n] + 2] * bptr[9 * biptr[n] + 8];

                auto q_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
                auto q_rho = q[2] * sqrt(1 / q_sq - 0.25) + sqrt(q[0] * q[0] + q[1] * q[1]) / 2;

                if (q_sq < 4 && abs(q_rho) < NA)
                {
                    solutions[0] = find_intersection(q, {0, 1}, {kmin[0], 0}, {kmin[1], kmax[1]});
                    solutions[1] = find_intersection(q, {1, 0}, {0, kmin[1]}, {kmin[0], kmax[0]});
                    solutions[2] = find_intersection(q, {0, 1}, {kmax[0], 0}, {kmin[1], kmax[1]});
                    solutions[3] = find_intersection(q, {1, 0}, {0, kmax[1]}, {kmin[0], kmax[0]});

                    if (std::transform_reduce(solutions.begin(), solutions.end(), 0, std::plus<int>(),
                                            [](std::optional<std::array<F, 3>> sol){return int(bool(sol));}) == 2)
                    {
                        auto oiter = optr + 6 * n;
                        for (auto sol: solutions)
                        {
                            if (sol) {std::copy(sol.value().begin(), sol.value().end(), oiter); oiter += sol.value().size();}
                        }
                        mptr[n] = true;
                    }
                    else
                    {
                        std::fill_n(optr + 6 * n, 6, F()); mptr[n] = false;
                    }
                }
                else
                {
                    std::fill_n(optr + 6 * n, 6, F()); mptr[n] = false;
                }
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::tuple(out, mask);
}

template<typename F, typename I, typename>
py::array_t<F> rotate_vec(py::array_t<F, py::array::c_style | py::array::forcecast> vecs,
                          py::array_t<F, py::array::c_style | py::array::forcecast> rmats,
                          std::optional<py::array_t<I, py::array::c_style | py::array::forcecast>> idxs,
                          unsigned threads)
{
    assert(PyArray_API);

    py::buffer_info vbuf = vecs.request(), rbuf = rmats.request();

    check_dimensions("vecs", vbuf.ndim - 1, vbuf.shape, 3);
    auto vsize = vbuf.size / vbuf.shape[vbuf.ndim - 1];

    check_dimensions("rmats", rbuf.ndim - 2, rbuf.shape, 3, 3);
    auto rsize = rbuf.size / (rbuf.shape[rbuf.ndim - 1] * rbuf.shape[rbuf.ndim - 2]);

    if (!idxs)
    {
        if (rsize == 1)
        {
            Py_intptr_t shape[1] = {vsize};
            auto descr = PyArray_DescrFromObject(py::dtype::of<I>().release().ptr(), nullptr);
            idxs = py::reinterpret_steal<py::array_t<I>>(PyArray_Zeros(1, shape, descr, 0));
        }
        else throw std::invalid_argument("idxs is not defined");
    }

    py::buffer_info ibuf = idxs.value().request();
    if (idxs.value().data()[ibuf.size - 1] + 1 > static_cast<I>(rsize))
        throw std::invalid_argument("invalid idxs value: " + std::to_string(idxs.value().data()[ibuf.size - 1]));

    auto out = py::array_t<F>(vbuf.shape);
    py::buffer_info obuf = out.request();

    auto *optr = static_cast<F *>(obuf.ptr);
    auto *vptr = static_cast<F *>(vbuf.ptr);
    auto *iptr = static_cast<I *>(ibuf.ptr);
    auto *rptr = static_cast<F *>(rbuf.ptr);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < vsize; i++)
    {
        optr[3 * i    ] = rptr[9 * iptr[i]    ] * vptr[3 * i] + rptr[9 * iptr[i] + 1] * vptr[3 * i + 1] + rptr[9 * iptr[i] + 2] * vptr[3 * i + 2];
        optr[3 * i + 1] = rptr[9 * iptr[i] + 3] * vptr[3 * i] + rptr[9 * iptr[i] + 4] * vptr[3 * i + 1] + rptr[9 * iptr[i] + 5] * vptr[3 * i + 2];
        optr[3 * i + 2] = rptr[9 * iptr[i] + 6] * vptr[3 * i] + rptr[9 * iptr[i] + 7] * vptr[3 * i + 1] + rptr[9 * iptr[i] + 8] * vptr[3 * i + 2];
    }

    py::gil_scoped_acquire acquire;

    return out;
}

}

PYBIND11_MODULE(geometry, m)
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

    m.def("euler_angles", &euler_angles<float>, py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("euler_angles", &euler_angles<double>, py::arg("rmats"), py::arg("num_threads") = 1);

    m.def("euler_matrix", &euler_matrix<float>, py::arg("angles"), py::arg("num_threads") = 1);
    m.def("euler_matrix", &euler_matrix<double>, py::arg("angles"), py::arg("num_threads") = 1);

    m.def("tilt_angles", &tilt_angles<float>, py::arg("rmats"), py::arg("num_threads") = 1);
    m.def("tilt_angles", &tilt_angles<double>, py::arg("rmats"), py::arg("num_threads") = 1);

    m.def("tilt_matrix", &tilt_matrix<float>, py::arg("angles"), py::arg("num_threads") = 1);
    m.def("tilt_matrix", &tilt_matrix<double>, py::arg("angles"), py::arg("num_threads") = 1);

    m.def("det_to_k", &det_to_k<float, size_t>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k", &det_to_k<double, size_t>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k", &det_to_k<float, long>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("det_to_k", &det_to_k<double, long>, py::arg("x"), py::arg("y"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);

    m.def("k_to_det", &k_to_det<float, size_t>, py::arg("karr"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det", &k_to_det<double, size_t>, py::arg("karr"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det", &k_to_det<float, long>, py::arg("karr"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_det", &k_to_det<double, long>, py::arg("karr"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);

    m.def("k_to_smp", &k_to_smp<float, size_t>, py::arg("karr"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp", &k_to_smp<double, size_t>, py::arg("karr"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp", &k_to_smp<float, long>, py::arg("karr"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("k_to_smp", &k_to_smp<double, long>, py::arg("karr"), py::arg("z"), py::arg("src"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);

    m.def("source_lines", &source_lines<float, size_t>, py::arg("hkl"), py::arg("basis"), py::arg("kmin"), py::arg("kmax"), py::arg("hidxs") = nullptr, py::arg("bidxs") = nullptr, py::arg("num_threads") = 1);
    m.def("source_lines", &source_lines<double, size_t>, py::arg("hkl"), py::arg("basis"), py::arg("kmin"), py::arg("kmax"), py::arg("hidxs") = nullptr, py::arg("bidxs") = nullptr, py::arg("num_threads") = 1);
    m.def("source_lines", &source_lines<float, long>, py::arg("hkl"), py::arg("basis"), py::arg("kmin"), py::arg("kmax"), py::arg("hidxs") = nullptr, py::arg("bidxs") = nullptr, py::arg("num_threads") = 1);
    m.def("source_lines", &source_lines<double, long>, py::arg("hkl"), py::arg("basis"), py::arg("kmin"), py::arg("kmax"), py::arg("hidxs") = nullptr, py::arg("bidxs") = nullptr, py::arg("num_threads") = 1);

    m.def("rotate", &rotate_vec<float, size_t>, py::arg("vecs"), py::arg("rmats"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("rotate", &rotate_vec<double, size_t>, py::arg("vecs"), py::arg("rmats"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("rotate", &rotate_vec<float, long>, py::arg("vecs"), py::arg("rmats"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);
    m.def("rotate", &rotate_vec<double, long>, py::arg("vecs"), py::arg("rmats"), py::arg("idxs") = nullptr, py::arg("num_threads") = 1);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}