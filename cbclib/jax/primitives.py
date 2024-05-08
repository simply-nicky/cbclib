from typing import Tuple
import numpy as np
import jax.numpy as jnp
from jax.core import Primitive, ShapedArray
from jax.lax import zeros_like_array
from jax.interpreters import ad
from ..src import geometry, image_proc

def make_zero(arr, prim):
    return zeros_like_array(prim) if isinstance(arr, ad.Zero) else arr

# Euler angles

euler_angles_p = Primitive('euler_angles')
euler_angles_jvp_p = Primitive('euler_angles_jvp')
euler_angles_vjp_p = Primitive('euler_angles_vjp')

def euler_angles(rmats, num_threads=1):
    return euler_angles_p.bind(rmats, num_threads)

@euler_angles_p.def_impl
def euler_angles_impl(rmats, num_threads):
    return geometry.euler_angles(np.asarray(rmats), num_threads)

def euler_angles_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    (rmats, num_threads), (rmats_dot, _) = primals, tangents
    angles = euler_angles(rmats, num_threads)
    angles_dot = euler_angles_jvp_p.bind(rmats, num_threads, rmats_dot)
    return angles, angles_dot

ad.primitive_jvps[euler_angles_p] = euler_angles_jvp

@euler_angles_jvp_p.def_abstract_eval
def euler_angles_jvp_abstract_eval(r_eval, nt_eval, rmats_dot):
    angles_dot_eval = ShapedArray(rmats_dot.shape[:-2] + (3,), rmats_dot.dtype)
    return angles_dot_eval

def euler_angles_jvp_transpose(ct_angles, rmats, num_threads, rmats_dot):
    assert ad.is_undefined_primal(rmats_dot)
    ct_rmats = euler_angles_vjp_p.bind(ct_angles, rmats, num_threads)
    return None, None, ct_rmats

ad.primitive_transposes[euler_angles_jvp_p] = euler_angles_jvp_transpose

@euler_angles_vjp_p.def_impl
def euler_angles_vjp_impl(ct_y, rmats, num_threads):
    return geometry.euler_angles_vjp(np.asarray(ct_y), np.asarray(rmats), num_threads)

# Euler matrix

euler_matrix_p = Primitive('euler_matrix')
euler_matrix_jvp_p = Primitive('euler_matrix_jvp')
euler_matrix_vjp_p = Primitive('euler_matrix_vjp')

def euler_matrix(angles, num_threads=1):
    return euler_matrix_p.bind(angles, num_threads)

@euler_matrix_p.def_impl
def euler_matrix_impl(angles, num_threads):
    return geometry.euler_matrix(np.asarray(angles), num_threads)

def euler_matrix_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    (angles, num_threads), (angles_dot, _) = primals, tangents
    rmats = euler_matrix(angles, num_threads)
    rmats_dot = euler_matrix_jvp_p.bind(angles, num_threads, angles_dot)
    return rmats, rmats_dot

ad.primitive_jvps[euler_matrix_p] = euler_matrix_jvp

@euler_matrix_jvp_p.def_abstract_eval
def euler_matrix_jvp_abstract_eval(a_eval, nt_eval, angles_dot):
    rmats_dot_eval = ShapedArray(angles_dot.shape[:-1] + (3, 3), angles_dot.dtype)
    return rmats_dot_eval

def euler_matrix_jvp_transpose(ct_rmats, angles, num_threads, angles_dot):
    assert ad.is_undefined_primal(angles_dot)
    ct_angles = euler_matrix_vjp_p.bind(ct_rmats, angles, num_threads)
    return None, None, ct_angles

ad.primitive_transposes[euler_matrix_jvp_p] = euler_matrix_jvp_transpose

@euler_matrix_vjp_p.def_impl
def euler_matrix_vjp_impl(ct_rmats, angles, num_threads):
    return geometry.euler_matrix_vjp(np.asarray(ct_rmats), np.asarray(angles), num_threads)

# Tilt angles

tilt_angles_p = Primitive('tilt_angles')
tilt_angles_jvp_p = Primitive('tilt_angles_jvp')
tilt_angles_vjp_p = Primitive('tilt_angles_vjp')

def tilt_angles(rmats, num_threads=1):
    return tilt_angles_p.bind(rmats, num_threads)

@tilt_angles_p.def_impl
def tilt_angles_impl(rmats, num_threads):
    return geometry.tilt_angles(np.asarray(rmats), num_threads)

def tilt_angles_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    (rmats, num_threads), (rmats_dot, _) = primals, tangents
    angles = tilt_angles(rmats, num_threads)
    angles_dot = tilt_angles_jvp_p.bind(rmats, num_threads, rmats_dot)
    return angles, angles_dot

ad.primitive_jvps[tilt_angles_p] = tilt_angles_jvp

@tilt_angles_jvp_p.def_abstract_eval
def tilt_angles_jvp_abstract_eval(r_eval, nt_eval, rmats_dot):
    angles_dot_eval = ShapedArray(rmats_dot.shape[:-2] + (3,), rmats_dot.dtype)
    return angles_dot_eval

def tilt_angles_jvp_transpose(ct_angles, rmats, num_threads, rmats_dot):
    assert ad.is_undefined_primal(rmats_dot)
    ct_rmats = tilt_angles_vjp_p.bind(ct_angles, rmats, num_threads)
    return None, None, ct_rmats

ad.primitive_transposes[tilt_angles_jvp_p] = tilt_angles_jvp_transpose

@tilt_angles_vjp_p.def_impl
def tilt_angles_vjp_impl(ct_y, rmats, num_threads):
    return geometry.tilt_angles_vjp(np.asarray(ct_y), np.asarray(rmats), num_threads)

# Tilt matrix

tilt_matrix_p = Primitive('tilt_matrix')
tilt_matrix_jvp_p = Primitive('tilt_matrix_jvp')
tilt_matrix_vjp_p = Primitive('tilt_matrix_vjp')

def tilt_matrix(angles, num_threads=1):
    return tilt_matrix_p.bind(angles, num_threads)

@tilt_matrix_p.def_impl
def tilt_matrix_impl(angles, num_threads):
    return geometry.tilt_matrix(np.asarray(angles), num_threads)

def tilt_matrix_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    (angles, num_threads), (angles_dot, _) = primals, tangents
    rmats = tilt_matrix(angles, num_threads)
    rmats_dot = tilt_matrix_jvp_p.bind(angles, num_threads, angles_dot)
    return rmats, rmats_dot

ad.primitive_jvps[tilt_matrix_p] = tilt_matrix_jvp

@tilt_matrix_jvp_p.def_abstract_eval
def tilt_matrix_jvp_abstract_eval(a_eval, nt_eval, angles_dot):
    rmats_dot_eval = ShapedArray(angles_dot.shape[:-1] + (3, 3), angles_dot.dtype)
    return rmats_dot_eval

def tilt_matrix_jvp_transpose(ct_rmats, angles, num_threads, angles_dot):
    assert ad.is_undefined_primal(angles_dot)
    ct_angles = tilt_matrix_vjp_p.bind(ct_rmats, angles, num_threads)
    return None, None, ct_angles

ad.primitive_transposes[tilt_matrix_jvp_p] = tilt_matrix_jvp_transpose

@tilt_matrix_vjp_p.def_impl
def tilt_matrix_vjp_impl(ct_rmats, angles, num_threads):
    return geometry.tilt_matrix_vjp(np.asarray(ct_rmats), np.asarray(angles), num_threads)

# det_to_k

det_to_k_p = Primitive('det_to_k')
det_to_k_jvp_p = Primitive('det_to_k_jvp')
det_to_k_vjp_p = Primitive('det_to_k_vjp')

def det_to_k(x, y, src, idxs=None, num_threads=1):
    idxs = geometry.fill_indices(src.size // src.shape[-1], x.size, idxs)
    return det_to_k_p.bind(x, y, src, idxs, num_threads)

@det_to_k_p.def_impl
def det_to_k_impl(x, y, src, idxs, num_threads):
    return geometry.det_to_k(np.asarray(x), np.asarray(y), np.asarray(src), np.asarray(idxs),
                             num_threads)

def det_to_k_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    x, y, src, idxs, num_threads = primals
    x_dot, y_dot, src_dot, _, _ = tangents

    k = det_to_k_p.bind(x, y, src, idxs, num_threads)

    k_dot = det_to_k_jvp_p.bind(x, y, src, idxs, num_threads, make_zero(x_dot, x),
                                make_zero(y_dot, x), make_zero(src_dot, src))
    return k, k_dot

ad.primitive_jvps[det_to_k_p] = det_to_k_jvp

@det_to_k_jvp_p.def_abstract_eval
def det_to_k_jvp_abstract_eval(x_eval, y_eval, s_eval, i_eval, nt_eval, x_dot, y_dot, src_dot):
    return ShapedArray(x_eval.shape + (3,), dtype=x_eval.dtype)

def det_to_k_jvp_transpose(ct_k, x, y, src, idxs, num_threads, x_dot, y_dot, src_dot):
    assert ad.is_undefined_primal(x_dot) or \
           ad.is_undefined_primal(y_dot) or \
           ad.is_undefined_primal(src_dot)
    ct_x, ct_y, ct_src = det_to_k_vjp_p.bind(ct_k, x, y, src, idxs, num_threads)
    return None, None, None, None, None, ct_x, ct_y, ct_src

ad.primitive_transposes[det_to_k_jvp_p] = det_to_k_jvp_transpose

@det_to_k_vjp_p.def_impl
def det_to_k_vjp_impl(ct_k, x, y, src, idxs, num_threads):
    return geometry.det_to_k_vjp(np.asarray(ct_k), np.asarray(x), np.asarray(y),
                                 np.asarray(src), np.asarray(idxs), num_threads)

# k_to_det

k_to_det_p = Primitive('k_to_det')
k_to_det_jvp_p = Primitive('k_to_det_jvp')
k_to_det_vjp_p = Primitive('k_to_det_vjp')

def k_to_det(k, src, idxs=None, num_threads=1):
    idxs = geometry.fill_indices(src.size // src.shape[-1], k.size // k.shape[-1], idxs)
    x, y = k_to_det_p.bind(k, src, idxs, num_threads)
    return x, y

@k_to_det_p.def_impl
def k_to_det_impl(k, src, idxs, num_threads):
    return np.stack(geometry.k_to_det(np.asarray(k), np.asarray(src), np.asarray(idxs),
                                      num_threads))

def k_to_det_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    k, src, idxs, num_threads = primals
    k_dot, src_dot, _, _ = tangents

    xy = k_to_det_p.bind(k, src, idxs, num_threads)

    xy_dot = k_to_det_jvp_p.bind(k, src, idxs, num_threads, make_zero(k_dot, k),
                                 make_zero(src_dot, src))
    return xy, xy_dot

ad.primitive_jvps[k_to_det_p] = k_to_det_jvp

@k_to_det_jvp_p.def_abstract_eval
def k_to_det_jvp_abstract_eval(k_eval, s_eval, i_eval, nt_eval, k_dot, src_dot):
    return ShapedArray((2,) + k_eval.shape[:-1], dtype=k_eval.dtype)

def k_to_det_jvp_transpose(cotangents, k, src, idxs, num_threads, k_dot, src_dot):
    ct_x, ct_y = cotangents
    assert ad.is_undefined_primal(k_dot) or ad.is_undefined_primal(src_dot)
    ct_k, ct_src = k_to_det_vjp_p.bind((ct_x, ct_y), k, src, idxs, num_threads)
    return None, None, None, None, ct_k, ct_src

ad.primitive_transposes[k_to_det_jvp_p] = k_to_det_jvp_transpose

@k_to_det_vjp_p.def_impl
def k_to_det_vjp_impl(cotangents, k, src, idxs, num_threads):
    return geometry.k_to_det_vjp(cotangents, np.asarray(k), np.asarray(src), np.asarray(idxs),
                                 num_threads)

# k_to_smp

k_to_smp_p = Primitive('k_to_smp')
k_to_smp_jvp_p = Primitive('k_to_smp_jvp')
k_to_smp_vjp_p = Primitive('k_to_smp_vjp')

def k_to_smp(k, z, src, idxs=None, num_threads=1):
    idxs = geometry.fill_indices(z.size, k.size // k.shape[-1], idxs)
    return k_to_smp_p.bind(k, z, jnp.asarray(src), idxs, num_threads)

@k_to_smp_p.def_impl
def k_to_smp_impl(k, z, src, idxs, num_threads):
    return geometry.k_to_smp(np.asarray(k), np.asarray(z), np.asarray(src), np.asarray(idxs),
                             num_threads)

def k_to_smp_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    k, z, src, idxs, num_threads = primals
    k_dot, z_dot, src_dot, _, _ = tangents

    pts = k_to_smp_p.bind(k, z, src, idxs, num_threads)

    pts_dot = k_to_smp_jvp_p.bind(k, z, src, idxs, num_threads, make_zero(k_dot, k),
                                  make_zero(z_dot, z), make_zero(src_dot, src))
    return pts, pts_dot

ad.primitive_jvps[k_to_smp_p] = k_to_smp_jvp

@k_to_smp_jvp_p.def_abstract_eval
def k_to_smp_jvp_abstract_eval(k_eval, z_eval, s_eval, i_eval, nt_eval, k_dot, z_dot, src_dot):
    return ShapedArray(k_eval.shape, dtype=k_eval.dtype)

def k_to_smp_jvp_transpose(ct_pts, k, z, src, idxs, num_threads, k_dot, z_dot, src_dot):
    assert ad.is_undefined_primal(k_dot) or \
           ad.is_undefined_primal(z_dot) or \
           ad.is_undefined_primal(src_dot)
    ct_k, ct_z, ct_src = k_to_smp_vjp_p.bind(ct_pts, k, z, src, idxs, num_threads)
    return None, None, None, None, None, ct_k, ct_z, ct_src

ad.primitive_transposes[k_to_smp_jvp_p] = k_to_smp_jvp_transpose

@k_to_smp_vjp_p.def_impl
def k_to_smp_vjp_impl(cotangents, k, z, src, idxs, num_threads):
    ct_k, ct_z, ct_src = geometry.k_to_smp_vjp(np.asarray(cotangents), np.asarray(k), np.asarray(z),
                                               np.asarray(src), np.asarray(idxs), num_threads)
    return ct_k, ct_z, np.array(ct_src)

# matmul

matmul_p = Primitive('matmul')
matmul_jvp_p = Primitive('matmul_jvp')
matmul_vjp_p = Primitive('matmul_vjp')

def matmul(vecs, mats, vidxs=None, midxs=None, num_threads=1):
    vsize, msize = vecs.size // vecs.shape[-1], mats.size // (mats.shape[-2] * mats.shape[-1])
    vidxs, midxs = geometry.matmul_indices(vsize, msize, vidxs, midxs)
    return matmul_p.bind(vecs, mats, vidxs, midxs, num_threads)

@matmul_p.def_impl
def matmul_impl(vecs, mats, vidxs, midxs, num_threads):
    return geometry.matmul(np.asarray(vecs), np.asarray(mats), np.asarray(vidxs), np.asarray(midxs),
                           num_threads)

def matmul_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    vecs, mats, vidxs, midxs, num_threads = primals
    v_dot, m_dot, _, _, _ = tangents

    out = matmul_p.bind(vecs, mats, vidxs, midxs, num_threads)
    out_dot = matmul_jvp_p.bind(vecs, mats, vidxs, midxs, num_threads, make_zero(v_dot, vecs),
                                make_zero(m_dot, mats))
    return out, out_dot

ad.primitive_jvps[matmul_p] = matmul_jvp

@matmul_jvp_p.def_abstract_eval
def matmul_jvp_abstract_eval(v_eval, m_eval, vi_eval, mi_eval, nt_eval, v_dot, m_dot):
    return ShapedArray(vi_eval.shape + (3,), dtype=v_eval.dtype)

def matmul_jvp_transpose(ct, vecs, mats, vidxs, midxs, num_threads, v_dot, m_dot):
    assert ad.is_undefined_primal(v_dot) or ad.is_undefined_primal(m_dot)
    ct_v, ct_m = matmul_vjp_p.bind(ct, vecs, mats, vidxs, midxs, num_threads)
    return None, None, None, None, None, ct_v, ct_m

ad.primitive_transposes[matmul_jvp_p] = matmul_jvp_transpose

@matmul_vjp_p.def_impl
def matmul_vjp_impl(cotangents, vecs, mats, vidxs, midxs, num_threads):
    return geometry.matmul_vjp(np.asarray(cotangents), np.asarray(vecs), np.asarray(mats),
                               np.asarray(vidxs), np.asarray(midxs), num_threads)

# source_lines

source_lines_p = Primitive('source_lines')
source_lines_jvp_p = Primitive('source_lines_jvp')
source_lines_vjp_p = Primitive('source_lines_vjp')

def source_lines(q, kmin, kmax, num_threads=1):
    return source_lines_p.bind(q, jnp.asarray(kmin), jnp.asarray(kmax), num_threads)

@source_lines_p.def_impl
def source_lines_impl(q, kmin, kmax, num_threads):
    return geometry.source_lines(np.asarray(q), np.asarray(kmin), np.asarray(kmax), num_threads)

def source_lines_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    q, kmin, kmax, num_threads = primals
    q_dot, kmin_dot, kmax_dot, _ = tangents

    kin = source_lines_p.bind(q, kmin, kmax, num_threads)
    kin_dot = source_lines_jvp_p.bind(kin, q, kmin, kmax, num_threads, make_zero(q_dot, q),
                                      make_zero(kmin_dot, kmin), make_zero(kmax_dot, kmax))

    return kin, kin_dot

ad.primitive_jvps[source_lines_p] = source_lines_jvp

@source_lines_jvp_p.def_abstract_eval
def source_lines_jvp_abstract_eval(k_eval, q_eval, k0_eval, k1_eval, nt_eval, q_dot, k0_dot, k1_dot):
    return ShapedArray(k_eval.shape, k_eval.dtype)

def source_lines_jvp_transpose(ct, kin, q, kmin, kmax, num_threads, q_dot, kmin_dot, kmax_dot):
    assert ad.is_undefined_primal(q_dot) or \
           ad.is_undefined_primal(kmin_dot) or \
           ad.is_undefined_primal(kmax_dot)
    ct_q, ct_kmin, ct_kmax = source_lines_vjp_p.bind(ct, kin, q, kmin, kmax, num_threads)
    return None, None, None, None, None, ct_q, ct_kmin, ct_kmax

ad.primitive_transposes[source_lines_jvp_p] = source_lines_jvp_transpose

@source_lines_vjp_p.def_impl
def source_lines_vjp_impl(cotangents, kin, q, kmin, kmax, num_threads):
    ct_q, ct_kmin, ct_kmax = geometry.source_lines_vjp(np.asarray(cotangents), np.asarray(kin), np.asarray(q),
                                                       np.asarray(kmin), np.asarray(kmax), num_threads)
    return ct_q, jnp.asarray(ct_kmin), jnp.asarray(ct_kmax)

# draw_line

draw_line_p = Primitive('draw_line')
draw_line_jvp_p = Primitive('draw_line_jvp')
draw_line_vjp_p = Primitive('draw_line_vjp')

KERNEL_NAMES = {0: 'biweight',
                1: 'gaussian',
                2: 'parabolic',
                3: 'rectangular',
                4: 'triangular'}
KERNEL_NUMS = dict(zip(KERNEL_NAMES.values(), KERNEL_NAMES.keys()))

def draw_line(out, lines, idxs=None, max_val=1.0, kernel='rectangular', num_threads=1):
    idxs = geometry.fill_indices(np.prod(out.shape[:-2], dtype=int),
                                 np.prod(lines.shape[:-1], dtype=int), idxs)
    return draw_line_p.bind(out, lines, idxs, max_val, KERNEL_NUMS[kernel], num_threads)

@draw_line_p.def_impl
def draw_line_impl(out, lines, idxs, max_val, kernel, num_threads):
    return image_proc.draw_line(np.asarray(out), np.asarray(lines), np.asarray(idxs),
                                max_val, KERNEL_NAMES[kernel], num_threads)

def draw_line_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    out, lines, idxs, max_val, kernel, num_threads = primals
    _, lines_dot, _, _, _, _ = tangents

    out = draw_line_p.bind(out, lines, idxs, max_val, kernel, num_threads)
    out_dot = draw_line_jvp_p.bind(out, lines, idxs, max_val, kernel, num_threads,
                                   make_zero(lines_dot, lines))
    return out, out_dot

ad.primitive_jvps[draw_line_p] = draw_line_jvp

@draw_line_jvp_p.def_abstract_eval
def draw_line_jvp_abstract_eval(o_eval, l_eval, i_eval, mv_eval, k_eval, nt_eval, l_dot):
    return ShapedArray(o_eval.shape, o_eval.dtype)

def draw_line_jvp_transpose(ct, out, lines, idxs, max_val, kernel, num_threads, lines_dot):
    assert ad.is_undefined_primal(lines_dot)
    ct_lines = draw_line_vjp_p.bind(ct, lines, idxs, max_val, kernel, num_threads)
    return None, None, None, None, None, None, ct_lines

ad.primitive_transposes[draw_line_jvp_p] = draw_line_jvp_transpose

@draw_line_vjp_p.def_impl
def draw_line_vjp_impl(cotangents, lines, idxs, max_val, kernel, num_threads):
    return image_proc.draw_line_vjp(np.asarray(cotangents), np.asarray(lines), np.asarray(idxs),
                                    max_val, KERNEL_NAMES[kernel], num_threads)
