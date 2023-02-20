#ifndef INCLUDE_H
#define INCLUDE_H

#define _GNU_SOURCE 1

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
#include <pthread.h>

#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#define WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#else
#define NOINLINE
#define WARN_UNUSED_RESULT
#endif

/** pi **/
#ifndef M_PI
<<<<<<< HEAD
#define M_PI    3.14159265358979323846f
=======
#define M_PI        3.14159265358979323846f
>>>>>>> dev-dataclass
#endif

/** pi / 2 **/
#ifndef M_PI_2
<<<<<<< HEAD
#define M_PI_2  1.57079632679489661923f
=======
#define M_PI_2      1.57079632679489661923f
>>>>>>> dev-dataclass
#endif

/** ln(10) **/
#ifndef M_LN10
<<<<<<< HEAD
#define M_LN10 2.30258509299404568402f
=======
#define M_LN10      2.30258509299404568402f
>>>>>>> dev-dataclass
#endif /* !M_LN10 */

/** 3/2 pi */
#define M_3_2_PI    4.71238898038f

/** 2 pi */
#define M_2__PI     6.28318530718f

/** 1 / sqrt(2 pi) **/
#define M_1_SQRT2PI 0.3989422804014327

#define SQ(x)   ((x) * (x))

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define CLIP(c, a, b) \
    do {(c) = ((c) > (a)) ? (c) : (a); (c) = ((c) < (b)) ? (c) : (b); } while (0)

#define MALLOC(type, num) \
    ((type *)malloc((num) * sizeof(type)))

#define REALLOC(buf, type, num) \
    ((type *)realloc((buf), (num) * sizeof(type)))

/* free() doesn't change ptr, it still points to (now invalid) location */
#define DEALLOC(ptr) \
    do { free(ptr); (ptr) = NULL; } while (0)

#define SWAP(a, b, type) \
    do { type tmp_ = (a); (a) = (b); (b) = tmp_; } while (0)

#define SWAP_BUF(a, b, size) \
    do { unsigned char buf[(size)]; memmove(buf, (a), (size)); memmove((a), (b), (size)); memmove((b), buf, (size)); } while(0)

#define ERROR(msg) \
    (fprintf(stderr, "C Error: %s\n", msg))

#define LSD_ERROR(msg) \
    (fprintf(stderr, "LSD Error: %s\n", msg))

#endif