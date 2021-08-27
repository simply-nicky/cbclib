#ifndef INCLUDE_H
#define INCLUDE_H

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

#ifndef M_PI
    #define M_PI 3.14159265358979323846264338327950288
#endif

#define RALLOC(type,num) \
    ((type *)malloc((num)*sizeof(type)))
#define DEALLOC(ptr) \
    do { free(ptr); (ptr)=NULL; } while(0)

#define SWAP(a,b,type) \
    do { type tmp_=(a); (a)=(b); (b)=tmp_; } while(0)
#define SWAP_BUF(a,b,size) \
    do{ unsigned char buf[(size)]; memmove(buf, (a), (size)); memmove((a), (b), (size)); memmove((b), buf, (size)); } while(0)

#define ERROR(msg) \
    (fprintf(stderr, "C Error: %s\n", msg))

#define LSD_ERROR(msg) \
    (fprintf(stderr, "LSD Error: %s\n", msg))

#endif