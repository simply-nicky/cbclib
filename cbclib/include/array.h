#ifndef ARRAY_H
#define ARRAY_H
#include "include.h"

typedef struct array_s
{
    int ndim;
    size_t size;            // Number of elements
    size_t item_size;       // Number of bytes per one element
    const size_t *dims;     // Shape of the array
    size_t *strides;
    void *data;
} array_s;
typedef struct array_s *array;

array new_array(int ndim, const size_t *dims, size_t item_size, void *data);
void free_array(array arr);

#define GET(_arr, _type, _idx) ((_type *)(_arr)->data)[(_idx)]

#define UNRAVEL_INDEX(_coord, _idx, _arr)               \
    do {int _i = *(_idx), _n;                           \
        for (_n = 0; _n < (_arr)->ndim; _n++)           \
        {                                               \
            (_coord)[_n] = _i / (_arr)->strides[_n];    \
            _i -= (_coord)[_n] * (_arr)->strides[_n];   \
        }                                               \
    } while (0)

#define RAVEL_INDEX(_coord, _idx, _arr)                     \
    do {*(_idx) = 0; int _n;                                \
        for (_n = 0; _n < (_arr)->ndim; _n++)               \
            *(_idx) += (_arr)->strides[_n] * (_coord)[_n];  \
    } while (0)

/*---------------------------------------------------------------------------
    struct line:
        npts        - number of points along the axis
        stride      - line stride
                      line[i + 1] = line[i] + line->stride * line->item_size
        item_size   - size of values in the array
        data        - pointer to the first line element
        first       - pointer to the first array element
---------------------------------------------------------------------------*/

typedef struct line_s
{
    size_t npts;
    size_t stride;
    size_t item_size;
    size_t line_size;
    void *data, *first;
} line_s;
typedef struct line_s *line;

line new_line(size_t npts, size_t stride, size_t item_size, void *data);
line init_line(array arr, int axis);

/*---------------------------------------------------------------------------
    Update the pointer to the first element of a line
----------------------------------------------------------------------------*/

#define UPDATE_LINE(_line, _iter)                                               \
do {int _div; _div = (_iter) / (_line)->stride;                                 \
    (_line)->data = (_line)->first + (_line)->line_size * _div +                \
                    ((_iter) - _div * (_line)->stride) * (_line)->item_size;    \
} while (0)

/*----------------------------------------------------------------------------*/
/*--------------------------- Extend line modes ------------------------------*/
/*----------------------------------------------------------------------------*/
/*
    EXTEND_CONSTANT: kkkkkkkk|abcd|kkkkkkkk
    EXTEND_NEAREST:  aaaaaaaa|abcd|dddddddd
    EXTEND_MIRROR:   cbabcdcb|abcd|cbabcdcb
    EXTEND_REFLECT:  abcddcba|abcd|dcbaabcd
    EXTEND_WRAP:     abcdabcd|abcd|abcdabcd
*/
typedef enum
{
    EXTEND_CONSTANT = 0,
    EXTEND_NEAREST = 1,
    EXTEND_MIRROR = 2,
    EXTEND_REFLECT = 3,
    EXTEND_WRAP = 4
} EXTEND_MODE;

void extend_line(void *out, size_t osize, line inp, EXTEND_MODE mode, const void *cval);
int extend_point(void *out, int *coord, array arr, array mask, EXTEND_MODE mode, const void *cval);

/*---------------------------------------------------------------------------
    Portable re-entrant quick sort macro
----------------------------------------------------------------------------*/
#if (defined __APPLE__ || defined __MACH__ || defined __DARWIN__ || defined __FREEBSD__ || defined __BSD__)
    typedef struct sort_r_args
    {
        void *arg;
        int (*compar)(const void *a, const void *b, void *arg);
    } sort_r_args;

    static int compar_swap(void *args, const void *a, const void *b)
    {
        struct sort_r_args *_args = (struct sort_r_args*)args;
        return (_args->compar)(a, b, _args->arg);
    }

    #define POSIX_QSORT_R(_base, _nmemb, _size, _compar, _arg) \
        do {struct sort_r_args _tmp; _tmp.arg = (_arg); _tmp.compar = (_compar); qsort_r((_base), (_nmemb), (_size), &_tmp, compar_swap); } while (0)
#elif (defined __GNUC__ || defined __linux__)
    #define POSIX_QSORT_R(_base, _nmemb, _size, _compar, _arg) \
        do {qsort_r((_base), (_nmemb), (_size), (_compar), (_arg)); } while (0)
#elif (defined _WIN32 || defined _WIN64 || defined __WINDOWS__)
    #define POSIX_QSORT_R(_base, _nmemb, _size, _compar, _arg) \
        do {qsort_s((_base), (_nmemb), (_size), (_compar), (_arg)); } while (0)
#else
    #define POSIX_QSORT_R NULL
#endif

/*---------------------------------------------------------------------------
    Comparing functions
----------------------------------------------------------------------------*/
int compare_double(const void *a, const void *b);
int compare_float(const void *a, const void *b);
int compare_int(const void *a, const void *b);
int compare_uint(const void *a, const void *b);
int compare_ulong(const void *a, const void *b);

int indirect_compare_double(const void *a, const void *b, void *data);
int indirect_compare_float(const void *a, const void *b, void *data);

int indirect_search_double(const void *key, const void *base, void *data);
int indirect_search_float(const void *key, const void *base, void *data);


/*---------------------------------------------------------------------------
    Binary search
----------------------------------------------------------------------------*/
// Array search
typedef enum
{
    SEARCH_LEFT = 0,
    SEARCH_RIGHT = 1
} SEARCH_SIDE;

/* find idx \el [0, npts], so that base[idx - 1] < key <= base[idx] */

size_t searchsorted(const void *key, const void *base, size_t npts, size_t size, SEARCH_SIDE side,
                    int (*compar)(const void *, const void *));

size_t searchsorted_r(const void *key, const void *base, size_t npts, size_t size, SEARCH_SIDE side,
                      int (*compar)(const void *, const void *, void *), void *arg);

/*---------------------------------------------------------------------------
    Wirth select
----------------------------------------------------------------------------*/
void *wirthselect(void *inp, int k, int n, size_t size, int (*compar)(const void *, const void *));
void *wirthselect_r(void *inp, int k, int n, size_t size, int (*compar)(const void *, const void *, void *), void *arg);

#define wirthmedian(a, n, size, compar) wirthselect(a, (((n) & 1) ? ((n) / 2) : (((n) / 2) - 1)), n, size, compar)
#define wirthmedian_r(a, n, size, compar, arg) wirthselect_r(a, (((n) & 1) ? ((n) / 2) : (((n) / 2) - 1)), n, size, compar, arg)

/*---------------------------------------------------------------------------
    Rectangular iterator
----------------------------------------------------------------------------*/

typedef struct rect_iter_s
{
    int ndim;
    size_t size;
    int *coord;
    size_t *strides;
    int index;
} rect_iter_s;
typedef struct rect_iter_s *rect_iter;

void ri_del(rect_iter ri);
rect_iter ri_ini(int ndim, int *pt0, int *pt1);
int ri_end(rect_iter ri);
void ri_inc(rect_iter ri);

#endif