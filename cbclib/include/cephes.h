#ifndef CEPHES_H
#define CEPHES_H
#include "include.h"

/*---------------------------------------------------------------------------
    Gamma and related functions
----------------------------------------------------------------------------*/

static inline double polevl(double x, const double coef[], int N)
{
    double ans;
    int i;
    const double *p;

    p = coef;
    ans = *p++;
    i = N;

    do ans = ans * x + *p++;
    while (--i);

    return ans;
}

static inline double p1evl(double x, const double coef[], int N)
{
    double ans;
    const double *p;
    int i;

    p = coef;
    ans = x + *p++;
    i = N - 1;

    do ans = ans * x + *p++;
    while (--i);

    return ans;
}

static inline double ratevl(double x, const double num[], int M,
                            const double denom[], int N)
{
    int i, dir;
    double y, num_ans, denom_ans;
    double absx = fabs(x);
    const double *p;

	/* Evaluate as a polynomial in 1/x. */
    if (absx > 1) {dir = -1; p = num + M; y = 1 / x;}
    else {dir = 1; p = num; y = x;}

    /* Evaluate the numerator */
    num_ans = *p;
    p += dir;
    for (i = 1; i <= M; i++) {num_ans = num_ans * y + *p; p += dir;}

    /* Evaluate the denominator */
    if (absx > 1) p = denom + N;
    else p = denom;

    denom_ans = *p;
    p += dir;
    for (i = 1; i <= N; i++) {denom_ans = denom_ans * y + *p; p += dir;}

    if (absx > 1) {i = N - M; return pow(x, i) * num_ans / denom_ans;}
    else return num_ans / denom_ans;
}

/*---------------------------------------------------------------------------  
    gamma - Gamma function
    lgam - Logarithm of Gamma function
    igam, igamc - Incomplete Gamma funtion
    igami, igamci - Incomplete Gamma inverse function
    chdtri - Inverse of Chi-squared survival function
----------------------------------------------------------------------------*/

double gamma(double x);
double lgam(double x);
double igam(double a, double x);
double igamc(double a, double x);
double igami(double a, double p);
double igamci(double a, double q);
double chdtri(double df, double y);

#endif