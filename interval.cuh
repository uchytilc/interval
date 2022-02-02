#ifndef INTERVAL_H
#define INTERVAL_H

#include <math_constants.h>
#include "vector_extension.h"

template<typename T>
struct Interval{
    inline Interval() : lo(0), hi(0) {}
    inline explicit Interval(T x) : lo(x), hi(x) {}
    inline Interval(T lo, T hi) : lo(lo), hi(hi) {}

#ifdef __CUDACC__
    __device__ inline T width() const {
        return __sub_ru(hi, lo);
    }
    __device__ inline T mid() const {
        return __div_ru(hi, static_cast<T>(2.0)) + __div_ru(lo, static_cast<T>(2.0));
    }
    __device__ inline T rad() const {
        const T m = mid();
        return max(__sub_ru(m, lo), __sub_ru(hi, m));
    }
#endif
    T lo;
    T hi;
};

typedef Interval<float> Intervalf;
typedef Interval<double> Intervald;

#ifdef __CUDACC__

inline float integral_as_floating(int f){
    return __int_as_float(f);
}

inline double integral_as_floating(long long f){
    return __longlong_as_double(f);
}

//https://stackoverflow.com/questions/2757816/class-template-specializations-with-shared-functionality

template<typename T>
struct IntervalConstants{
};

template<>
struct IntervalConstants<float>{
    //no rounding float version of scalar functions is defined. The value will be cast to double and then rounded after the operation
    static float __val_rd(double f) { return __double2float_rd(f); };
    static float __val_ru(double f) { return __double2float_ru(f); };
    static float __integral_as_floating(int f) { return __int_as_float(f); }

    static const int nan;
    static const int inf;
    static const int pi_half_lo;
    static const int pi_half;
    static const int pi_half_hi;
    static const int pi_lo;
    static const float pi;
    static const int pi_hi;
    static const int pi2_lo;
    static const int pi2;
    static const int pi2_hi;
};

const int IntervalConstants<float>::nan = 0x7fffffff;
const int IntervalConstants<float>::inf = 0x7f800000;
const int IntervalConstants<float>::pi_half_lo = 0x3fc90fda;
const int IntervalConstants<float>::pi_half    = 0x3fc90fdb;
const int IntervalConstants<float>::pi_half_hi = 0x3fc90fdc;
const int IntervalConstants<float>::pi_lo = 0x40490fda;
const float IntervalConstants<float>::pi  = CUDART_PI_F;
const int IntervalConstants<float>::pi_hi = 0x40490fdc;
const int IntervalConstants<float>::pi2_lo = 0x40c90fda;
const int IntervalConstants<float>::pi2    = 0x40c90fdb;
const int IntervalConstants<float>::pi2_hi = 0x40c90fdc;

template<>
struct IntervalConstants<double>{
    //for double there is no 128 bit version of the op. The resulting value is always rounded using `nextafter`
    static double __val_rd(double f) { return nextafter(f, -integral_as_floating(inf)); };
    static double __val_ru(double f) { return nextafter(f,  integral_as_floating(inf)); };
    static double __integral_as_floating(long long f) { return __longlong_as_double(f); }

    static const long long nan;
    static const long long inf;

    static const long long pi_half_lo;
    static const long long pi_half;
    static const long long pi_half_hi;
    static const long long pi_lo;
    static const double pi;
    static const long long pi_hi;
    static const long long pi2_lo;
    static const long long pi2;
    static const long long pi2_hi;
};

const long long IntervalConstants<double>::nan = 0xfff8000000000000;
const long long IntervalConstants<double>::inf = 0x7ff0000000000000;
const long long IntervalConstants<double>::pi_half_lo = 0x3ff921fb54442d17;
const long long IntervalConstants<double>::pi_half    = 0x3ff921fb54442d18;
const long long IntervalConstants<double>::pi_half_hi = 0x3ff921fb54442d19;
const long long IntervalConstants<double>::pi_lo = 0x400921fb54442d17;
const double IntervalConstants<double>::pi       = CUDART_PI;
const long long IntervalConstants<double>::pi_hi = 0x400921fb54442d19;
const long long IntervalConstants<double>::pi2_lo = 0x401921fb54442d17;
const long long IntervalConstants<double>::pi2    = 0x401921fb54442d18;
const long long IntervalConstants<double>::pi2_hi = 0x401921fb54442d19;

#define NAN_INTERVAL integral_as_floating(IntervalConstants<T>::nan) //IntervalConstants<T>::__integral_as_floating(IntervalConstants<T>::nan)
#define INF_INTERVAL integral_as_floating(IntervalConstants<T>::inf)
#define PI_HALF_LO_INTERVAL integral_as_floating(IntervalConstants<T>::pi_half_lo)
#define PI_HALF_INTERVAL integral_as_floating(IntervalConstants<T>::pi_half)
#define PI_HALF_HI_INTERVAL integral_as_floating(IntervalConstants<T>::pi_half_hi)
#define PI_LO_INTERVAL integral_as_floating(IntervalConstants<T>::pi_lo)
#define PI_INTERVAL IntervalConstants<T>::pi_lo
#define PI_HI_INTERVAL integral_as_floating(IntervalConstants<T>::pi_hi)
#define PI2_LO_INTERVAL integral_as_floating(IntervalConstants<T>::pi2_lo)
#define PI2_INTERVAL integral_as_floating(IntervalConstants<T>::pi2)
#define PI2_HI_INTERVAL integral_as_floating(IntervalConstants<T>::pi2_hi)

#define __VAL_RD IntervalConstants<T>::__val_rd
#define __VAL_RU IntervalConstants<T>::__val_ru


// template<typename T>
// inline bool is_empty(const Interval<T>& f){
//     // return isnan(f.lo) || isnan(f.hi);
//     return !(f.lo <= f.hi);
// }


inline float max(float f, float g) {
    return fmaxf(f, g);
}
inline double max(double f, double g) {
    return fmax(f, g);
}

inline float min(float f, float g) {
    return fminf(f, g);
}
inline double min(double f, double g) {
    return fmin(f, g);
}

inline float round(float f) {
    return roundf(f);
}

inline float floor(float f) {
    return floorf(f);
}

inline float ceil(float f) {
    return ceilf(f);
}



inline float  __add_rd(float f, float g) {
    return __fadd_rd(f, g);
}
inline float  __add_ru(float f, float g) {
    return __fadd_ru(f, g);
}

inline double __add_rd(double f, double g) {
    return __dadd_rd(f, g);
}
inline double __add_ru(double f, double g) {
    return __dadd_ru(f, g);
}


inline float  __sub_rd(float f, float g) {
    return __fsub_rd(f, g);
}
inline float  __sub_ru(float f, float g) {
    return __fsub_rd(f, g);
}

inline double __sub_rd(double f, double g) {
    return __dsub_rd(f, g);
}
inline double __sub_ru(double f, double g) {
    return __dsub_ru(f, g);
}


inline float  __mul_rd(float f, float g) {
    return __fmul_rd(f, g);
}
inline float  __mul_ru(float f, float g) {
    return __fmul_ru(f, g);
}

inline double __mul_rd(double f, double g) {
    return __dmul_rd(f, g);
}
inline double __mul_ru(double f, double g) {
    return __dmul_ru(f, g);
}


inline float  __div_rd(float f, float g) {
    return __fdiv_rd(f, g);
}
inline float  __div_ru(float f, float g) {
    return __fdiv_ru(f, g);
}

inline double __div_rd(double f, double g) {
    return __ddiv_rd(f, g);
}
inline double __div_ru(double f, double g) {
    return __ddiv_ru(f, g);
}


inline float  __sqrt_rd(float f) {
    return __fsqrt_rd(f);
}
inline float  __sqrt_ru(float f) {
    return __fsqrt_ru(f);
}

inline double __sqrt_rd(double f) {
    return __dsqrt_rd(f);
}
inline double __sqrt_ru(double f) {
    return __dsqrt_ru(f);
}


//////////
//BINARY//
//////////


template<typename T>
__device__ inline bool operator==(const Interval<T>& f, const Interval<T>& g) {
  return (f.lo == g.lo) && (f.hi == g.hi);
}

template<typename T>
__device__ inline bool operator==(const Interval<T>& f, const T& g) {
  return (f.lo == g) && (f.hi == g);
}

template<typename T>
__device__ inline bool operator==(const T& f, const Interval<T>& g) {
  return (f == g.lo) && (f == g.hi);
}


template<typename T>
__device__ inline bool operator!=(const Interval<T>& f, const Interval<T>& g) {
  return (f.lo != g.lo) && (f.hi != g.hi);
}

template<typename T>
__device__ inline bool operator!=(const Interval<T>& f, const T& g) {
  return (f.lo != g) && (f.hi != g);
}

template<typename T>
__device__ inline bool operator!=(const T& f, const Interval<T>& g) {
  return (f != g.lo) && (f != g.hi);
}


template<typename T>
__device__ inline bool operator>=(const Interval<T>& f, const Interval<T>& g) {
  return f.lo >= g.hi;
}

template<typename T>
__device__ inline bool operator>=(const Interval<T>& f, const T& g) {
  return f.lo >= g;
}

template<typename T>
__device__ inline bool operator>=(const T& f, const Interval<T>& g) {
  return f >= g.hi;
}


template<typename T>
__device__ inline bool operator>(const Interval<T>& f, const Interval<T>& g) {
  return f.lo > g.hi;
}

template<typename T>
__device__ inline bool operator>(const Interval<T>& f, const T& g) {
  return f.lo > g;
}

template<typename T>
__device__ inline bool operator>(const T& f, const Interval<T>& g) {
  return f > g.hi;
}


template<typename T>
__device__ inline bool operator<=(const Interval<T>& f, const Interval<T>& g) {
  return f.hi <= g.lo;
}

template<typename T>
__device__ inline bool operator<=(const Interval<T>& f, const T& g) {
  return f.hi <= g;
}

template<typename T>
__device__ inline bool operator<=(const T& f, const Interval<T>& g) {
  return f <= g.lo;
}


template<typename T>
__device__ inline bool operator<(const Interval<T>& f, const Interval<T>& g) {
  return f.hi < g.lo;
}

template<typename T>
__device__ inline bool operator<(const Interval<T>& f, const T& g) {
  return f.hi < g;
}

template<typename T>
__device__ inline bool operator<(const T& f, const Interval<T>& g) {
  return f < g.lo;
}


template<typename T>
__device__ inline Interval<T> operator+(const Interval<T>& f, const Interval<T>& g) {
    // if (is_empty(f) || is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__add_rd(f.lo, g.lo), __add_ru(f.hi, g.hi));
}
template<typename T>
__device__ inline Interval<T> operator+(const Interval<T>& f, const T& g) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__add_rd(f.lo, g), __add_ru(f.hi, g));
}
template<typename T>
__device__ inline Interval<T> operator+(const T& f, const Interval<T>& g) {
    return g + f;
}

template<typename T>
__device__ inline Interval<T> operator-(const Interval<T>& f, const Interval<T>& g) {
    // if (is_empty(f) || is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__sub_rd(f.lo, g.hi), __sub_ru(f.hi, g.lo));
}
template<typename T>
__device__ inline Interval<T> operator-(const Interval<T>& f, const T& g) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__sub_rd(f.lo, g), __sub_ru(f.hi, g));
}
template<typename T>
__device__ inline Interval<T> operator-(const T& f, const Interval<T>& g) {
    // if (is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__sub_rd(f, g.hi), __sub_ru(f, g.lo));
}


template<typename T>
__device__ inline Interval<T> operator*(const Interval<T>& f, const Interval<T>& g) {
    // if (is_empty(f) || is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(min(__mul_rd(f.lo, g.lo),
                       min(__mul_rd(f.lo, g.hi),
                       min(__mul_rd(f.hi, g.lo),
                           __mul_rd(f.hi, g.hi)))),
                       max(__mul_ru(f.lo, g.lo),
                       max(__mul_ru(f.lo, g.hi),
                       max(__mul_ru(f.hi, g.lo),
                           __mul_ru(f.hi, g.hi)))));
}
template<typename T>
__device__ inline Interval<T> operator*(const Interval<T>& f, const T& g) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    T lo = __mul_rd(g, f.lo);
    T hi = __mul_ru(g, f.hi);
    if (hi < lo){
        const T temp = lo;
        lo = hi;
        hi = temp;
    }
    return Interval<T>(lo, hi);
}
template<typename T>
__device__ inline Interval<T> operator*(const T& f, const Interval<T>& g) {
    return g*f;
}


template<typename T>
__device__ inline Interval<T> operator/(const Interval<T>& f, const Interval<T>& g) {
    // if (is_empty(f) || is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (g.lo <= static_cast<T>(0.0) && static_cast<T>(0.0) <= g.hi) {
        return Interval<T>(-INF_INTERVAL, INF_INTERVAL);
    } else{
        return Interval<T>(min(__div_rd(f.lo, g.lo),
                           min(__div_rd(f.lo, g.hi),
                           min(__div_rd(f.hi, g.lo),
                               __div_rd(f.hi, g.hi)))),
                           max(__div_ru(f.lo, g.lo),
                           max(__div_ru(f.lo, g.hi),
                           max(__div_ru(f.hi, g.lo),
                               __div_ru(f.hi, g.hi)))));
    }
}
template<typename T>
__device__ inline Interval<T> operator/(const Interval<T>& f, const T& g) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (g < static_cast<T>(0.0)) {
        return Interval<T>(__div_rd(f.hi, g), __div_ru(f.lo, g));
    } else if (g > static_cast<T>(0.0)) {
        return Interval<T>(__div_rd(f.lo, g), __div_ru(f.hi, g));
    } else {
        return Interval<T>(-INF_INTERVAL, INF_INTERVAL);
    }
}
template<typename T>
__device__ inline Interval<T> operator/(const T& f, const Interval<T>& g) {
    return Interval<T>(f)/g;
}


template<typename T>
__device__ inline Interval<T> min(const Interval<T>& f, const Interval<T>& g) {
    // if (is_empty(f) || is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(min(f.lo, g.lo), min(f.hi, g.hi));
}
template<typename T>
__device__ inline Interval<T> min(const Interval<T>& f, const T& g) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(min(f.lo, g), min(f.hi, g));
}
template<typename T>
__device__ inline Interval<T> min(const T& f, const Interval<T>& g) {
    return min(g, f);
}


template<typename T>
__device__ inline Interval<T> max(const Interval<T>& f, const Interval<T>& g) {
    // if (is_empty(f) || is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(max(f.lo, g.lo), max(f.hi, g.hi));
}
template<typename T>
__device__ inline Interval<T> max(const Interval<T>& f, const T& g) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(max(f.lo, g), max(f.hi, g));
}
template<typename T>
__device__ inline Interval<T> max(const T& f, const Interval<T>& g) {
    return max(g, f);
}

template<typename T>
__device__ inline Interval<T> fmod(const Interval<T>& f, const T& g) {
    g = abs(g);

    T lo = fmod(f.lo, g);
    T hi = fmod(f.hi, g);

    if (f.hi < static_cast<T>(0.0)){
        return -fmod(-f, g);
    }
    else if (f.lo < static_cast<T>(0.0)){
        return Interval<T>(max(-g, f.lo), min(g, f.hi));
    }
    else if (width < g && lo <= hi){
        return Interval<T>(min(lo, hi), max(lo, hi));
    }
    else{
        return Interval<T>(0, g);
    }
}

// template<typename T>
// __device__ inline Interval<T> modf(const Interval<T>& f, const Interval<T>& g) {
//     // def mod2([a,b], [m,n]):
//     //     // (1): empty interval
//     //     if a > b || m > n:
//     //         return []
//     //     // (2): compute modulo with positive interval and negate
//     //     else if b < 0:
//     //         return -mod2([-b,-a], [m,n])
//     //     // (3): split into negative and non-negative interval, compute, and join 
//     //     else if a < 0:
//     //         return mod2([a,-1], [m,n]) u mod2([0,b], [m,n])
//     //     // (4): use the simpler function from before
//     //     else if m == n:
//     //         return mod1([a,b], m)
//     //     // (5): use only non-negative m and n
//     //     else if n <= 0:
//     //         return mod2([a,b], [-n,-m])
//     //     // (6): similar to (5), make modulus non-negative
//     //     else if m <= 0:
//     //         return mod2([a,b], [1, max(-m,n)])
//     //     // (7): compare to (4) in mod1, check b-a < |modulus|
//     //     else if b-a >= n:
//     //         return [0,n-1]
//     //     // (8): similar to (7), split interval, compute, and join
//     //     else if b-a >= m:
//     //         return [0, b-a-1] u mod2([a,b], [b-a+1,n])
//     //     // (9): modulo has no effect
//     //     else if m > b:
//     //         return [a,b]
//     //     // (10): there is some overlapping of [a,b] and [n,m]
//     //     else if n > b:
//     //         return [0,b]
//     //     // (11): either compute all possibilities and join, or be imprecise
//     //     else:
//     //         return [0,n-1] // imprecise
// }


/////////
//UNARY//
/////////


template<typename T>
__device__ inline Interval<T> operator-(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(-f.hi, -f.lo);
}


template<typename T>
__device__ inline Interval<T> abs(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.lo >= static_cast<T>(0.0)) {
        return f;
    } else if (f.hi < static_cast<T>(0.0)) {
        return -f;
    } else {
        return Interval<T>(static_cast<T>(0.0), max(-f.lo, f.hi));
    }
}


template<typename T>
__device__ inline Interval<T> round(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(round(f.lo), round(f.hi));
}


template<typename T>
__device__ inline Interval<T> floor(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(floor(f.lo), floor(f.hi));
}


template<typename T>
__device__ inline Interval<T> ceil(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(ceil(f.lo), ceil(f.hi));
}


template<typename T>
__device__ inline Interval<T> sqrt(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(0.0)) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else if (f.lo <= static_cast<T>(0.0)) {
        return Interval<T>(static_cast<T>(0.0), __sqrt_ru(f.hi));
    } else {
        return Interval<T>(__sqrt_rd(f.lo), __sqrt_ru(f.hi));
    }
}


template<typename T>
__device__ inline Interval<T> square(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(0.0)) {
        return Interval<T>(__mul_rd(f.hi, f.hi), __mul_rd(f.lo, f.lo));
    } else if (f.lo > static_cast<T>(0.0)) {
        return Interval<T>(__mul_rd(f.lo, f.lo), __mul_rd(f.hi, f.hi));
    } else if (-f.lo > f.hi) {
        return Interval<T>(static_cast<T>(0.0), __mul_rd(f.lo, f.lo));
    } else {
        return Interval<T>(static_cast<T>(0.0), __mul_rd(f.hi, f.hi));
    }
}
template<typename T>
inline T square(const T& f) {
    return f*f;
}



template<typename T>
__device__ inline Interval<T> exp(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__VAL_RD(exp(static_cast<double>(f.lo))),
                       __VAL_RU(exp(static_cast<double>(f.hi))));
}
template<typename T>
__device__ inline Interval<T> exp10(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__VAL_RD(exp10(static_cast<double>(f.lo))),
                       __VAL_RU(exp10(static_cast<double>(f.hi))));
}
template<typename T>
__device__ inline Interval<T> exp2(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__VAL_RD(exp2(static_cast<double>(f.lo))),
                       __VAL_RU(exp2(static_cast<double>(f.hi))));
}

// template<typename T>
// __device__ inline Interval<T> expm1(const Interval<T>& f) {
//   return Interval<T>(__VAL_RD(expm1(static_cast<double>(f.lo))),
//            __VAL_RU(expm1(static_cast<double>(f.hi))));
// }

template<typename T>
__device__ inline Interval<T> log(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(0.0)) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else if (f.lo <= static_cast<T>(0.0)) {
        return Interval<T>(static_cast<T>(0.0),
                           __VAL_RU(log(static_cast<double>(f.hi))));
    } else {
        return Interval<T>(__VAL_RD(log(static_cast<double>(f.lo))),
                           __VAL_RU(log(static_cast<double>(f.hi))));
    }
}
template<typename T>
__device__ inline Interval<T> log10(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(0.0)) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else if (f.lo <= static_cast<T>(0.0)) {
        return Interval<T>(static_cast<T>(0.0),
                           __VAL_RU(log10(static_cast<double>(f.hi))));
    } else {
        return Interval<T>(__VAL_RD(log10(static_cast<double>(f.lo))),
                           __VAL_RU(log10(static_cast<double>(f.hi))));
    }
}
template<typename T>
__device__ inline Interval<T> log2(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(0.0)) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else if (f.lo <= static_cast<T>(0.0)) {
        return Interval<T>(static_cast<T>(0.0),
                           __VAL_RU(log2(static_cast<double>(f.hi))));
    } else {
        return Interval<T>(__VAL_RD(log2(static_cast<double>(f.lo))),
                           __VAL_RU(log2(static_cast<double>(f.hi))));
    }
}
// template<typename T>
// __device__ inline Interval<T> log1p(const Interval<T>& f) {
//   if (f.hi < 0.0) {
//     return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
//   } else if (f.lo <= 0.0) {
//     return Interval<T>(static_cast<T>(0.0),
//              __VAL_RU(log1p(static_cast<double>(f.hi))));
//   } else {
//     return Interval<T>(__VAL_RD(log1p(static_cast<double>(f.lo))),
//              __VAL_RU(log1p(static_cast<double>(f.hi))));
//   }
// }

template<typename T>
__device__ inline Interval<T> trig_fmod(const Interval<T>& f, const Interval<T>& g) {
    // if (is_empty(f) || is_empty(g)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    //https://github.com/mauriciopoppe/interval-arithmetic/blob/fce59e39f08a18bedbd7ff1d8a2ae2bf39ad9039/src/operations/algebra.ts
    // if (den == static_cast<T>(0.0)){
    //     return Interval<T>(static_cast<T>(0.0), static_cast<T>(0.0));
    // }
    const T den = (f.lo < static_cast<T>(0.0)) ? g.lo : g.hi;
    T n =  f.lo / den;
    n = (n < static_cast<T>(0.0)) ? ceil(n) : floor(n);
    return f - g*Interval<T>(n);
}

template<typename T>
__device__ inline Interval<T> shift_negative_interval(const Interval<T>& f) {
    if (f.lo < static_cast<T>(0.0)){
        if (f.lo == -INF_INTERVAL){
            return Interval<T>(static_cast<T>(0.0), INF_INTERVAL);
        }
        else{
            const shift = ceil(-f.lo / PI2_LO_INTERVAL) * PI2_LO_INTERVAL;
            return Interval<T>(f.lo + shift,
                               f.hi + shift);
        }
    }
    return f;
}

template<typename T>
__device__ inline Interval<T> cos(const Interval<T>& f) {
    //https://github.com/mauriciopoppe/interval-arithmetic/blob/bc9e779/src/operations/trigonometric.ts#L240

    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }

    static const Interval<T> pi = Interval<T>(PI_LO_INTERVAL, PI_HI_INTERVAL);
    static const Interval<T> pi2 = Interval<T>(PI2_LO_INTERVAL, PI2_HI_INTERVAL);

    const Interval<T> g = trig_fmod(shift_negative_interval(f), pi2);

    if (g.width() >= pi2.lo){
        return Interval<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
    }

    if (g.lo >= pi.hi) {
        return -cos(g - pi);
    }

    const T cos_lo = __VAL_RD(cos(static_cast<double>(g.lo)));
    const T cos_hi = __VAL_RU(cos(static_cast<double>(g.hi)));

    if (g.hi <= pi.lo){
        return Interval<T>(cos_hi, cos_lo);
    }
    else if (g.hi <= pi2.lo){
        return Interval<T>(static_cast<T>(-1.0), max(cos_lo, cos_hi));
    }
    else{
        return Interval<T>(static_cast<T>(-1.0), static_cast<T>(1.0));
    }
}

template<typename T>
__device__ inline Interval<T> sin(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }

  return cos(f - Interval<T>(PI_HALF_LO_INTERVAL, PI_HALF_HI_INTERVAL));
}

template<typename T>
__device__ inline Interval<T> tan(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }

    static const Interval<T> pi = Interval<T>(PI_LO_INTERVAL, PI_HI_INTERVAL);

    Interval<T> g = trig_fmod(shift_negative_interval(f), pi);
    if (g.lo >= PI_HALF_LO_INTERVAL) {
        g -= pi;
    }

    if (g.lo <= -PI_HALF_LO_INTERVAL || g.hi >= PI_HALF_LO_INTERVAL) {
        return Interval<T>(-INF_INTERVAL, INF_INTERVAL);
    }
    return new Interval(__VAL_RD(tan(static_cast<double>(g.lo))),
                        __VAL_RU(tan(static_cast<double>(g.hi))))
}


template<typename T>
__device__ inline Interval<T> sinh(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__VAL_RD(sinh(static_cast<double>(f.lo))),
                       __VAL_RU(sinh(static_cast<double>(f.hi))));
}
template<typename T>
__device__ inline Interval<T> cosh(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    T cosh_lo = cosh(static_cast<double>(f.lo));
    T cosh_hi = cosh(static_cast<double>(f.hi));

    if (f.hi < static_cast<T>(0.0)){
        return Interval<T>(__VAL_RD(cosh_hi), __VAL_RU(cosh_lo));
    } else if (f.lo > static_cast<T>(0.0)){
        return Interval<T>(__VAL_RD(cosh_lo), __VAL_RU(cosh_hi));
    } else{
        return Interval<T>(static_cast<T>(1.0), max(__VAL_RD(cosh_lo), __VAL_RU(cosh_hi)));
    }
}
template<typename T>
__device__ inline Interval<T> tanh(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__VAL_RD(tanh(static_cast<double>(f.lo))),
                       __VAL_RU(tanh(static_cast<double>(f.hi))));
}


template<typename T>
__device__ inline Interval<T> asin(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(-1.0) || static_cast<T>(1.0) < f.lo) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else {
        return Interval<T>(__VAL_RD(asin(max(static_cast<double>(f.lo), -1.0))),
                           __VAL_RU(asin(min(static_cast<double>(f.hi),  1.0))));
    }
}
template<typename T>
__device__ inline Interval<T> acos(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(-1.0) || static_cast<T>(1.0) < f.lo) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else {
        return Interval<T>(__VAL_RD(acos(max(static_cast<double>(f.lo), -1.0))),
                           __VAL_RU(acos(min(static_cast<double>(f.hi),  1.0))));
    }
}
template<typename T>
__device__ inline Interval<T> atan(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__VAL_RD(atan(static_cast<double>(f.lo))),
                       __VAL_RU(atan(static_cast<double>(f.hi))));
}


template<typename T>
__device__ inline Interval<T> asinh(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    return Interval<T>(__VAL_RD(asinh(static_cast<double>(f.lo))),
                       __VAL_RU(asinh(static_cast<double>(f.hi))));
}
template<typename T>
__device__ inline Interval<T> acosh(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(1.0)) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else {
        return Interval<T>(__VAL_RD(acosh(max(static_cast<double>(f.lo), 1.0))),
                           __VAL_RU(acosh(static_cast<double>(f.hi))));
    }
}
template<typename T>
__device__ inline Interval<T> atanh(const Interval<T>& f) {
    // if (is_empty(f)){ return Interval<T>(NAN_INTERVAL, NAN_INTERVAL); }
    if (f.hi < static_cast<T>(-1.0) || static_cast<T>(1.0) < f.lo) {
        return Interval<T>(NAN_INTERVAL, NAN_INTERVAL);
    } else {
        return Interval<T>(__VAL_RD(atanh(max(static_cast<double>(f.lo), -1.0))),
                           __VAL_RU(atanh(min(static_cast<double>(f.hi),  1.0))));
    }
}

#endif


//////////
//VECTOR//
//////////


template<typename T>
struct Interval2{
    Interval<T> x;
    Interval<T> y;
};

typedef Interval2<float> Intervalf2;
typedef Interval2<double> Intervald2;

template<typename T>
inline Interval2<T> make_interval2(const Interval<T> x, const Interval<T> y){
  Interval2<t> interval;
  interval.x = x;
  interval.y = y;
  return interval;
}

template<typename T>
struct Interval3{
    Interval<T> x;
    Interval<T> y;
    Interval<T> z;
};

typedef Interval3<float> Intervalf3;
typedef Interval3<double> Intervald3;

template<typename T>
inline Interval3<T> make_interval3(const Interval<T> x, const Interval<T> y, const Interval<T> z){
    Interval3<T> interval;
    interval.x = x;
    interval.y = y;
    interval.z = z;
    return interval;
}

template<typename T>
struct Interval4{
    Interval<T> x;
    Interval<T> y;
    Interval<T> z;
    Interval<T> w;
};

typedef Interval4<float> Intervalf4;
typedef Interval4<double> Intervald4;

template<typename T>
inline Interval4<T> make_interval4(const Interval<T> x, const Interval<T> y, const Interval<T> z, const Interval<T> w){
    Interval4<T> interval;
    interval.x = x;
    interval.y = y;
    interval.z = z;
    interval.w = w;
    return interval;
}


#endif
