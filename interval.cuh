#ifndef INTERVAL_H
#define INTERVAL_H

#include <math_constants.h>
#include "vector_extension.cuh"

//TO DO
    //implement tan
    //validate implementation of all operations

#ifdef INTERVAL_PRECISION
#define INTERVAL_PRECISION 32 //32 = FLOAT, 64 = DOUBLE
#endif

//this is almost certainly not the best way to do this but I am lazy and don't want to have to define every function for both double and float
#if PRECISION == 64 // defined(DOUBLE)
    #define T double

    #define PI CUDART_PI
    #define NAN CUDART_NAN
    #define INF CUDART_INF

    #define CALL(op) op
    #define CALL2(op) CONCAT(__double, op)

    //round value during operation
    #define __op_rd(op) __d ## op ## _rd
    #define __op_ru(op) __d ## op ## _ru

    //round value after operation
    #define __val_rd(val) nextafter(val, -INF)
    #define __val_ru(val) nextafter(val,  INF)
#else //if PRECISION == 32 //ifdef FLOAT
    #define T float

    #define PI CUDART_PI_F
    #define NAN CUDART_NAN_F
    #define INF CUDART_INF_F

    #define CALL(op) CONCAT(op, f)
    #define CALL2(op) CONCAT(__float, op)

    //round value during operation
    #define __op_rd(op) __f ## op ## _rd
    #define __op_ru(op) __f ## op ## _ru

    //round value after operation
    #define __val_rd(val) __double2float_rd(val)
    #define __val_ru(val) __double2float_ru(val)
#endif

struct Interval {
    __device__ inline Interval() {}
    __device__ inline Interval(T lo, T hi) : i(MAKEVEC2(T)(lo, hi)) {}
    __device__ inline explicit Interval(T i) : i(MAKEVEC2(T)(i, i)) {}

    __device__ inline T lo() const { return i.x; }
    __device__ inline T hi() const { return i.y; }

    __device__ inline float width() const {
        return __op_ru(sub)(hi(), lo());
    }
    __device__ inline float mid() const {
        return  __op_ru(div)(hi(), 2.0f) + __op_ru(div)(lo(), 2.0f);
    }
    __device__ inline float rad() const {
        const T m = mid();
        return CALL(fmax)(__op_ru(sub)(m, lo()), __op_ru(sub)(hi(), m));
    }

    VEC2(T) i;
};


//////////
//BINARY//
//////////


__device__ inline Interval operator+(const Interval& f, const Interval& g) {
    return Interval(__op_rd(add)(f.lo(), g.lo()), __op_ru(add)(f.hi(), g.hi()));
}
__device__ inline Interval operator+(const Interval& f, const T& g) {
    return Interval(__op_rd(add)(f.lo(), g), __op_ru(add)(f.hi(), g));
}
__device__ inline Interval operator+(const T& f, const Interval& g) {
    return g + f;
}

__device__ inline Interval operator-(const Interval& f, const Interval& g) {
    return Interval(__op_rd(sub)(f.lo(), g.hi()), __op_ru(sub)(f.hi(), g.lo()));
}
__device__ inline Interval operator-(const Interval& f, const T& g) {
    return Interval(__op_rd(sub)(f.lo(), g), __op_ru(sub)(f.hi(), g));
}
__device__ inline Interval operator-(const T& f, const Interval& g) {
    return Interval(__op_rd(sub)(f, g.hi()), __op_ru(sub)(f, g.lo()));
}


__device__ inline Interval operator*(const Interval& f, const Interval& g) {
    return Interval(CALL(fmin)(__op_rd(mul)(f.lo(), g.lo()),
                    CALL(fmin)(__op_rd(mul)(f.lo(), g.hi()),
                    CALL(fmin)(__op_rd(mul)(f.hi(), g.lo()),
                               __op_rd(mul)(f.hi(), g.hi())))),
                    CALL(fmax)(__op_ru(mul)(f.lo(), g.lo()),
                    CALL(fmax)(__op_ru(mul)(f.lo(), g.hi()),
                    CALL(fmax)(__op_ru(mul)(f.hi(), g.lo()),
                               __op_ru(mul)(f.hi(), g.hi())))));
}
__device__ inline Interval operator*(const Interval& f, const T& g) {
    T lo = __op_rd(mul)(g, f.lo());
    T hi = __op_ru(mul)(g, f.hi());
    if (hi < lo){
        const T temp = lo;
        lo = hi;
        hi = temp;
    }
    return Interval(lo, hi);
}
__device__ inline Interval operator*(const T& f, const Interval& g) {
    return g*f;
}

__device__ inline Interval operator/(const Interval& f, const Interval& g) {
    if (g.lo() <= 0.0f && 0.0f <= g.hi()) {
        return Interval(-INF, INF);
    } else{
        return Interval(CALL(fmin)(__op_rd(div)(f.lo(), g.lo()),
                        CALL(fmin)(__op_rd(div)(f.lo(), g.hi()),
                        CALL(fmin)(__op_rd(div)(f.hi(), g.lo()),
                                   __op_rd(div)(f.hi(), g.hi())))),
                        CALL(fmax)(__op_ru(div)(f.lo(), g.lo()),
                        CALL(fmax)(__op_ru(div)(f.lo(), g.hi()),
                        CALL(fmax)(__op_ru(div)(f.hi(), g.lo()),
                                   __op_ru(div)(f.hi(), g.hi())))));
    }
}
__device__ inline Interval operator/(const Interval& f, const T& g) {
    if (g < 0.0f) {
        return Interval(__op_rd(div)(f.hi(), g), __op_ru(div)(f.lo(), g));
    } else if (g > 0.0f) {
        return Interval(__op_rd(div)(f.lo(), g), __op_ru(div)(f.hi(), g));
    } else {
        return Interval(-INF, INF);
    }
}
__device__ inline Interval operator/(const T& f, const Interval& g) {
    return Interval(f)/g;
}

__device__ inline Interval min(const Interval& f, const Interval& g) {
    return Interval(CALL(fmin)(f.lo(), g.lo()), CALL(fmin)(f.hi(), g.hi()));
}
__device__ inline Interval min(const Interval& f, const T& g) {
    return Interval(CALL(fmin)(f.lo(), g), CALL(fmin)(f.hi(), g));
}
__device__ inline Interval min(const T& f, const Interval& g) {
    return min(g, f);
}

__device__ inline Interval max(const Interval& f, const Interval& g) {
    return Interval(CALL(fmax)(f.lo(), g.lo()), CALL(fmax)(f.hi(), g.hi()));
}
__device__ inline Interval max(const Interval& f, const T& g) {
    return Interval(CALL(fmax)(f.lo(), g), CALL(fmax)(f.hi(), g));
}
__device__ inline Interval max(const T& f, const Interval& g) {
    return max(g, f);
}


/////////
//UNARY//
/////////


__device__ inline Interval operator-(const Interval& f) {
    return Interval(-f.hi(), -f.lo());
}

__device__ inline Interval abs(const Interval& f) {
    if (f.lo() >= 0.0f) {
        return f; //Interval(f)
    } else if (f.hi() < 0.0f) {
        return -f;
    } else {
        return Interval(0.0f, CALL(fmax)(-f.lo(), f.hi()));
    }
}
__device__ inline Interval round(const Interval& f) {
    return Interval(CALL(round)(f.lo()), CALL(round)(f.hi()));
}
__device__ inline Interval floor(const Interval& f) {
    return Interval(CALL(floor)(f.lo()), CALL(floor)(f.hi()));
}
__device__ inline Interval ceil(const Interval& f) {
    return Interval(CALL(ceil)(f.lo()), CALL(ceil)(f.hi()));
}

__device__ inline Interval sqrt(const Interval& f) {
    if (f.hi() < 0.f) {
        return Interval(NAN, NAN);
    } else if (f.lo() <= 0.f) {
        return Interval(0.f, __op_ru(sqrt)(f.hi()));
    } else {
        return Interval(__op_rd(sqrt)(f.lo()), __op_ru(sqrt)(f.hi()));
    }
}
__device__ inline Interval square(const Interval& f) {
    if (f.hi() < 0.0f) {
        return Interval(__op_rd(mul)(f.hi(), f.hi()), __op_rd(mul)(f.lo(), f.lo()));
    } else if (f.lo() > 0.0f) {
        return Interval(__op_rd(mul)(f.lo(), f.lo()), __op_rd(mul)(f.hi(), f.hi()));
    } else if (-f.lo() > f.hi()) {
        return Interval(0.0f, __op_rd(mul)(f.lo(), f.lo()));
    } else {
        return Interval(0.0f, __op_rd(mul)(f.hi(), f.hi()));
    }
}
__device__ inline T square(const T& f) {
    return f*f;
}

__device__ inline Interval exp(const Interval& f) {
    return Interval(__val_rd(CALL(exp)(f.lo())), __val_ru(CALL(exp)(f.hi())));
}
__device__ inline Interval exp10(const Interval& f) {
    return Interval(__val_rd(CALL(exp10)(f.lo())), __val_ru(CALL(exp10)(f.hi())));
}
__device__ inline Interval exp2(const Interval& f) {
    return Interval(__val_rd(CALL(exp2)(f.lo())), __val_ru(CALL(exp2)(f.hi())));
}
__device__ inline Interval expm1(const Interval& f) {
    return Interval(__val_rd(CALL(expm1)(f.lo())), __val_ru(CALL(expm1)(f.hi())));
}

__device__ inline Interval log(const Interval& f) {
    if (f.hi() < 0.0f) {
        return Interval(NAN, NAN);
    } else if (f.lo() <= 0.0f) {
        return Interval(0.f, __val_ru(CALL(log)(f.hi())));
    } else {
        return Interval(__val_rd(CALL(log)(f.lo())), __val_ru(CALL(log)(f.hi())));
    }
}
__device__ inline Interval log10(const Interval& f) {
    if (f.hi() < 0.0f) {
        return Interval(NAN, NAN);
    } else if (f.lo() <= 0.0f) {
        return Interval(0.f, __val_ru(CALL(log10)(f.hi())));
    } else {
        return Interval(__val_rd(CALL(log10)(f.lo())), __val_ru(CALL(log)(f.hi())));
    }
}
__device__ inline Interval log2(const Interval& f) {
    if (f.hi() < 0.0f) {
        return Interval(NAN, NAN);
    } else if (f.lo() <= 0.0f) {
        return Interval(0.f, __val_ru(CALL(log2)(f.hi())));
    } else {
        return Interval(__val_rd(CALL(log2)(f.lo())), __val_ru(CALL(log)(f.hi())));
    }
}
__device__ inline Interval log1p(const Interval& f) {
    if (f.hi() < 0.0f) {
        return Interval(NAN, NAN);
    } else if (f.lo() <= 0.0f) {
        return Interval(0.f, __val_ru(CALL(log1p)(f.hi())));
    } else {
        return Interval(__val_rd(CALL(log1p)(f.lo())), __val_ru(CALL(log)(f.hi())));
    }
}


#if INTERVAL_PRECISION == 64
    static const int pi2_lo_int = 0x401921fb54442d18;
    static const T* pi2_lo = ((T*)&pi2_lo_int);

    static const int pi_lo_int = 0x400921fb54442d18;
    static const int pi_hi_int = 0x400921fb54442d19;
    static const T* pi_lo = ((T*)&pi_lo_int);
    static const T* pi_hi = ((T*)&pi_hi_int);
#else
    static const int pi2_lo_int = 0x40c90fda;
    static const T* pi2_lo = ((T*)&pi2_lo_int);

    static const int pi_lo_int = 0x40490fda;
    static const int pi_hi_int = 0x40490fdb;
    static const T* pi_lo = ((T*)&pi_lo_int);
    static const T* pi_hi = ((T*)&pi_hi_int);
#endif

__device__ inline Interval cos(const Interval& f) {
    if (f.hi() - f.lo() > *pi2_lo){
        return Interval(-1.f, 1.f);
    }
    else{
        VEC2(T) bounds;

        T cos_lo = cos(f.lo());
        T cos_hi = cos(f.hi());

        if (f.lo() > 0.f)
            bounds.x = __op_rd(div)(f.lo(), *pi_hi);
        else
            bounds.x = __op_rd(div)(f.lo(), *pi_lo);

        if (f.hi() > 0.f)
            bounds.y = __op_ru(div)(f.hi(), *pi_lo);
        else
            bounds.y = __op_ru(div)(f.hi(), *pi_hi);

        int2 span = make_int2(CALL2(2int_rd)(bounds.x), CALL2(2int_ru)(bounds.y));
        int periods = span.y - span.x;
        int odd = span.x & 1;

        if (periods < 2){
            if (odd){
                return Interval(__val_rd(cos_hi), __val_ru(cos_lo));
            } else{
                return Interval(__val_rd(cos_lo), __val_ru(cos_hi));
            }
        } else if (periods == 2){
            if (odd){
                return Interval(CALL(fmin)(__val_rd(cos_lo), __val_ru(cos_hi)), 1.f);
            } else{
                return Interval(-1.f, CALL(fmax)(__val_rd(cos_lo), __val_ru(cos_hi)));
            }
        //should never happen
        } else{
            return Interval(-1.f, 1.f);
        }
    }
}

__device__ inline Interval sin(const Interval& f) {
    return cos(f - PI/2.0f);
}
// __device__ inline Interval tan(const Interval& f) {
// }

__device__ inline Interval sinh(const Interval& f) {
    return Interval(__val_rd(sinh(f.lo())), __val_ru(sinh(f.hi())));
}
__device__ inline Interval cosh(const Interval& f) {
    T cosh_lo = cosh(f.lo());
    T cosh_hi = cosh(f.hi());

    if (f.hi() < 0.f){
        return Interval(__val_rd(cosh_hi), __val_ru(cosh_lo));
    } else if (f.lo() > 0.f){
        return Interval(__val_rd(cosh_lo), __val_ru(cosh_hi));
    } else{
        return Interval(1.f, CALL(fmax)(__val_rd(cosh_lo), __val_ru(cosh_hi)));
    }
}
__device__ inline Interval tanh(const Interval& f) {
    return Interval(__val_rd(tanh(f.lo())), __val_ru(tanh(f.hi())));
}

__device__ inline Interval asin(const Interval& f) {
    if (f.hi() < -1.0f || 1.0f < f.lo()) {
        return Interval(NAN, NAN);
    } else {
        return Interval(__val_rd(asin(f.lo())), __val_ru(asin(f.hi())));
    }
}
__device__ inline Interval acos(const Interval& f) {
    if (f.hi() < -1.0f || 1.0f < f.lo()) {
        return Interval(NAN, NAN);
    } else {
        return Interval(__val_rd(acos(f.lo())), __val_ru(acos(f.hi())));
    }
}
__device__ inline Interval atan(const Interval& f) {
    return Interval(__val_rd(atan(f.lo())), __val_ru(atan(f.hi())));
}

__device__ inline Interval asinh(const Interval& f) {
    return Interval(__val_rd(asinh(f.lo())), __val_ru(asinh(f.hi())));
}
__device__ inline Interval acosh(const Interval& f) {
    if (f.hi() < 1.0f) {
        return Interval(NAN, NAN);
    } else {
        return Interval(__val_rd(CALL(fmax)(f.lo(), 1.f)), __val_ru(acos(f.hi())));
    }
}
__device__ inline Interval atanh(const Interval& f) {
    if (f.hi() < -1.0f || 1.0f < f.lo()) {
        return Interval(NAN, NAN);
    } else {
        return Interval(__val_rd(atanh(CALL(fmax)(f.lo(), -1.f))), __val_ru(atanh(CALL(fmin)(f.hi(),  1.f))));
    }
}

#endif