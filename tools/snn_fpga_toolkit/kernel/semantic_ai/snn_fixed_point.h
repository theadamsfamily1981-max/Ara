/*
 * Fixed-Point Arithmetic Library
 *
 * High-performance integer-only fixed-point math for kernel space.
 * Uses Q24.8 format (24 integer bits, 8 fractional bits) for balance
 * between range and precision.
 *
 * This eliminates floating-point emulation overhead and enables
 * microsecond-level AI inference latency in the kernel.
 */

#ifndef _SNN_FIXED_POINT_H
#define _SNN_FIXED_POINT_H

#include <linux/types.h>
#include <linux/kernel.h>

/*
 * Fixed-point format: Q24.8
 * - 24 bits for integer part: range [-8388608, 8388607]
 * - 8 bits for fractional part: precision ~0.004
 * - Total: 32 bits (s32)
 *
 * For Q16.16 (higher precision, smaller range):
 * #define FP_SHIFT 16
 * #define FP_ONE (1 << 16)
 */
#define FP_SHIFT 8
#define FP_ONE (1 << FP_SHIFT)              /* 1.0 in fixed-point */
#define FP_HALF (1 << (FP_SHIFT - 1))       /* 0.5 in fixed-point */
#define FP_QUARTER (1 << (FP_SHIFT - 2))    /* 0.25 in fixed-point */

typedef s32 fp_t;  /* Fixed-point type */

/* Conversion macros */
#define FP_FROM_INT(x)      ((fp_t)((x) << FP_SHIFT))
#define FP_TO_INT(x)        ((s32)((x) >> FP_SHIFT))
#define FP_FROM_FLOAT(x)    ((fp_t)((x) * FP_ONE))
#define FP_TO_FLOAT(x)      ((float)(x) / FP_ONE)

/* Rounding for better accuracy */
#define FP_TO_INT_ROUND(x)  ((s32)(((x) + FP_HALF) >> FP_SHIFT))

/*
 * Basic arithmetic operations
 */

/* Addition: a + b */
static inline fp_t fp_add(fp_t a, fp_t b)
{
    return a + b;
}

/* Subtraction: a - b */
static inline fp_t fp_sub(fp_t a, fp_t b)
{
    return a - b;
}

/* Multiplication: a * b
 * Result needs to be shifted back to maintain fixed-point format
 */
static inline fp_t fp_mul(fp_t a, fp_t b)
{
    /* Use 64-bit intermediate to prevent overflow */
    return (fp_t)(((s64)a * (s64)b) >> FP_SHIFT);
}

/* Division: a / b */
static inline fp_t fp_div(fp_t a, fp_t b)
{
    if (b == 0)
        return 0;  /* Avoid division by zero */

    /* Shift dividend left before division */
    return (fp_t)(((s64)a << FP_SHIFT) / (s64)b);
}

/* Absolute value */
static inline fp_t fp_abs(fp_t x)
{
    return (x < 0) ? -x : x;
}

/* Minimum */
static inline fp_t fp_min(fp_t a, fp_t b)
{
    return (a < b) ? a : b;
}

/* Maximum */
static inline fp_t fp_max(fp_t a, fp_t b)
{
    return (a > b) ? a : b;
}

/* Clamp value between min and max */
static inline fp_t fp_clamp(fp_t x, fp_t min_val, fp_t max_val)
{
    return fp_max(min_val, fp_min(x, max_val));
}

/*
 * Advanced mathematical functions
 */

/*
 * Fast exponential approximation using 2nd-order Padé approximant
 * exp(x) ≈ (1 + x/2) / (1 - x/2)
 *
 * Valid for small |x| (< 2). For larger values, use exp(x) = exp(x/2)^2
 * Error: < 1% for |x| < 1
 */
static inline fp_t fp_exp_small(fp_t x)
{
    fp_t half_x = x >> 1;  /* x / 2 */
    fp_t numerator = FP_ONE + half_x;
    fp_t denominator = FP_ONE - half_x;

    if (denominator == 0)
        denominator = 1;  /* Avoid division by zero */

    return fp_div(numerator, denominator);
}

/*
 * Exponential for larger range using scaling
 * exp(x) = exp(x/2) * exp(x/2)
 */
static inline fp_t fp_exp(fp_t x)
{
    /* For very large values, clamp to prevent overflow */
    if (x > FP_FROM_INT(10))
        return FP_FROM_INT(22026);  /* e^10 ≈ 22026 */
    if (x < FP_FROM_INT(-10))
        return 0;  /* e^-10 ≈ 0 */

    /* Scale down for better accuracy */
    if (fp_abs(x) > FP_ONE) {
        fp_t half_x = x >> 1;
        fp_t half_exp = fp_exp_small(half_x);
        return fp_mul(half_exp, half_exp);
    }

    return fp_exp_small(x);
}

/*
 * Natural logarithm approximation using Taylor series
 * ln(1+x) ≈ x - x²/2 + x³/3 - x⁴/4
 * Valid for |x| < 1
 */
static inline fp_t fp_log_1p(fp_t x)
{
    /* Taylor series: x - x²/2 + x³/3 */
    fp_t x2 = fp_mul(x, x);
    fp_t x3 = fp_mul(x2, x);

    fp_t term1 = x;
    fp_t term2 = fp_div(x2, FP_FROM_INT(2));
    fp_t term3 = fp_div(x3, FP_FROM_INT(3));

    return term1 - term2 + term3;
}

/*
 * Sigmoid function: σ(x) = 1 / (1 + e^(-x))
 * Used for neural network activations
 */
static inline fp_t fp_sigmoid(fp_t x)
{
    fp_t exp_neg_x = fp_exp(-x);
    fp_t denominator = FP_ONE + exp_neg_x;
    return fp_div(FP_ONE, denominator);
}

/*
 * Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 */
static inline fp_t fp_tanh(fp_t x)
{
    fp_t exp_pos = fp_exp(x);
    fp_t exp_neg = fp_exp(-x);
    fp_t numerator = exp_pos - exp_neg;
    fp_t denominator = exp_pos + exp_neg;

    if (denominator == 0)
        return 0;

    return fp_div(numerator, denominator);
}

/*
 * Power function: x^n for integer n
 */
static inline fp_t fp_pow_int(fp_t x, s32 n)
{
    fp_t result = FP_ONE;
    fp_t base = x;
    s32 exp = n;

    if (exp < 0) {
        base = fp_div(FP_ONE, base);
        exp = -exp;
    }

    /* Fast exponentiation by squaring */
    while (exp > 0) {
        if (exp & 1)
            result = fp_mul(result, base);
        base = fp_mul(base, base);
        exp >>= 1;
    }

    return result;
}

/*
 * Square root using Newton-Raphson iteration
 * x_{n+1} = (x_n + a/x_n) / 2
 */
static inline fp_t fp_sqrt(fp_t x)
{
    if (x <= 0)
        return 0;

    /* Initial guess: x / 2 */
    fp_t guess = x >> 1;

    if (guess == 0)
        guess = FP_ONE;

    /* Newton-Raphson iterations (4 iterations for good accuracy) */
    for (int i = 0; i < 4; i++) {
        fp_t quotient = fp_div(x, guess);
        guess = (guess + quotient) >> 1;  /* Average */
    }

    return guess;
}

/*
 * Efficient power-of-2 operations for EMA smoothing
 * Alpha values that are powers of 2 enable ultra-fast implementation
 */

/* EMA update with power-of-2 alpha
 * y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
 *
 * For alpha = 1/2^k (e.g., 1/2, 1/4, 1/8, 1/16):
 * y[n] = y[n-1] + (x[n] - y[n-1]) >> k
 *
 * This is pure integer arithmetic with a single shift!
 */
static inline fp_t fp_ema_pow2(fp_t x, fp_t y_prev, u32 alpha_shift)
{
    fp_t delta = x - y_prev;
    return y_prev + (delta >> alpha_shift);
}

/* Standard EMA with arbitrary alpha (slower but more flexible) */
static inline fp_t fp_ema(fp_t x, fp_t y_prev, fp_t alpha)
{
    /* y = alpha * x + (1 - alpha) * y_prev */
    fp_t one_minus_alpha = FP_ONE - alpha;
    return fp_mul(alpha, x) + fp_mul(one_minus_alpha, y_prev);
}

/*
 * Softmax computation for probability distributions
 * Used for stable, continuous action selection in RL
 *
 * softmax(x_i) = exp(x_i / T) / sum(exp(x_j / T))
 * where T is temperature parameter
 */

/* Compute softmax probabilities from Q-values
 * Returns index sampled from softmax distribution
 *
 * @q_values: Array of Q-values
 * @n: Number of actions
 * @temperature: Temperature parameter (higher = more random)
 * @random_val: Random value in [0, sum_exp) for sampling
 *
 * Note: Uses numerically stable softmax with max subtraction
 */
static inline u32 fp_softmax_sample(const fp_t *q_values, u32 n,
                                    fp_t temperature, u64 random_val)
{
    fp_t max_q = q_values[0];
    fp_t exp_values[128];  /* Max 128 actions */
    fp_t sum_exp = 0;
    u32 i;

    /* Find max Q-value for numerical stability */
    for (i = 1; i < n; i++) {
        if (q_values[i] > max_q)
            max_q = q_values[i];
    }

    /* Compute exp((Q - max_Q) / T) and sum */
    for (i = 0; i < n; i++) {
        fp_t scaled = fp_div(q_values[i] - max_q, temperature);
        exp_values[i] = fp_exp(scaled);
        sum_exp += exp_values[i];
    }

    /* Sample from distribution */
    if (sum_exp == 0)
        return 0;  /* Fallback */

    fp_t threshold = (fp_t)(random_val % sum_exp);
    fp_t cumsum = 0;

    for (i = 0; i < n; i++) {
        cumsum += exp_values[i];
        if (threshold < cumsum)
            return i;
    }

    return n - 1;  /* Fallback to last action */
}

/*
 * Quantization helpers
 */

/* Quantize fixed-point to INT8 */
static inline s8 fp_quantize_s8(fp_t value, fp_t scale, s8 zero_point)
{
    fp_t scaled = fp_div(value, scale);
    s32 quantized = FP_TO_INT_ROUND(scaled) + zero_point;

    /* Clamp to INT8 range */
    if (quantized < -128)
        return -128;
    if (quantized > 127)
        return 127;

    return (s8)quantized;
}

/* Dequantize INT8 to fixed-point */
static inline fp_t fp_dequantize_s8(s8 q_value, fp_t scale, s8 zero_point)
{
    return fp_mul(FP_FROM_INT(q_value - zero_point), scale);
}

/*
 * Debug helpers (only in debug builds)
 */

#ifdef DEBUG
/* Print fixed-point value (for debugging) */
static inline void fp_print(const char *name, fp_t value)
{
    s32 integer = FP_TO_INT(value);
    s32 fractional = fp_abs(value & (FP_ONE - 1));
    fractional = (fractional * 100) >> FP_SHIFT;

    if (value < 0 && integer == 0)
        pr_debug("%s: -%d.%02d\n", name, integer, fractional);
    else
        pr_debug("%s: %d.%02d\n", name, integer, fractional);
}
#else
#define fp_print(name, value) do { } while (0)
#endif

#endif /* _SNN_FIXED_POINT_H */
