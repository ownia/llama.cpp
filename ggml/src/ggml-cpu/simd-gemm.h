#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "simd-mappings.h"

// TODO: add support for sizeless vector types
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON__)
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        for (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        for (; jj < N; jj++) {
            for (int64_t i = 0; i < GEMM_RM; i++) {
                float a = C[i * N + jj];
                for (int64_t kk = 0; kk < K; kk++) {
                    a += A[i + kk] * B[kk * N + jj];
                }
                C[i * N + jj] = a;
            }
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    for (; ii < M; ii++) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        for (; jj < N; jj++) {
            float a = C[jj];
            for (int64_t kk = 0; kk < K; kk++) {
                a += A[kk] * B[kk * N + jj];
            }
            C[jj] = a;
        }

        A += K;
        C += N;
    }
}
#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <int RM>
static inline void rvv_simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    if constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    if constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    if constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    if constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    if constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    if constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    for (int kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        if constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        if constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        if constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        if constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        if constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        if constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    if constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    if constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    if constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    if constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    if constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    if constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

template <int RM>
static inline void rvv_simd_gemm_dispatch_tail(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int KN, int remaining_rows)
{
    if constexpr (RM > 0) {
        if (remaining_rows == RM) {
            int64_t jj = 0;
            for (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            if (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } else {
            rvv_simd_gemm_dispatch_tail<RM - 1>(C, A, B, K, N, KN, remaining_rows);
        }
    }
}

static constexpr int GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    const int KN = (int)__riscv_vlenb();
    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        if (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    int remaining_rows = M - ii;
    rvv_simd_gemm_dispatch_tail<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#elif defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

static constexpr int GEMM_RM = 6;
static constexpr int GEMM_RN = 4;

template <int RM, int RN>
static inline void sve_simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int vl, svbool_t pg)
{
    // Load accumulators — RM rows × RN vector columns
    svfloat32_t c00 = svld1_f32(pg, C + 0 * N + 0 * vl);
    svfloat32_t c01, c02, c03;
    if constexpr (RN > 1) c01 = svld1_f32(pg, C + 0 * N + 1 * vl);
    if constexpr (RN > 2) c02 = svld1_f32(pg, C + 0 * N + 2 * vl);
    if constexpr (RN > 3) c03 = svld1_f32(pg, C + 0 * N + 3 * vl);

    svfloat32_t c10, c11, c12, c13;
    if constexpr (RM > 1) {
        c10 = svld1_f32(pg, C + 1 * N + 0 * vl);
        if constexpr (RN > 1) c11 = svld1_f32(pg, C + 1 * N + 1 * vl);
        if constexpr (RN > 2) c12 = svld1_f32(pg, C + 1 * N + 2 * vl);
        if constexpr (RN > 3) c13 = svld1_f32(pg, C + 1 * N + 3 * vl);
    }

    svfloat32_t c20, c21, c22, c23;
    if constexpr (RM > 2) {
        c20 = svld1_f32(pg, C + 2 * N + 0 * vl);
        if constexpr (RN > 1) c21 = svld1_f32(pg, C + 2 * N + 1 * vl);
        if constexpr (RN > 2) c22 = svld1_f32(pg, C + 2 * N + 2 * vl);
        if constexpr (RN > 3) c23 = svld1_f32(pg, C + 2 * N + 3 * vl);
    }

    svfloat32_t c30, c31, c32, c33;
    if constexpr (RM > 3) {
        c30 = svld1_f32(pg, C + 3 * N + 0 * vl);
        if constexpr (RN > 1) c31 = svld1_f32(pg, C + 3 * N + 1 * vl);
        if constexpr (RN > 2) c32 = svld1_f32(pg, C + 3 * N + 2 * vl);
        if constexpr (RN > 3) c33 = svld1_f32(pg, C + 3 * N + 3 * vl);
    }

    svfloat32_t c40, c41, c42, c43;
    if constexpr (RM > 4) {
        c40 = svld1_f32(pg, C + 4 * N + 0 * vl);
        if constexpr (RN > 1) c41 = svld1_f32(pg, C + 4 * N + 1 * vl);
        if constexpr (RN > 2) c42 = svld1_f32(pg, C + 4 * N + 2 * vl);
        if constexpr (RN > 3) c43 = svld1_f32(pg, C + 4 * N + 3 * vl);
    }

    svfloat32_t c50, c51, c52, c53;
    if constexpr (RM > 5) {
        c50 = svld1_f32(pg, C + 5 * N + 0 * vl);
        if constexpr (RN > 1) c51 = svld1_f32(pg, C + 5 * N + 1 * vl);
        if constexpr (RN > 2) c52 = svld1_f32(pg, C + 5 * N + 2 * vl);
        if constexpr (RN > 3) c53 = svld1_f32(pg, C + 5 * N + 3 * vl);
    }

    for (int kk = 0; kk < K; kk++) {
        // Load RN B vectors — shared across all RM rows
        svfloat32_t b0 = svld1_f32(pg, B + kk * N + 0 * vl);
        svfloat32_t b1, b2, b3;
        if constexpr (RN > 1) b1 = svld1_f32(pg, B + kk * N + 1 * vl);
        if constexpr (RN > 2) b2 = svld1_f32(pg, B + kk * N + 2 * vl);
        if constexpr (RN > 3) b3 = svld1_f32(pg, B + kk * N + 3 * vl);

        {
            svfloat32_t a = svdup_f32(A[0 * K + kk]);
            c00 = svmla_f32_x(pg, c00, b0, a);
            if constexpr (RN > 1) c01 = svmla_f32_x(pg, c01, b1, a);
            if constexpr (RN > 2) c02 = svmla_f32_x(pg, c02, b2, a);
            if constexpr (RN > 3) c03 = svmla_f32_x(pg, c03, b3, a);
        }
        if constexpr (RM > 1) {
            svfloat32_t a = svdup_f32(A[1 * K + kk]);
            c10 = svmla_f32_x(pg, c10, b0, a);
            if constexpr (RN > 1) c11 = svmla_f32_x(pg, c11, b1, a);
            if constexpr (RN > 2) c12 = svmla_f32_x(pg, c12, b2, a);
            if constexpr (RN > 3) c13 = svmla_f32_x(pg, c13, b3, a);
        }
        if constexpr (RM > 2) {
            svfloat32_t a = svdup_f32(A[2 * K + kk]);
            c20 = svmla_f32_x(pg, c20, b0, a);
            if constexpr (RN > 1) c21 = svmla_f32_x(pg, c21, b1, a);
            if constexpr (RN > 2) c22 = svmla_f32_x(pg, c22, b2, a);
            if constexpr (RN > 3) c23 = svmla_f32_x(pg, c23, b3, a);
        }
        if constexpr (RM > 3) {
            svfloat32_t a = svdup_f32(A[3 * K + kk]);
            c30 = svmla_f32_x(pg, c30, b0, a);
            if constexpr (RN > 1) c31 = svmla_f32_x(pg, c31, b1, a);
            if constexpr (RN > 2) c32 = svmla_f32_x(pg, c32, b2, a);
            if constexpr (RN > 3) c33 = svmla_f32_x(pg, c33, b3, a);
        }
        if constexpr (RM > 4) {
            svfloat32_t a = svdup_f32(A[4 * K + kk]);
            c40 = svmla_f32_x(pg, c40, b0, a);
            if constexpr (RN > 1) c41 = svmla_f32_x(pg, c41, b1, a);
            if constexpr (RN > 2) c42 = svmla_f32_x(pg, c42, b2, a);
            if constexpr (RN > 3) c43 = svmla_f32_x(pg, c43, b3, a);
        }
        if constexpr (RM > 5) {
            svfloat32_t a = svdup_f32(A[5 * K + kk]);
            c50 = svmla_f32_x(pg, c50, b0, a);
            if constexpr (RN > 1) c51 = svmla_f32_x(pg, c51, b1, a);
            if constexpr (RN > 2) c52 = svmla_f32_x(pg, c52, b2, a);
            if constexpr (RN > 3) c53 = svmla_f32_x(pg, c53, b3, a);
        }
    }

    // Store accumulators
    svst1_f32(pg, C + 0 * N + 0 * vl, c00);
    if constexpr (RN > 1) svst1_f32(pg, C + 0 * N + 1 * vl, c01);
    if constexpr (RN > 2) svst1_f32(pg, C + 0 * N + 2 * vl, c02);
    if constexpr (RN > 3) svst1_f32(pg, C + 0 * N + 3 * vl, c03);

    if constexpr (RM > 1) {
        svst1_f32(pg, C + 1 * N + 0 * vl, c10);
        if constexpr (RN > 1) svst1_f32(pg, C + 1 * N + 1 * vl, c11);
        if constexpr (RN > 2) svst1_f32(pg, C + 1 * N + 2 * vl, c12);
        if constexpr (RN > 3) svst1_f32(pg, C + 1 * N + 3 * vl, c13);
    }
    if constexpr (RM > 2) {
        svst1_f32(pg, C + 2 * N + 0 * vl, c20);
        if constexpr (RN > 1) svst1_f32(pg, C + 2 * N + 1 * vl, c21);
        if constexpr (RN > 2) svst1_f32(pg, C + 2 * N + 2 * vl, c22);
        if constexpr (RN > 3) svst1_f32(pg, C + 2 * N + 3 * vl, c23);
    }
    if constexpr (RM > 3) {
        svst1_f32(pg, C + 3 * N + 0 * vl, c30);
        if constexpr (RN > 1) svst1_f32(pg, C + 3 * N + 1 * vl, c31);
        if constexpr (RN > 2) svst1_f32(pg, C + 3 * N + 2 * vl, c32);
        if constexpr (RN > 3) svst1_f32(pg, C + 3 * N + 3 * vl, c33);
    }
    if constexpr (RM > 4) {
        svst1_f32(pg, C + 4 * N + 0 * vl, c40);
        if constexpr (RN > 1) svst1_f32(pg, C + 4 * N + 1 * vl, c41);
        if constexpr (RN > 2) svst1_f32(pg, C + 4 * N + 2 * vl, c42);
        if constexpr (RN > 3) svst1_f32(pg, C + 4 * N + 3 * vl, c43);
    }
    if constexpr (RM > 5) {
        svst1_f32(pg, C + 5 * N + 0 * vl, c50);
        if constexpr (RN > 1) svst1_f32(pg, C + 5 * N + 1 * vl, c51);
        if constexpr (RN > 2) svst1_f32(pg, C + 5 * N + 2 * vl, c52);
        if constexpr (RN > 3) svst1_f32(pg, C + 5 * N + 3 * vl, c53);
    }
}

template <int RM>
static inline void sve_simd_gemm_dispatch_tail(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int vl, int remaining_rows)
{
    if constexpr (RM > 0) {
        if (remaining_rows == RM) {
            int64_t jj = 0;

            for (; jj + GEMM_RN * vl <= N; jj += GEMM_RN * vl) {
                svbool_t pg = svptrue_b32();
                sve_simd_gemm_ukernel<RM, GEMM_RN>(C + jj, A, B + jj, K, N, vl, pg);
            }

            for (; jj + vl <= N; jj += vl) {
                svbool_t pg = svptrue_b32();
                sve_simd_gemm_ukernel<RM, 1>(C + jj, A, B + jj, K, N, vl, pg);
            }

            if (jj < N) {
                svbool_t pg = svwhilelt_b32_u64(jj, N);
                sve_simd_gemm_ukernel<RM, 1>(C + jj, A, B + jj, K, N, vl, pg);
            }
        } else {
            sve_simd_gemm_dispatch_tail<RM - 1>(C, A, B, K, N, vl, remaining_rows);
        }
    }
}

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    const int vl = (int)svcntw();
    int64_t ii = 0;
    int remaining_rows = 0;

    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;

        for (; jj + GEMM_RN * vl <= N; jj += GEMM_RN * vl) {
            svbool_t pg = svptrue_b32();
            sve_simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N, vl, pg);
        }

        for (; jj + vl <= N; jj += vl) {
            svbool_t pg = svptrue_b32();
            sve_simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N, vl, pg);
        }

        if (jj < N) {
            svbool_t pg = svwhilelt_b32_u64(jj, N);
            sve_simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N, vl, pg);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    remaining_rows = M - ii;
    if (remaining_rows > 0) {
        sve_simd_gemm_dispatch_tail<GEMM_RM - 1>(C, A, B, K, N, vl, remaining_rows);
    }
}

#else // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
