// Host test for the GMM scorer and the streaming window mean (Phase H5), on values worked
// out by hand. The exported real-fan vectors are checked end to end by test_selftest.c.
#include <math.h>
#include <stdio.h>

#include "gmm.h"

static int failures = 0;
#define CHECK(cond)                                                \
    do {                                                           \
        if (!(cond)) {                                             \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                            \
        }                                                          \
    } while (0)

static int close_to(float got, double want, double tol) { return fabs(got - want) <= tol; }

static const double LOG_2PI = 1.8378770664093453;

// One component, P = I in 2-D: the score is log(2 pi) + |x - mean|^2 / 2.
static void test_identity(void) {
    const float mean[2] = {0.0f, 0.0f};
    const float chol[3] = {1.0f, 0.0f, 1.0f};  // column 0: P00; column 1: P01, P11
    const float log_const[1] = {(float)-LOG_2PI};
    const gmm_t g = {1, 2, mean, chol, log_const};
    const float x[2] = {1.0f, 2.0f};
    CHECK(close_to(gmm_score(&g, x), LOG_2PI + 2.5, 1e-5));
    CHECK(close_to(gmm_score(&g, mean), LOG_2PI, 1e-6));
}

// P = [[2, 1], [0, 3]] (upper triangular), diff = (1, 1): y = diff P = (2, 4), |y|^2 = 20.
// The packing is column by column, so a row-by-row reading would give y = (3, 3) instead.
static void test_packed_columns(void) {
    const float mean[2] = {1.0f, -1.0f};
    const float chol[3] = {2.0f, 1.0f, 3.0f};
    const float log_const[1] = {0.5f};
    const gmm_t g = {1, 2, mean, chol, log_const};
    const float x[2] = {2.0f, 0.0f};
    CHECK(close_to(gmm_score(&g, x), -(0.5 - 10.0), 1e-5));
}

// Two identical components with weight 1/2 each give the same score as one with weight 1.
static void test_logsumexp(void) {
    const float mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float chol[6] = {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const float half = (float)(log(0.5) - LOG_2PI);
    const float log_const[2] = {half, half};
    const gmm_t g = {2, 2, mean, chol, log_const};
    const float x[2] = {3.0f, -4.0f};
    CHECK(close_to(gmm_score(&g, x), LOG_2PI + 12.5, 1e-5));
}

// Far from every component, exp(log p) underflows to 0 in float32; the score must not.
static void test_far_away_stays_finite(void) {
    const float mean[4] = {0.0f, 0.0f, 10.0f, 10.0f};
    const float chol[6] = {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const float log_const[2] = {0.0f, 0.0f};
    const gmm_t g = {2, 2, mean, chol, log_const};
    const float x[2] = {1000.0f, 1000.0f};
    // The nearer component dominates: |x - (10, 10)|^2 / 2 = 990^2.
    const float s = gmm_score(&g, x);
    CHECK(isfinite(s));
    CHECK(close_to(s, 990.0 * 990.0, 1.0));
}

// A model larger than the scratch buffers is refused, not scored past their end.
static void test_oversized_model_is_refused(void) {
    const float zero[GMM_MAX_COMPONENTS + 1] = {0.0f};
    const gmm_t too_many = {GMM_MAX_COMPONENTS + 1, 1, zero, zero, zero};
    const gmm_t too_wide = {1, GMM_MAX_DIM + 1, zero, zero, zero};
    const gmm_t empty = {0, 1, zero, zero, zero};
    CHECK(isnan(gmm_score(&too_many, zero)));
    CHECK(isnan(gmm_score(&too_wide, zero)));
    CHECK(isnan(gmm_score(&empty, zero)));
}

// hop 2: windows are frames 0-3, 2-5, 4-7, ... as h3 cuts them (width = 2 * hop).
static void test_window_mean(void) {
    gmm_window_t w;
    CHECK(gmm_window_reset(&w, 0, 1) == -1);
    CHECK(gmm_window_reset(&w, 2, GMM_MAX_DIM + 1) == -1);
    CHECK(gmm_window_reset(&w, 2, 2) == 0);
    float mean[2] = {-1.0f, -1.0f};
    int done[8];
    float got[8][2];
    for (int t = 0; t < 8; t++) {
        const float frame[2] = {(float)(t + 1), (float)(-10 * (t + 1))};
        done[t] = gmm_window_push(&w, frame, mean);
        got[t][0] = mean[0];
        got[t][1] = mean[1];
    }
    CHECK(!done[0] && !done[1] && !done[2] && done[3] && !done[4] && done[5] && !done[6]);
    CHECK(done[7]);
    CHECK(close_to(got[3][0], 2.5, 1e-6) && close_to(got[3][1], -25.0, 1e-5));  // 1..4
    CHECK(close_to(got[5][0], 4.5, 1e-6) && close_to(got[5][1], -45.0, 1e-5));  // 3..6
    CHECK(close_to(got[7][0], 6.5, 1e-6));                                        // 5..8
    CHECK(got[2][0] == -1.0f);  // untouched until the first window is complete
    // A reset starts over: the first window needs 2 * hop new frames again.
    gmm_window_reset(&w, 2, 2);
    const float z[2] = {0.0f, 0.0f};
    CHECK(!gmm_window_push(&w, z, mean) && !gmm_window_push(&w, z, mean));
    CHECK(!gmm_window_push(&w, z, mean) && gmm_window_push(&w, z, mean));
    CHECK(mean[0] == 0.0f);
}

int main(void) {
    test_identity();
    test_packed_columns();
    test_logsumexp();
    test_far_away_stays_finite();
    test_oversized_model_is_refused();
    test_window_mean();
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("gmm: static RAM per window state %u bytes\n", (unsigned)sizeof(gmm_window_t));
    printf("all checks passed\n");
    return 0;
}
