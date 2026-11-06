// Host test for the int8 CNN kernels (Phase H5): for every exported patch, including the
// stress patch that saturates the int8 input, the C code must reproduce Python's int8 input,
// every conv layer's int8 output and the head's int32 sums exactly, and the logits and
// anomaly score within the tolerance below.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "andon_model.h"  // for the shape defines; the kernels use andon_model
#include "andon_test_vectors.h"
#include "model.h"
#include "model_params.h"

// Logits and scores are float: Python computes them in float64, C in float32, and the
// header stores Python's values as float32. |C - Python| <= TOL_ABS + TOL_REL * |Python|.
#define TOL_ABS 1e-5f
#define TOL_REL 1e-5f

#define PATCH_LEN (ANDON_TV_N_MELS * ANDON_PATCH_FRAMES)

static int failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                  \
        }                                                                \
    } while (0)

static int close_enough(float c, float py) {
    return fabsf(c - py) <= TOL_ABS + TOL_REL * fabsf(py);
}

// Number of positions where two int8 tensors differ.
static long count_diff(const int8_t *a, const int8_t *b, long n) {
    long d = 0;
    for (long i = 0; i < n; i++) d += a[i] != b[i];
    return d;
}

static const int8_t *const expected_layers[ANDON_N_CONV] = {
    andon_tv_conv0, andon_tv_conv1, andon_tv_conv2, andon_tv_conv3};
static const long layer_sizes[ANDON_N_CONV] = {
    sizeof andon_tv_conv0 / ANDON_TV_N, sizeof andon_tv_conv1 / ANDON_TV_N,
    sizeof andon_tv_conv2 / ANDON_TV_N, sizeof andon_tv_conv3 / ANDON_TV_N};

static void test_patch(int n) {
    static int8_t input[PATCH_LEN];
    static int8_t layer_store[ANDON_N_CONV][MODEL_ACT_MAX];
    int8_t *layers[ANDON_N_CONV];
    for (int i = 0; i < ANDON_N_CONV; i++) layers[i] = layer_store[i];

    model_quantize_input(&andon_model, &andon_tv_patch[n * PATCH_LEN], input);
    const long input_diff = count_diff(input, &andon_tv_input[n * PATCH_LEN], PATCH_LEN);
    CHECK(input_diff == 0);

    // Each layer runs on Python's exact input, so a wrong layer is reported on its own.
    int32_t acc[ANDON_N_CLASSES];
    CHECK(model_head_acc(&andon_model, &andon_tv_input[n * PATCH_LEN], acc, layers) == 0);
    long layer_diff[ANDON_N_CONV];
    for (int i = 0; i < ANDON_N_CONV; i++) {
        layer_diff[i] = count_diff(layers[i], &expected_layers[i][n * layer_sizes[i]],
                                   layer_sizes[i]);
        CHECK(layer_diff[i] == 0);
    }
    for (int k = 0; k < ANDON_N_CLASSES; k++) {
        CHECK(acc[k] == andon_tv_head_acc[n * ANDON_N_CLASSES + k]);
    }

    float logits[ANDON_N_CLASSES];
    model_logits(&andon_model, acc, logits);
    float worst_logit = 0.0f;
    for (int k = 0; k < ANDON_N_CLASSES; k++) {
        const float py = andon_tv_logits[n * ANDON_N_CLASSES + k];
        const float err = fabsf(logits[k] - py);
        if (err > worst_logit) worst_logit = err;
        CHECK(close_enough(logits[k], py));
    }
    const float score = model_score(ANDON_N_CLASSES, logits, ANDON_TV_TRUE_ID);
    CHECK(close_enough(score, andon_tv_score[n]));

    // The whole patch path (float patch to score) gives the same score.
    const float whole = model_score_patch(&andon_model, &andon_tv_patch[n * PATCH_LEN],
                                          ANDON_TV_TRUE_ID);
    CHECK(close_enough(whole, andon_tv_score[n]));

    printf("patch %d%s: int8 input diffs %ld, layer diffs %ld/%ld/%ld/%ld,"
           " worst logit err %.2e, score C %.6f vs Python %.6f\n",
           n, n == ANDON_TV_N - 1 ? " (stress)" : "", input_diff, layer_diff[0],
           layer_diff[1], layer_diff[2], layer_diff[3], worst_logit, score, andon_tv_score[n]);
}

static void test_stress_patch_saturates(void) {
    const int8_t *q = &andon_tv_input[(ANDON_TV_N - 1) * PATCH_LEN];
    int lo = 0, hi = 0;
    for (int i = 0; i < PATCH_LEN; i++) {
        lo |= q[i] == -128;
        hi |= q[i] == 127;
    }
    CHECK(lo && hi);
}

// np.round rounds exact halves to even; the audio vectors never hit an exact half.
static void test_input_rounds_half_to_even(void) {
    static const float zeros[1] = {0.0f}, ones[1] = {1.0f};
    model_t m = andon_model;
    m.n_mels = 1;
    m.patch_frames = 6;
    m.band_mean = zeros;
    m.band_std = ones;
    m.input_scale = 1.0;
    m.input_zero = 0;
    const float patch[6] = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, 200.0f};
    const int8_t expected[6] = {0, 2, 2, 0, -2, 127};
    int8_t q[6];
    model_quantize_input(&m, patch, q);
    CHECK(memcmp(q, expected, sizeof q) == 0);
}

static void test_rejects_wrong_shape(void) {
    model_t bad = andon_model;
    bad.patch_frames = ANDON_PATCH_FRAMES * 2;  // pool would see 64 positions, not 32
    int32_t acc[ANDON_N_CLASSES];
    CHECK(model_head_acc(&bad, andon_tv_input, acc, NULL) != 0);
}

int main(void) {
    CHECK(ANDON_TV_N_MELS == ANDON_N_MELS);
    for (int n = 0; n < ANDON_TV_N; n++) test_patch(n);
    test_stress_patch_saturates();
    test_input_rounds_half_to_even();
    test_rejects_wrong_shape();
    printf("kernel static RAM: %u bytes\n", (unsigned)model_ram_bytes());
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
