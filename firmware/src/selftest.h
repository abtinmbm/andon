// End-to-end self-test: the exported test audio through the C front end and the int8 CNN,
// and each real fan's test audio through the front end, the window mean and the per-fan
// GMMs, all checked against Python and timed. The same code runs on the PC (test_selftest.c) and on
// the Pico 2 (bench.c, timed with time_us_64()), so the board reports exactly what CI checks.
#ifndef SELFTEST_H
#define SELFTEST_H

#include <stdint.h>

// Tolerances, each against the Python values in andon_test_vectors.h.
#define SELFTEST_LOGMEL_TOL_DB 0.001f  // C log-mel frame vs Python's
#define SELFTEST_SCORE_TOL 1e-5f       // score from Python's int8 input (only float differs)
// Score from the C log-mel. A log-mel value within 1e-5 dB of a rounding boundary can land
// on the other int8 step, which moves the score slightly; the measured error is printed.
#define SELFTEST_E2E_SCORE_TOL 1e-3f
// GMM score errors are relative, |C - Python| / max(|Python|, 1): scores run from hundreds
// (a fan's own audio) to hundreds of thousands (another fan's), and float32 keeps about 7
// digits of each. From Python's window means only the scorer's float32 arithmetic differs;
// end to end, the C log-mel's last bits move the window mean too.
#define SELFTEST_GMM_MEAN_TOL_DB 0.001f  // C window mean vs Python's
#define SELFTEST_GMM_SCORE_RTOL 1e-5f
#define SELFTEST_GMM_E2E_RTOL 1e-4f

typedef uint64_t (*selftest_clock_fn)(void);  // microseconds

typedef struct {
    int frames;
    int patches;
    float worst_logmel_db;
    long input_diffs;  // int8 inputs from the C log-mel that differ from Python's
    long input_values;
    int head_mismatches;  // head sums from Python's int8 input that differ
    float worst_score_err;
    float worst_e2e_score_err;
    uint64_t logmel_us_total;  // all frames
    uint64_t logmel_us_max;    // slowest frame
    uint64_t model_us_total;   // quantize + CNN + score, all patches
    uint64_t model_us_max;     // slowest patch
    int gmm_frames;            // real-fan audio frames, all fans
    int gmm_scores;            // windows x models
    float worst_gmm_logmel_db;
    float worst_gmm_mean_db;
    float worst_gmm_score_rel;  // from Python's window means
    float worst_gmm_e2e_rel;    // from the C log-mel and the C window mean
    uint64_t gmm_us_total;      // gmm_score only, every end-to-end score
    uint64_t gmm_us_max;
    uint32_t ram_bytes;        // static RAM of the front end, kernels and this test
    int pass;
} selftest_result_t;

// Runs everything once. Returns result->pass.
int selftest_run(selftest_clock_fn now_us, selftest_result_t *result);

// Prints the result as a few readable lines (printf; USB serial on the board).
void selftest_print(const selftest_result_t *result);

#endif  // SELFTEST_H
