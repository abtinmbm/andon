// Host test for the andon light (Phase H6) against andon/light.py: the limits fitted on the
// first ANDON_LV_N_CALIB scores of the exported light vectors, and the light after each
// later score, with the light settings of the final config (andon_gmm.h).
#include <math.h>
#include <stdio.h>

#include "andon_light.h"
#include "andon_gmm_vectors.h"
#include "gmm_params.h"

static int failures = 0;
#define CHECK(cond)                                                \
    do {                                                           \
        if (!(cond)) {                                             \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                            \
        }                                                          \
    } while (0)

// Python fits in float64 with numpy's two-pass mean and ddof=1 sd; Welford's single pass
// differs from it by rounding only. Measured worst on the vectors: printed below.
#define LIMIT_RTOL 1e-12

static double rel(double got, double want) { return fabs(got - want) / fmax(fabs(want), 1.0); }

static void test_limits_and_light_match_python(void) {
    detector_config_t cfg;
    andon_detector_config(&cfg);
    light_calib_t c;
    light_calib_reset(&c);
    for (int i = 0; i < ANDON_LV_N_CALIB; i++) light_calib_add(&c, andon_lv_score[i]);
    light_limits_t lim;
    CHECK(light_calib_limits(&c, cfg.warn_sigma, cfg.control_sigma, &lim));
    const double got[4] = {lim.center, lim.sigma, lim.warn, lim.control};
    double worst = 0.0;
    for (int k = 0; k < 4; k++) worst = fmax(worst, rel(got[k], andon_lv_limits[k]));
    printf("light: limits, worst relative difference from Python %.2e (tolerance %.0e)\n", worst,
           LIMIT_RTOL);
    CHECK(worst <= LIMIT_RTOL);

    // Run on Python's limits, so a rounding difference above can't move a comparison.
    const light_limits_t py = {andon_lv_limits[0], andon_lv_limits[1], andon_lv_limits[2],
                               andon_lv_limits[3]};
    light_t l;
    light_reset(&l);
    int mismatches = 0, saw[3] = {0, 0, 0};
    for (int i = ANDON_LV_N_CALIB; i < ANDON_LV_N_SCORES; i++) {
        andon_light_t s = light_update(&l, &py, &cfg.light, andon_lv_score[i]);
        mismatches += (int)s != andon_lv_light[i - ANDON_LV_N_CALIB];
        saw[s]++;
    }
    printf("light: %d of %d states differ from Python\n", mismatches,
           ANDON_LV_N_SCORES - ANDON_LV_N_CALIB);
    CHECK(mismatches == 0);
    CHECK(saw[ANDON_GREEN] && saw[ANDON_YELLOW] && saw[ANDON_RED]);  // every state reached
}

static void test_red_latches_until_reset(void) {
    const light_settings_t s = {2, 3, 3};
    const light_limits_t lim = {0.0, 1.0, 2.0, 3.0};
    light_t l;
    light_reset(&l);
    for (int i = 0; i < 3; i++) light_update(&l, &lim, &s, 10.0f);
    CHECK(l.state == ANDON_RED);
    for (int i = 0; i < 10; i++) CHECK(light_update(&l, &lim, &s, 0.0f) == ANDON_RED);
    light_reset(&l);
    CHECK(l.state == ANDON_GREEN && l.above_control == 0);
    // After a reset the run counts start over: two more high scores are not red yet.
    light_update(&l, &lim, &s, 10.0f);
    CHECK(light_update(&l, &lim, &s, 10.0f) == ANDON_YELLOW);
    light_trip(&l);
    CHECK(l.state == ANDON_RED);
}

static void test_unusable_calibration_is_refused(void) {
    light_calib_t c;
    light_limits_t lim = {-1.0, -1.0, -1.0, -1.0};
    light_calib_reset(&c);
    light_calib_add(&c, 5.0);
    CHECK(!light_calib_limits(&c, 2.0, 3.0, &lim));  // one score
    light_calib_add(&c, 5.0);
    CHECK(!light_calib_limits(&c, 2.0, 3.0, &lim));  // sigma 0
    light_calib_add(&c, NAN);
    CHECK(!light_calib_limits(&c, 2.0, 3.0, &lim));  // not finite
    CHECK(lim.center == -1.0);                       // left alone
    light_calib_reset(&c);
    light_calib_add(&c, 1.0);
    light_calib_add(&c, 3.0);
    CHECK(light_calib_limits(&c, 2.0, 3.0, &lim));
    CHECK(lim.center == 2.0 && fabs(lim.sigma - sqrt(2.0)) < 1e-15);
}

int main(void) {
    test_limits_and_light_match_python();
    test_red_latches_until_reset();
    test_unusable_calibration_is_refused();
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
