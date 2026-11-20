// Host test for the on-board detector (Phase H6). The exported real-mic test audio is fed
// as I2S words, 256 at a time as the mic delivers them, through the whole pipeline: the
// framing, the log-mel, the window and the score must give the scores Python gives for the
// board's stream (andon_gv_stream_score). Then the phases: hold-off, calibration, the trip,
// the reset, and the dead-microphone checks.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "andon_gmm_vectors.h"
#include "detector.h"
#include "gmm_params.h"
#include "mic_stream.h"

static int failures = 0;
#define CHECK(cond)                                                \
    do {                                                           \
        if (!(cond)) {                                             \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                            \
        }                                                          \
    } while (0)

// End to end from int16 audio, as test_selftest.c's GMM check: float32 log-mel and window
// means against float64 Python, so a relative tolerance, |C - Python| / max(|Python|, 1).
#define SCORE_RTOL 1e-4
#define CHUNK MIC_FRAME_SAMPLES
#define CHUNK_US 16000u  // 256 samples at 16 kHz

static detector_config_t base;

typedef struct {
    float score[64];
    int n;
    unsigned ev;
    uint64_t t_us;
} run_t;

// The I2S word that decodes to s at the gain shift: mic_word_to_sample keeps word >> (16 - g).
static uint32_t word_of(int16_t s) { return (uint32_t)((int32_t)s * (1 << (16 - base.gain_shift))); }

static void feed(detector_t *d, const int16_t *s, int n, bool overrun, run_t *r) {
    uint32_t words[CHUNK];
    for (int at = 0; at < n; at += CHUNK) {
        const int k = n - at < (int)CHUNK ? n - at : (int)CHUNK;
        for (int i = 0; i < k; i++) words[i] = word_of(s[at + i]);
        r->t_us += CHUNK_US;
        unsigned ev = detector_feed(d, words, (uint32_t)k, overrun && at == 0, r->t_us);
        ev |= detector_tick(d, r->t_us);
        if ((ev & DET_EV_WINDOW) && r->n < 64) r->score[r->n++] = d->last_score;
        r->ev |= ev;
    }
}

static const int16_t *audio(int fan) { return &andon_gv_audio[fan * ANDON_GV_AUDIO_LEN]; }

static double want(int fan, int w, int model) {
    return andon_gv_stream_score[(fan * ANDON_GV_STREAM_WINDOWS + w) * ANDON_GV_N_FANS + model];
}

static double rel(double got, double w) { return fabs(got - w) / fmax(fabs(w), 1.0); }

static void start(detector_t *d, uint32_t holdoff, int calib, int model) {
    detector_config_t cfg = base;
    cfg.holdoff_samples = holdoff;
    cfg.calib_windows = calib;
    detector_init(d, &cfg);
    detector_watch(d, &andon_gmm[model], model, 0);
    detector_set_running(d, true, 0);
}

static void test_stream_scores_match_python(void) {
    static detector_t d;
    double worst = 0.0;
    for (int f = 0; f < ANDON_GV_N_FANS; f++) {
        for (int m = 0; m < ANDON_GV_N_FANS; m++) {
            start(&d, 0, base.calib_windows, m);
            run_t r = {0};
            feed(&d, audio(f), ANDON_GV_AUDIO_LEN, false, &r);
            CHECK(r.n == ANDON_GV_STREAM_WINDOWS);
            for (int w = 0; w < r.n && w < ANDON_GV_STREAM_WINDOWS; w++) {
                worst = fmax(worst, rel(r.score[w], want(f, w, m)));
            }
            CHECK(!(r.ev & DET_EV_TRIP) && d.phase == DET_CALIBRATING);
        }
    }
    printf("detector: %d fans x %d stream windows x %d models, worst relative score "
           "difference %.2e (tolerance %.0e)\n",
           ANDON_GV_N_FANS, ANDON_GV_STREAM_WINDOWS, ANDON_GV_N_FANS, worst, SCORE_RTOL);
    CHECK(worst <= SCORE_RTOL);
}

static void test_holdoff_skips_the_spin_up(void) {
    static detector_t d, ref;
    static int16_t spin[3000];  // zeros: they would trip the flat check if they were framed
    start(&ref, 0, base.calib_windows, 0);
    start(&d, 3000, base.calib_windows, 0);
    CHECK(d.phase == DET_HOLDOFF);
    run_t r0 = {0}, r = {0};
    feed(&ref, audio(0), ANDON_GV_AUDIO_LEN, false, &r0);
    feed(&d, spin, 2999, false, &r);
    CHECK(d.phase == DET_HOLDOFF);
    feed(&d, spin, 1, false, &r);
    CHECK(d.phase == DET_CALIBRATING);
    feed(&d, audio(0), ANDON_GV_AUDIO_LEN, false, &r);
    CHECK(r.n == r0.n && r.n == ANDON_GV_STREAM_WINDOWS);
    CHECK(memcmp(r.score, r0.score, sizeof(float) * (size_t)r.n) == 0);
}

static void test_calibrates_then_trips_and_resets(void) {
    static detector_t d;
    start(&d, 0, 2, 0);
    run_t r = {0};
    feed(&d, audio(0), ANDON_GV_AUDIO_LEN, false, &r);
    CHECK(r.ev & DET_EV_CALIBRATED);
    CHECK(d.phase == DET_WATCHING && d.have_limits);
    const double mu = (want(0, 0, 0) + want(0, 1, 0)) / 2.0;
    const double sd = fabs(want(0, 0, 0) - want(0, 1, 0)) / sqrt(2.0);
    CHECK(rel(d.limits.center, mu) <= SCORE_RTOL);
    CHECK(fabs(d.limits.sigma - sd) <= SCORE_RTOL * fmax(fabs(mu), 1.0));
    CHECK(d.limits.warn == d.limits.center + base.warn_sigma * d.limits.sigma);

    // Another fan's audio under fan 0's model scores far above the limits.
    run_t t = {0};
    for (int i = 0; i < 4 && !(t.ev & DET_EV_TRIP); i++) {
        feed(&d, audio(1), ANDON_GV_AUDIO_LEN, false, &t);
    }
    CHECK(t.ev & DET_EV_TRIP);
    CHECK(d.trip == DET_TRIP_SCORE && detector_light(&d) == ANDON_RED);
    CHECK(t.n >= base.light.red_hold);  // never on fewer windows than red_hold
    run_t after = {0};
    feed(&d, audio(0), ANDON_GV_AUDIO_LEN, false, &after);
    CHECK(after.ev == 0 && detector_light(&d) == ANDON_RED);  // latched, and nothing scored

    // Red cuts the fan: idle. The button clears red; the restart is watched on the same
    // limits after the hold-off, without a new calibration.
    detector_set_running(&d, false, 0);
    CHECK(d.phase == DET_IDLE);
    detector_reset_light(&d);
    CHECK(detector_light(&d) == ANDON_GREEN && d.trip == DET_TRIP_NONE);
    detector_set_running(&d, true, 0);
    CHECK(d.phase == DET_WATCHING && d.have_limits);
    // A speed change is not a start: the stream and the phase carry on.
    detector_set_running(&d, true, 0);
    CHECK(d.phase == DET_WATCHING);
    detector_recalibrate(&d, 0);
    CHECK(d.phase == DET_CALIBRATING && !d.have_limits);
    detector_watch(&d, &andon_gmm[1], 1, 0);
    CHECK(d.fan == 1 && d.phase == DET_CALIBRATING && !d.have_limits);
}

static void test_a_flat_mic_trips(void) {
    static detector_t d;
    static int16_t flat[1024 + 31 * 512];
    for (size_t i = 0; i < sizeof flat / sizeof flat[0]; i++) flat[i] = (int16_t)(-7 + (i % 3 == 0) * 4);
    start(&d, 0, base.calib_windows, 0);
    run_t r = {0};
    feed(&d, flat, 1024 + 30 * 512, false, &r);  // 31 frames: one short
    CHECK(!(r.ev & DET_EV_TRIP));
    feed(&d, flat, 512, false, &r);
    CHECK((r.ev & DET_EV_TRIP) && d.trip == DET_TRIP_MIC_FLAT);
    CHECK(detector_light(&d) == ANDON_RED);

    // Peak to peak just above the limit is not flat.
    static int16_t quiet[1024 + 63 * 512];
    for (size_t i = 0; i < sizeof quiet / sizeof quiet[0]; i++) {
        quiet[i] = (int16_t)((i % 2) * (DETECTOR_FLAT_P2P + 1));
    }
    start(&d, 0, base.calib_windows, 0);
    run_t q = {0};
    feed(&d, quiet, (int)(sizeof quiet / sizeof quiet[0]), false, &q);
    CHECK(!(q.ev & DET_EV_TRIP) && q.n == 1);
}

static void test_a_silent_mic_trips(void) {
    static detector_t d;
    start(&d, 0, base.calib_windows, 0);
    CHECK(detector_tick(&d, DETECTOR_SILENT_US) == 0);
    CHECK(detector_tick(&d, DETECTOR_SILENT_US + 1) == DET_EV_TRIP);
    CHECK(d.trip == DET_TRIP_MIC_SILENT && detector_light(&d) == ANDON_RED);
    // Words arriving reset the clock.
    start(&d, 0, base.calib_windows, 0);
    uint32_t w[4] = {0};
    detector_feed(&d, w, 4, false, 200000);
    CHECK(detector_tick(&d, 200000 + DETECTOR_SILENT_US) == 0);
    CHECK(detector_tick(&d, 200001 + DETECTOR_SILENT_US) == DET_EV_TRIP);
}

static void test_idle_without_a_fan_or_when_stopped(void) {
    static detector_t d;
    detector_init(&d, &base);
    detector_set_running(&d, true, 0);  // no fan watched: nothing is checked
    CHECK(d.phase == DET_IDLE);
    run_t r = {0};
    feed(&d, audio(0), ANDON_GV_AUDIO_LEN, false, &r);
    CHECK(r.ev == 0 && detector_tick(&d, 10u * DETECTOR_SILENT_US) == 0);
    detector_watch(&d, &andon_gmm[2], 2, 0);  // watching a running fan starts the hold-off
    CHECK(d.phase == DET_HOLDOFF && d.fan == 2);
    detector_set_running(&d, false, 0);
    CHECK(d.phase == DET_IDLE && detector_tick(&d, 100u * DETECTOR_SILENT_US) == 0);
    detector_watch(&d, NULL, 0, 0);
    CHECK(d.fan == -1);
    detector_set_running(&d, true, 0);
    CHECK(d.phase == DET_IDLE);
}

static void test_an_overrun_restarts_the_stream(void) {
    static detector_t d;
    start(&d, 0, base.calib_windows, 0);
    run_t r = {0};
    feed(&d, audio(1), 32768, false, &r);  // 63 frames: no window yet
    CHECK(r.n == 0);
    feed(&d, audio(0), ANDON_GV_AUDIO_LEN, true, &r);  // the gap: those 63 frames are dropped
    CHECK(d.restarts == 1 && r.n == ANDON_GV_STREAM_WINDOWS);
    double worst = 0.0;
    for (int w = 0; w < r.n && w < ANDON_GV_STREAM_WINDOWS; w++) {
        worst = fmax(worst, rel(r.score[w], want(0, w, 0)));
    }
    CHECK(worst <= SCORE_RTOL);
}

static void test_a_bad_score_trips(void) {
    static detector_t d;
    static const float zero[1] = {0.0f};
    static const gmm_t broken = {0, ANDON_GV_N_MELS, zero, zero, zero};  // scores NaN
    detector_config_t cfg = base;
    cfg.holdoff_samples = 0;
    detector_init(&d, &cfg);
    detector_watch(&d, &broken, 0, 0);
    detector_set_running(&d, true, 0);
    run_t r = {0};
    feed(&d, audio(0), ANDON_GV_AUDIO_LEN, false, &r);
    CHECK((r.ev & DET_EV_TRIP) && d.trip == DET_TRIP_BAD_SCORE && r.n == 1);
}

static void test_names(void) {
    CHECK(strcmp(detector_phase_name(DET_WATCHING), "watching") == 0);
    CHECK(strcmp(detector_trip_name(DET_TRIP_MIC_FLAT), "mic_flat") == 0);
    CHECK(strcmp(detector_trip_name((det_trip_t)99), "invalid") == 0);
}

int main(void) {
    andon_detector_config(&base);
    printf("detector: gain shift %u, hold-off %lu samples, %d calibration windows, window hop "
           "%d frames\n",
           base.gain_shift, (unsigned long)base.holdoff_samples, base.calib_windows,
           base.window_hop);
    test_stream_scores_match_python();
    test_holdoff_skips_the_spin_up();
    test_calibrates_then_trips_and_resets();
    test_a_flat_mic_trips();
    test_a_silent_mic_trips();
    test_idle_without_a_fan_or_when_stopped();
    test_an_overrun_restarts_the_stream();
    test_a_bad_score_trips();
    test_names();
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("detector: static RAM %u bytes of state + %u bytes of frame scratch\n",
           (unsigned)sizeof(detector_t), (unsigned)(sizeof(float) * (LOGMEL_N_FFT + 2 * GMM_MAX_DIM)));
    printf("all checks passed\n");
    return 0;
}
