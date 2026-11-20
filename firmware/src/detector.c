#include "detector.h"

#include <math.h>
#include <string.h>

#include "mic_stream.h"

// Scratch for one frame; static, as the loop never allocates. One detector per board.
static float frame_f[LOGMEL_N_FFT];
static float mel[GMM_MAX_DIM];
static float mean[GMM_MAX_DIM];

void detector_init(detector_t *d, const detector_config_t *cfg) {
    logmel_init();  // builds the same tables every time, so calling it again is harmless
    memset(d, 0, sizeof *d);
    d->cfg = *cfg;
    d->fan = -1;
    d->phase = DET_IDLE;
    light_reset(&d->light);
}

// The stream starts over: no frame or window carries audio from before this point.
static void restart_stream(detector_t *d) {
    d->fill = 0;
    d->flat_frames = 0;
    gmm_window_reset(&d->window, d->cfg.window_hop, d->gmm->n_dim);
}

static void begin_listening(detector_t *d) {
    restart_stream(d);
    if (d->have_limits) {
        d->phase = DET_WATCHING;
    } else {
        light_calib_reset(&d->calib);
        d->phase = DET_CALIBRATING;
    }
}

static void begin_holdoff(detector_t *d, uint64_t now_us) {
    d->last_word_us = now_us;  // the silence check counts from the start
    d->holdoff_left = d->cfg.holdoff_samples;
    d->phase = DET_HOLDOFF;
    if (d->holdoff_left == 0) begin_listening(d);
}

void detector_watch(detector_t *d, const gmm_t *gmm, int fan, uint64_t now_us) {
    d->gmm = gmm;
    d->fan = gmm ? fan : -1;
    d->have_limits = false;
    d->phase = DET_IDLE;
    if (gmm && d->running) begin_holdoff(d, now_us);
}

void detector_recalibrate(detector_t *d, uint64_t now_us) {
    d->have_limits = false;
    if (d->gmm && d->running) begin_holdoff(d, now_us);
}

void detector_set_running(detector_t *d, bool running, uint64_t now_us) {
    if (running && !d->running && d->gmm) begin_holdoff(d, now_us);
    if (!running) d->phase = DET_IDLE;
    d->running = running;
}

void detector_trip(detector_t *d, det_trip_t why) {
    light_trip(&d->light);
    d->trip = why;
}

void detector_reset_light(detector_t *d) {
    light_reset(&d->light);
    d->trip = DET_TRIP_NONE;
}

andon_light_t detector_light(const detector_t *d) { return d->light.state; }

// One complete frame in d->buf: the flat-signal check, then the front end, the window and,
// when a window completes, the score.
static unsigned process_frame(detector_t *d) {
    int16_t lo = d->buf[0], hi = d->buf[0];
    for (int i = 0; i < LOGMEL_N_FFT; i++) {
        if (d->buf[i] < lo) lo = d->buf[i];
        if (d->buf[i] > hi) hi = d->buf[i];
        frame_f[i] = (float)d->buf[i] / 32768.0f;
    }
    d->flat_frames = (hi - lo <= DETECTOR_FLAT_P2P) ? d->flat_frames + 1 : 0;
    if (d->flat_frames >= DETECTOR_FLAT_FRAMES) {
        detector_trip(d, DET_TRIP_MIC_FLAT);
        return DET_EV_TRIP;
    }
    logmel_frame(d->cfg.bank, frame_f, mel);
    if (!gmm_window_push(&d->window, mel, mean)) return 0;

    const float score = gmm_score(d->gmm, mean);
    d->last_score = score;
    d->windows++;
    unsigned ev = DET_EV_WINDOW;
    if (!isfinite(score)) {
        detector_trip(d, DET_TRIP_BAD_SCORE);
        return ev | DET_EV_TRIP;
    }
    if (d->phase == DET_CALIBRATING) {
        light_calib_add(&d->calib, score);
        if (d->calib.n >= d->cfg.calib_windows) {
            if (!light_calib_limits(&d->calib, d->cfg.warn_sigma, d->cfg.control_sigma,
                                    &d->limits)) {
                detector_trip(d, DET_TRIP_CALIBRATION);
                return ev | DET_EV_TRIP;
            }
            d->have_limits = true;
            d->phase = DET_WATCHING;
            ev |= DET_EV_CALIBRATED;
        }
    } else if (light_update(&d->light, &d->limits, &d->cfg.light, score) == ANDON_RED) {
        d->trip = DET_TRIP_SCORE;
        ev |= DET_EV_TRIP;
    }
    return ev;
}

unsigned detector_feed(detector_t *d, const uint32_t *words, uint32_t n, bool overrun,
                       uint64_t now_us) {
    if (n > 0) d->last_word_us = now_us;
    if (d->phase == DET_IDLE || d->light.state == ANDON_RED) return 0;
    if (overrun && d->phase != DET_HOLDOFF) {
        restart_stream(d);
        d->restarts++;
    }
    if (n > DETECTOR_MAX_WORDS) n = DETECTOR_MAX_WORDS;  // see DETECTOR_MAX_WORDS
    unsigned ev = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (d->phase == DET_HOLDOFF) {
            if (--d->holdoff_left == 0) begin_listening(d);
            continue;
        }
        bool clipped = false;
        d->buf[d->fill++] = mic_word_to_sample(words[i], d->cfg.gain_shift, &clipped);
        if (d->fill < LOGMEL_N_FFT) continue;
        ev |= process_frame(d);
        // The next frame starts DETECTOR_HOP samples later and overlaps this one.
        memmove(d->buf, d->buf + DETECTOR_HOP, (LOGMEL_N_FFT - DETECTOR_HOP) * sizeof d->buf[0]);
        d->fill = LOGMEL_N_FFT - DETECTOR_HOP;
        if (ev & DET_EV_TRIP) break;
    }
    return ev;
}

unsigned detector_tick(detector_t *d, uint64_t now_us) {
    if (d->phase == DET_IDLE || d->light.state == ANDON_RED) return 0;
    if (now_us - d->last_word_us > DETECTOR_SILENT_US) {
        detector_trip(d, DET_TRIP_MIC_SILENT);
        return DET_EV_TRIP;
    }
    return 0;
}

const char *detector_phase_name(det_phase_t phase) {
    switch (phase) {
    case DET_IDLE:
        return "idle";
    case DET_HOLDOFF:
        return "holdoff";
    case DET_CALIBRATING:
        return "calibrating";
    case DET_WATCHING:
        return "watching";
    default:
        return "invalid";
    }
}

const char *detector_trip_name(det_trip_t trip) {
    switch (trip) {
    case DET_TRIP_NONE:
        return "none";
    case DET_TRIP_SCORE:
        return "score";
    case DET_TRIP_MIC_SILENT:
        return "mic_silent";
    case DET_TRIP_MIC_FLAT:
        return "mic_flat";
    case DET_TRIP_BAD_SCORE:
        return "bad_score";
    case DET_TRIP_CALIBRATION:
        return "calibration";
    case DET_TRIP_WATCHDOG:
        return "watchdog";
    default:
        return "invalid";
    }
}
