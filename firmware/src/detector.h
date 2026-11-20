// The on-board detector (Phase H6): microphone words in, the andon light out. Plain C with
// no Pico SDK calls, so firmware/test/test_detector.c runs it on the PC against Python.
//
// The pipeline, all on one core (logmel_frame is not reentrant):
//   I2S words -> int16 at the recording gain -> 1024-sample frames every 512 samples, with no
//   padding -> logmel_frame -> gmm_window_push (a window of 2 * hop frames every hop) ->
//   gmm_score under the watched fan's GMM -> the light.
//
// Phases, while the watched fan runs:
//   holdoff      the first holdoff_samples after a start are skipped (spin-up and the kick
//                start are not normal running; H2 recorded nothing in them either)
//   calibrating  the first calib_windows scores fit the limits, as H3 fits them on its first
//                10 minutes of normal running; the light stays green meanwhile
//   watching     each score updates the light; red trips the detector
// When the fan stops (turned off, or cut by red) the detector is idle and the stream starts
// over at the next start. A speed change is not a start: the stream carries on, which is
// how the underspeed fault is caught.
//
// The detector also stops the fan when it can no longer hear: no microphone words for
// DETECTOR_SILENT_US, or DETECTOR_FLAT_FRAMES frames in a row with a peak-to-peak below
// DETECTOR_FLAT_P2P (a stuck or disconnected data line reads a constant). Red latches until
// detector_reset_light (the button).
#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdbool.h>
#include <stdint.h>

#include "andon_light.h"
#include "gmm.h"
#include "logmel.h"

#define DETECTOR_HOP LOGMEL_HOP
// At most this many words per detector_feed call: then a call completes at most one frame,
// so at most one window, and its score is in last_score.
#define DETECTOR_MAX_WORDS 512u
#define DETECTOR_SILENT_US 250000u
// Normal running at the recording gain is about -39 dBFS, hundreds of LSB peak to peak;
// the mic's own noise floor alone is far above 4.
#define DETECTOR_FLAT_P2P 4
#define DETECTOR_FLAT_FRAMES 32  // about 1 s of frames

typedef struct {
    const mel_bank_t *bank;
    unsigned gain_shift;
    int window_hop;  // frames; a window is twice this (gmm_window_t)
    uint32_t holdoff_samples;
    int calib_windows;
    double warn_sigma;
    double control_sigma;
    light_settings_t light;
} detector_config_t;

typedef enum {
    DET_IDLE = 0,  // no fan watched, or the fan is stopped
    DET_HOLDOFF,
    DET_CALIBRATING,
    DET_WATCHING,
} det_phase_t;

typedef enum {
    DET_TRIP_NONE = 0,
    DET_TRIP_SCORE,        // red_hold windows in a row above the control limit
    DET_TRIP_MIC_SILENT,   // no words from the microphone
    DET_TRIP_MIC_FLAT,     // a constant signal
    DET_TRIP_BAD_SCORE,    // a score that is not a finite number
    DET_TRIP_CALIBRATION,  // the calibration scores gave no usable limits
    DET_TRIP_WATCHDOG,     // set by main after a watchdog reboot
} det_trip_t;

// detector_feed's and detector_tick's return value: a bit for each thing that happened.
#define DET_EV_WINDOW 1u      // a window was scored: last_score, windows
#define DET_EV_CALIBRATED 2u  // the limits were just fitted: limits
#define DET_EV_TRIP 4u        // the light just went red: trip

typedef struct {
    detector_config_t cfg;
    const gmm_t *gmm;  // the watched fan's model, or NULL
    int fan;           // its index, or -1
    bool running;
    det_phase_t phase;
    uint32_t holdoff_left;
    int16_t buf[LOGMEL_N_FFT];
    int fill;
    int flat_frames;
    uint64_t last_word_us;
    gmm_window_t window;
    light_calib_t calib;
    bool have_limits;
    light_limits_t limits;
    light_t light;
    det_trip_t trip;
    float last_score;
    uint32_t windows;   // scored since boot
    uint32_t restarts;  // stream restarts after the mic ring overran
} detector_t;

void detector_init(detector_t *d, const detector_config_t *cfg);

// Watch fan `fan` with `gmm` (NULL: watch nothing). The limits belong to one fan on one
// mount, so they are cleared, and a running fan starts over from the hold-off.
void detector_watch(detector_t *d, const gmm_t *gmm, int fan, uint64_t now_us);

// Clear the limits (after a remount); a running fan starts over from the hold-off.
void detector_recalibrate(detector_t *d, uint64_t now_us);

// Call on every loop pass with whether the fan is being driven. Stopped -> running starts
// the hold-off; running -> stopped makes the detector idle.
void detector_set_running(detector_t *d, bool running, uint64_t now_us);

// Feed up to DETECTOR_MAX_WORDS microphone words. overrun: words before these were lost,
// so the stream starts over (no frame or window spans the gap). Returns DET_EV_* bits;
// after DET_EV_TRIP it stops reading, so the caller can cut the fan at once.
unsigned detector_feed(detector_t *d, const uint32_t *words, uint32_t n, bool overrun,
                       uint64_t now_us);

// The silent-microphone check; call on every loop pass. Returns DET_EV_TRIP or 0.
unsigned detector_tick(detector_t *d, uint64_t now_us);

// Force red with a reason (DET_TRIP_WATCHDOG after a watchdog reboot).
void detector_trip(detector_t *d, det_trip_t why);

// The reset button: red -> green. The limits are kept; the restarted fan goes through the
// hold-off and is watched again without a new calibration.
void detector_reset_light(detector_t *d);

andon_light_t detector_light(const detector_t *d);
const char *detector_phase_name(det_phase_t phase);
const char *detector_trip_name(det_trip_t trip);

#endif  // DETECTOR_H
