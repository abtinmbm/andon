// The andon light: SPC-style limits on the anomaly score and the green/yellow/red state
// machine, as andon/light.py does it. Plain C with no Pico SDK calls, so it also builds on
// the PC, where firmware/test/test_light.c checks it against vectors from andon.light.
//
// Limits are fitted like a control chart on normal running: warn = mean + warn_sigma * sd,
// control = mean + control_sigma * sd (sample sd, as numpy's ddof=1). The light never acts
// on one window: yellow needs warn_on windows in a row above the warning limit and goes off
// after warn_off at or below it; red needs red_hold in a row above the control limit and is
// latched until light_reset (the button).
#ifndef ANDON_LIGHT_H
#define ANDON_LIGHT_H

#include <stdbool.h>

typedef enum {
    ANDON_GREEN = 0,
    ANDON_YELLOW = 1,
    ANDON_RED = 2,
} andon_light_t;

const char *andon_light_name(andon_light_t light);

typedef struct {
    int warn_on;   // above the warning limit: green -> yellow
    int warn_off;  // at or below the warning limit: yellow -> green
    int red_hold;  // above the control limit: -> red, latched
} light_settings_t;

typedef struct {
    double center;
    double sigma;
    double warn;
    double control;
} light_limits_t;

// Running mean and variance of the calibration scores (Welford's method, in double, so
// hundreds of float scores lose nothing to rounding).
typedef struct {
    int n;
    double mean;
    double m2;  // sum of squared differences from the mean
} light_calib_t;

void light_calib_reset(light_calib_t *c);
void light_calib_add(light_calib_t *c, double score);

// Fits the limits on what was added. Returns false, leaving *out alone, with fewer than two
// scores, or when they don't give a usable chart (non-finite, or a sigma of zero: every
// score the same, which no live microphone gives).
bool light_calib_limits(const light_calib_t *c, double warn_sigma, double control_sigma,
                        light_limits_t *out);

typedef struct {
    andon_light_t state;
    int above_warn;
    int below_warn;
    int above_control;
} light_t;

// Green, with every run count cleared: what the reset button does.
void light_reset(light_t *l);

// The light after one more window score. Red stays red. A NaN score counts as at or below
// both limits, as in Python; the detector treats a non-finite score as a fault itself.
andon_light_t light_update(light_t *l, const light_limits_t *lim, const light_settings_t *s,
                           float score);

// Straight to red, for a fault the scores can't show (a dead microphone).
void light_trip(light_t *l);

#endif
