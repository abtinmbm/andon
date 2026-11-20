#include "andon_light.h"

#include <math.h>

const char *andon_light_name(andon_light_t light) {
    switch (light) {
    case ANDON_GREEN:
        return "green";
    case ANDON_YELLOW:
        return "yellow";
    case ANDON_RED:
        return "red";
    default:
        return "invalid";
    }
}

void light_calib_reset(light_calib_t *c) {
    c->n = 0;
    c->mean = 0.0;
    c->m2 = 0.0;
}

void light_calib_add(light_calib_t *c, double score) {
    c->n++;
    const double delta = score - c->mean;
    c->mean += delta / c->n;
    c->m2 += delta * (score - c->mean);
}

bool light_calib_limits(const light_calib_t *c, double warn_sigma, double control_sigma,
                        light_limits_t *out) {
    if (c->n < 2) return false;
    const double sigma = sqrt(c->m2 / (c->n - 1));
    if (!isfinite(c->mean) || !isfinite(sigma) || sigma <= 0.0) return false;
    out->center = c->mean;
    out->sigma = sigma;
    out->warn = c->mean + warn_sigma * sigma;
    out->control = c->mean + control_sigma * sigma;
    return true;
}

void light_reset(light_t *l) {
    l->state = ANDON_GREEN;
    l->above_warn = 0;
    l->below_warn = 0;
    l->above_control = 0;
}

andon_light_t light_update(light_t *l, const light_limits_t *lim, const light_settings_t *s,
                           float score) {
    if (l->state == ANDON_RED) return l->state;
    // Compared in double, as Python compares the float32 score with its float64 limits.
    const double x = (double)score;
    const bool above_warn = x > lim->warn;
    l->above_warn = above_warn ? l->above_warn + 1 : 0;
    l->below_warn = above_warn ? 0 : l->below_warn + 1;
    l->above_control = x > lim->control ? l->above_control + 1 : 0;

    if (l->above_control >= s->red_hold) {
        l->state = ANDON_RED;
    } else if (l->state == ANDON_GREEN && l->above_warn >= s->warn_on) {
        l->state = ANDON_YELLOW;
    } else if (l->state == ANDON_YELLOW && l->below_warn >= s->warn_off) {
        l->state = ANDON_GREEN;
    }
    return l->state;
}

void light_trip(light_t *l) { l->state = ANDON_RED; }
