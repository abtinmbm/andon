#include "fan.h"

void fan_init(fan_t *fan) {
    fan->want_on = false;
    fan->speed_pct = FAN_FULL_SPEED_PCT;
}

fan_result_t fan_turn_on(fan_t *fan, andon_light_t light) {
    if (light == ANDON_RED) {
        return FAN_REFUSED_RED;
    }
    fan->want_on = true;
    return FAN_OK;
}

void fan_turn_off(fan_t *fan) { fan->want_on = false; }

fan_result_t fan_set_speed(fan_t *fan, int speed_pct) {
    if (speed_pct < 0 || speed_pct > (int)FAN_FULL_SPEED_PCT) {
        return FAN_BAD_SPEED;
    }
    fan->speed_pct = (uint8_t)speed_pct;
    return FAN_OK;
}

uint8_t fan_duty_pct(const fan_t *fan, andon_light_t light) {
    // Anything but green or yellow (red, or a corrupted value) stops the fan.
    if (light != ANDON_GREEN && light != ANDON_YELLOW) {
        return 0;
    }
    if (!fan->want_on || fan->speed_pct > FAN_FULL_SPEED_PCT) {
        return 0;
    }
    return fan->speed_pct;
}

void fan_output_init(fan_output_t *out) {
    out->duty_pct = 0;
    out->kicking = false;
    out->kick_start_ms = 0;
}

uint8_t fan_output_update(fan_output_t *out, uint8_t want_pct, uint32_t now_ms) {
    if (want_pct == 0) {
        // The kill switch: off is immediate and cancels a kick in progress.
        out->kicking = false;
        out->duty_pct = 0;
        return 0;
    }
    if (out->duty_pct == 0 && want_pct < FAN_FULL_SPEED_PCT) {
        out->kicking = true;
        out->kick_start_ms = now_ms;
    }
    if (out->kicking && (int32_t)(now_ms - out->kick_start_ms) >= (int32_t)FAN_KICK_MS) {
        out->kicking = false;
    }
    out->duty_pct = out->kicking ? FAN_FULL_SPEED_PCT : want_pct;
    return out->duty_pct;
}

uint32_t fan_pwm_level(uint8_t duty_pct, uint32_t wrap) {
    if (duty_pct >= FAN_FULL_SPEED_PCT) {
        return wrap + 1u;
    }
    // 64-bit so a large wrap can't overflow; rounds to the nearest tick.
    uint64_t period = (uint64_t)wrap + 1u;
    return (uint32_t)((period * duty_pct + FAN_FULL_SPEED_PCT / 2u) / FAN_FULL_SPEED_PCT);
}

const char *fan_result_name(fan_result_t result) {
    switch (result) {
    case FAN_OK:
        return "ok";
    case FAN_REFUSED_RED:
        return "refused_red";
    case FAN_BAD_SPEED:
        return "bad_speed";
    default:
        return "invalid";
    }
}
