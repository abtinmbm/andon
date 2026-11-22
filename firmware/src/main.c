// Andon firmware: the detector watches the fan by ear and stops it on red (jidoka).
//
// H0 added the light and the button, H1 the fan, H2 the microphone, H6 the detector
// (detector.h): mic -> log-mel -> the watched fan's GMM -> the andon light. Red cuts the fan
// and only the button (red -> green) clears it; a fan that was running then restarts. The
// button does nothing on green or yellow. Serial commands, one per line:
//   fan on | fan off | fan speed <0-100> | status | rec start <gain 0-8> | rec stop |
//   watch <machine id> | watch none | calibrate
// "watch" picks the fan's model (and clears the limits); until it is given the fan runs
// unwatched. The limits are fitted on the first 10 minutes the watched fan runs after its
// hold-off, kept in RAM, and fitted again after "calibrate", "watch" or a reboot.
// Every command is answered with one "ack" line carrying the Pico's timer (microseconds) when
// the command arrived, when the PWM was changed and when the gate pin was read back at its
// new level, so the time from a command to the fan stopping can be measured on the board's
// own clock (andon.fan_latency, andon.jidoka). Every scored window is one "score" line; a red
// is a "trip" line, then the "fan update" that cut the fan. Starting from stopped at a part
// speed runs a 1 s full-speed kick first (see fan.h). The fan is off at boot.
//
// Safety: the light and the fan are set before anything is printed, and nothing printed
// ever waits on the PC (say.h). A hardware watchdog reboots the board if a pass of the loop
// hangs for WATCHDOG_MS; the gate's pull-down and the PWM's 0 % at boot keep the fan off, and
// the board comes back red (trip=watchdog), so someone has to look before it runs again.
//
// Wiring (Pico 2 physical pin in brackets):
//   GP10 [14] -> 220 ohm -> green LED  -> GND [13]
//   GP11 [15] -> 220 ohm -> yellow LED -> GND [13]
//   GP12 [16] -> 220 ohm -> red LED    -> GND [13]
//   GP14 [19] -> pushbutton -> GND [18]   (internal pull-up, so pressed reads low)
//   GP16 [21] -> 220 ohm -> IRLZ44N gate; 5.1 kohm gate -> GND keeps the fan off until the
//              pin is driven (8.2 kohm or less: RP2350 erratum E9 lets an undriven pin float
//              near 2 V against a weaker pull-down). Fan + to VBUS [40] (5 V from USB),
//              fan - to the drain, source to GND [23]; 1N4007 across the fan, stripe to
//              +5 V. The pin only drives the gate, never the fan.
//   GP18 [24] -> mic BCLK, GP19 [25] -> mic LRCLK (WS), GP20 [26] <- mic DOUT; mic 3V to
//              3V3(OUT) [36], GND to GND [38], SEL to GND (left channel). See mic.h.
//   GP25 is the onboard LED on the Pico 2 (no wiring).
#include <stdio.h>
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"

#include "andon_light.h"
#include "andon_version.h"
#include "button.h"
#include "command.h"
#include "detector.h"
#include "fan.h"
#include "gmm_params.h"
#include "mic.h"
#include "mic_stream.h"
#include "recorder.h"
#include "say.h"

#define PIN_GREEN 10
#define PIN_YELLOW 11
#define PIN_RED 12
#define PIN_BUTTON 14
#define PIN_FAN 16
#define FAN_PWM_HZ 25000u  // above hearing, so the PWM itself doesn't whine
#define HEARTBEAT_MS 500u
// The boot line is printed before a terminal can open the port, so repeat the state.
#define STATUS_MS 5000u
// How long to wait for the gate pin to read back at its new level. The PWM takes a new
// level at the end of its period (40 us at 25 kHz), so this is generous.
#define READBACK_TIMEOUT_US 1000u
#define PWM_PERIOD_US (1000000u / FAN_PWM_HZ)
// The slowest pass is a frame plus a score, about 9 ms; a whole second means a hang.
#define WATCHDOG_MS 1000u

static const uint LIGHT_PINS[] = {PIN_GREEN, PIN_YELLOW, PIN_RED};
static uint32_t fan_pwm_wrap;
static detector_t det;
static uint32_t mic_words[DETECTOR_MAX_WORDS];

static void init_pins(void) {
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    for (size_t i = 0; i < count_of(LIGHT_PINS); i++) {
        gpio_init(LIGHT_PINS[i]);
        gpio_set_dir(LIGHT_PINS[i], GPIO_OUT);
    }
    gpio_init(PIN_BUTTON);
    gpio_set_dir(PIN_BUTTON, GPIO_IN);
    gpio_pull_up(PIN_BUTTON);
}

// 25 kHz PWM on the gate, starting at 0 % (fan off) until the main loop sets the duty.
static void init_fan_pwm(void) {
    uint slice = pwm_gpio_to_slice_num(PIN_FAN);
    fan_pwm_wrap = clock_get_hz(clk_sys) / FAN_PWM_HZ - 1u;  // 5999 at 150 MHz
    // The PWM counter is 16 bits and 100 % needs wrap + 1 to fit; stop rather than truncate.
    if (fan_pwm_wrap >= 0xFFFFu) {
        panic("fan PWM wrap %lu does not fit 16 bits", (unsigned long)fan_pwm_wrap);
    }
    pwm_config cfg = pwm_get_default_config();
    pwm_config_set_clkdiv_int(&cfg, 1);
    pwm_config_set_wrap(&cfg, (uint16_t)fan_pwm_wrap);
    pwm_init(slice, &cfg, false);
    pwm_set_gpio_level(PIN_FAN, 0);
    gpio_set_function(PIN_FAN, GPIO_FUNC_PWM);
    pwm_set_enabled(slice, true);
}

static bool button_is_down(void) { return !gpio_get(PIN_BUTTON); }

static void show_light(andon_light_t light) {
    for (size_t i = 0; i < count_of(LIGHT_PINS); i++) {
        gpio_put(LIGHT_PINS[i], (size_t)light == i);
    }
}

static void heartbeat(uint32_t now_ms) { gpio_put(PICO_DEFAULT_LED_PIN, (now_ms / HEARTBEAT_MS) % 2); }

typedef struct {
    uint8_t duty_pct;
    uint64_t t_set_us;
    int64_t t_pin_us;  // -1 if not confirmed (partial duty cycles toggle, so there's no level)
    int pin;           // gate pin level read back, or -1
} fan_drive_t;

// Drive the gate at a new duty cycle; at 0 % and 100 % wait for the pin to read back.
static fan_drive_t drive_fan(uint8_t duty_pct) {
    fan_drive_t d = {duty_pct, 0, -1, -1};
    pwm_set_gpio_level(PIN_FAN, (uint16_t)fan_pwm_level(duty_pct, fan_pwm_wrap));
    d.t_set_us = time_us_64();
    if (duty_pct == 0 || duty_pct >= FAN_FULL_SPEED_PCT) {
        bool want_high = duty_pct != 0;
        while (time_us_64() - d.t_set_us < READBACK_TIMEOUT_US) {
            // The new level takes effect at the end of the current PWM period. A low pin
            // after "off" stays low from then on, but a high pin after "on" may still be the
            // old duty cycle's high phase, so only trust it once a full period has passed.
            if (want_high && time_us_64() - d.t_set_us <= PWM_PERIOD_US) {
                continue;
            }
            if (gpio_get(PIN_FAN) == want_high) {
                d.t_pin_us = (int64_t)time_us_64();
                d.pin = want_high;
                break;
            }
        }
    }
    return d;
}

static const char *watched_id(void) { return det.fan >= 0 ? andon_gmm_fan_ids[det.fan] : "none"; }

// The fields every ack and fan update ends with.
static void say_drive(const char *head, const fan_drive_t *d, andon_light_t light,
                      const char *tail) {
    say("%st_set_us=%llu t_pin_us=%lld pin=%d duty=%u light=%s%s\n", head,
        (unsigned long long)d->t_set_us, (long long)d->t_pin_us, d->pin, d->duty_pct,
        andon_light_name(light), tail);
}

// Work out the duty the fan should have now and drive it if it changed. Called before
// anything is printed, so printing can never delay cutting the fan.
static bool update_fan(const fan_t *fan, fan_output_t *out, andon_light_t light,
                       uint32_t now_ms, uint8_t *driven_pct, fan_drive_t *d) {
    uint8_t duty = fan_output_update(out, fan_duty_pct(fan, light), now_ms);
    if (duty == *driven_pct) {
        return false;
    }
    *driven_pct = duty;
    *d = drive_fan(duty);
    return true;
}

static void say_recorder(void) {
    const recorder_stats_t *r = recorder_stats();
    say("rec streaming=%d gain=%u frames_sent=%lu frames_dropped=%lu overruns=%lu "
        "frames_clipped=%lu\n",
        r->streaming, r->gain_shift, (unsigned long)r->frames_sent,
        (unsigned long)r->frames_dropped, (unsigned long)r->overruns,
        (unsigned long)r->frames_clipped);
}

static void say_detector(void) {
    say("detector watch=%s phase=%s light=%s trip=%s windows=%lu calib=%d/%d restarts=%lu "
        "say_dropped=%lu\n",
        watched_id(), detector_phase_name(det.phase), andon_light_name(detector_light(&det)),
        detector_trip_name(det.trip), (unsigned long)det.windows, det.calib.n,
        det.cfg.calib_windows, (unsigned long)det.restarts, (unsigned long)say_dropped());
}

// The limits in RAM. Also printed after "status", so a tool started after the calibration
// (H7's trial series) still has the limits it is judged against in its own capture.
static void say_limits(uint64_t now_us) {
    say("limits t_us=%llu fan=%s n=%d center=%.4f sigma=%.4f warn=%.4f control=%.4f\n",
        (unsigned long long)now_us, watched_id(), det.calib.n, det.limits.center,
        det.limits.sigma, det.limits.warn, det.limits.control);
}

// "watch <id>": the fan's index in the exported models, or "none". False if unknown.
static bool watch(const char *name, uint64_t now_us) {
    if (strcmp(name, "none") == 0) {
        detector_watch(&det, NULL, 0, now_us);
        return true;
    }
    for (int i = 0; i < andon_gmm_n_fans; i++) {
        if (strcmp(name, andon_gmm_fan_ids[i]) == 0) {
            detector_watch(&det, &andon_gmm[i], i, now_us);
            return true;
        }
    }
    return false;
}

static const char *run_command(fan_t *fan, command_t cmd, andon_light_t light, uint64_t now_us) {
    switch (cmd.kind) {
    case CMD_FAN_ON:
        return fan_result_name(fan_turn_on(fan, light));
    case CMD_FAN_OFF:
        fan_turn_off(fan);
        return "ok";
    case CMD_FAN_SPEED:
        return fan_result_name(fan_set_speed(fan, cmd.arg));
    case CMD_REC_START:
        recorder_start((unsigned)cmd.arg);
        return "ok";
    case CMD_REC_STOP:
        recorder_stop();
        return "ok";
    case CMD_WATCH:
        return watch(cmd.name, now_us) ? "ok" : "unknown_fan";
    case CMD_CALIBRATE:
        if (det.fan < 0) return "no_fan";
        detector_recalibrate(&det, now_us);
        return "ok";
    default:
        return "ok";  // status, and invalid lines (reported by the caller)
    }
}

// Read every character waiting on USB serial; handle each complete line.
static void poll_serial(command_reader_t *reader, fan_t *fan, fan_output_t *out,
                        uint8_t *driven_pct) {
    int c;
    while ((c = getchar_timeout_us(0)) != PICO_ERROR_TIMEOUT) {
        if (!command_reader_feed(reader, (char)c)) {
            continue;
        }
        uint64_t t_rx_us = time_us_64();
        command_t cmd = command_parse(reader->line);
        if (cmd.kind == CMD_INVALID) {
            say("error: unknown command \"%s\" (fan on | fan off | fan speed <0-100> | status | "
                "rec start <gain 0-8> | rec stop | watch <id> | watch none | calibrate)\n",
                reader->line);
            continue;
        }
        andon_light_t light = detector_light(&det);
        const char *result = run_command(fan, cmd, light, t_rx_us);
        // Always drive and read back, even when nothing changed, so every ack is a
        // measurement of the pin's current level.
        uint32_t now_ms = to_ms_since_boot(get_absolute_time());
        *driven_pct = fan_output_update(out, fan_duty_pct(fan, light), now_ms);
        fan_drive_t d = drive_fan(*driven_pct);
        detector_set_running(&det, *driven_pct > 0, time_us_64());
        if (cmd.kind == CMD_STATUS) {
            say("build=%s model=%s\n", ANDON_BUILD, andon_gmm_model_source);
        }
        char head[96];
        snprintf(head, sizeof head, "ack cmd=\"%s\" result=%s t_rx_us=%llu ", reader->line,
                 result, (unsigned long long)t_rx_us);
        char tail[96];
        snprintf(tail, sizeof tail, " speed=%u want_on=%d kick=%d watch=%s phase=%s",
                 fan->speed_pct, fan->want_on, out->kicking, watched_id(),
                 detector_phase_name(det.phase));
        say_drive(head, &d, light, tail);
        if (cmd.kind == CMD_STATUS) {
            say_detector();
            if (det.have_limits) {
                say_limits(t_rx_us);
            }
        }
        if (cmd.kind == CMD_STATUS || cmd.kind == CMD_REC_START || cmd.kind == CMD_REC_STOP) {
            say_recorder();
        }
        if (strcmp(result, "refused_red") == 0) {
            say("  the light is red: press the button to reset; the fan restarts if it was "
                "running\n");
        }
    }
}

// Read what the mic delivered since the last pass and give it to the recorder and the
// detector. Returns the detector's DET_EV_* bits.
static unsigned listen(uint64_t now_us) {
    bool overrun = false;
    uint32_t n = mic_read(mic_words, DETECTOR_MAX_WORDS, &overrun);
    recorder_feed(mic_words, n, overrun);
    unsigned ev = detector_feed(&det, mic_words, n, overrun, now_us);
    return ev | detector_tick(&det, now_us);
}

// Printed after the fan has been set, so a print never delays a stop.
static void report(unsigned ev, uint64_t now_us) {
    if (ev & DET_EV_WINDOW) {
        say("score t_us=%llu n=%lu fan=%s phase=%s s=%.4f light=%s\n",
            (unsigned long long)now_us, (unsigned long)det.windows, watched_id(),
            detector_phase_name(det.phase), (double)det.last_score,
            andon_light_name(detector_light(&det)));
    }
    if (ev & DET_EV_CALIBRATED) {
        say_limits(now_us);
    }
    if (ev & DET_EV_TRIP) {
        say("trip t_us=%llu reason=%s fan=%s n=%lu s=%.4f\n", (unsigned long long)now_us,
            detector_trip_name(det.trip), watched_id(), (unsigned long)det.windows,
            (double)det.last_score);
    }
}

int main(void) {
    // A reboot by the watchdog means the loop hung: come back red, with the fan off.
    const bool watchdog_reboot = watchdog_enable_caused_reboot();
    stdio_init_all();
    init_pins();
    init_fan_pwm();
    mic_init();
    recorder_init();
    detector_config_t cfg;
    andon_detector_config(&cfg);
    detector_init(&det, &cfg);
    if (watchdog_reboot) {
        detector_trip(&det, DET_TRIP_WATCHDOG);
    }

    fan_t fan;
    fan_init(&fan);
    fan_output_t fan_out;
    fan_output_init(&fan_out);
    button_t button;
    button_init(&button, button_is_down(), to_ms_since_boot(get_absolute_time()));
    command_reader_t reader;
    command_reader_init(&reader);
    andon_light_t shown = detector_light(&det);
    show_light(shown);
    uint8_t duty_pct = 0;  // the PWM starts at 0 % and the fan is off at boot
    watchdog_enable(WATCHDOG_MS, true);  // true: paused while a debugger holds the cores
    say("andon h6 build=%s: light=%s trip=%s fan duty=%u pwm=%lu Hz wrap=%lu\n", ANDON_BUILD,
        andon_light_name(shown), detector_trip_name(det.trip), duty_pct,
        (unsigned long)FAN_PWM_HZ, (unsigned long)fan_pwm_wrap);
    say("mic i2s rate=%lu Hz bclk=GP%u ws=GP%u dout=GP%u ring=%lu B frame=%lu B gain=%u\n",
        (unsigned long)MIC_SAMPLE_RATE_HZ, MIC_PIN_BCLK, MIC_PIN_BCLK + 1u, MIC_PIN_DOUT,
        (unsigned long)mic_ring_bytes(), (unsigned long)MIC_FRAME_BYTES, cfg.gain_shift);
    say("detector ram=%lu B logmel ram=%lu B holdoff=%lu samples calib=%d windows model=%s\n",
        (unsigned long)sizeof det, (unsigned long)logmel_ram_bytes(),
        (unsigned long)cfg.holdoff_samples, cfg.calib_windows, andon_gmm_model_source);
    uint32_t next_status_ms = STATUS_MS;

    while (true) {
        watchdog_update();
        uint64_t now_us = time_us_64();
        uint32_t now_ms = to_ms_since_boot(get_absolute_time());
        heartbeat(now_ms);
        // The button only clears red; a running fan is restarted by update_fan below.
        bool reset = button_pressed(&button, button_is_down(), now_ms) &&
                     detector_light(&det) == ANDON_RED;
        if (reset) {
            detector_reset_light(&det);
        }
        unsigned ev = listen(now_us);
        // Safety first: the light and the fan are set before anything is printed.
        andon_light_t light = detector_light(&det);
        bool light_changed = light != shown;
        if (light_changed) {
            show_light(light);
            shown = light;
        }
        fan_drive_t d;
        bool fan_changed = update_fan(&fan, &fan_out, light, now_ms, &duty_pct, &d);
        detector_set_running(&det, duty_pct > 0, now_us);

        report(ev, now_us);
        if (light_changed) {
            say("light t_us=%llu light=%s trip=%s reset=%d\n", (unsigned long long)now_us,
                andon_light_name(light), detector_trip_name(det.trip), reset);
        }
        if (fan_changed) {
            char tail[48];
            snprintf(tail, sizeof tail, " kick=%d trip=%s", fan_out.kicking,
                     detector_trip_name(det.trip));
            say_drive("fan update ", &d, light, tail);
        }
        poll_serial(&reader, &fan, &fan_out, &duty_pct);
        // Signed difference, so the check survives the 32-bit ms counter wrapping (~49.7 days).
        if ((int32_t)(now_ms - next_status_ms) >= 0) {
            say("t=%lu ms alive build=%s light=%s fan duty=%u watch=%s phase=%s\n",
                (unsigned long)now_ms, ANDON_BUILD, andon_light_name(light), duty_pct,
                watched_id(), detector_phase_name(det.phase));
            next_status_ms += STATUS_MS;
        }
        sleep_ms(1);
    }
}
