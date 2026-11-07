// Andon firmware: the andon light, the reset button and the fan's kill switch.
//
// H0 added the light and the button; H1 adds the fan. The button steps the light green ->
// yellow -> red -> green (until the detector drives it in H6). Red cuts the fan, and only the
// button (red -> green) starts it again. Serial commands, one per line:
//   fan on | fan off | fan speed <0-100> | status
// Every command is answered with one "ack" line carrying the Pico's timer (microseconds) when
// the command arrived, when the PWM was changed and when the gate pin was read back at its
// new level, so the time to cut the fan can be measured from the log (andon.fan_latency).
// Starting from stopped at a part speed runs a 1 s full-speed kick first (see fan.h). The fan
// is off at boot: a reset or power blip never restarts the machine by itself.
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
//   GP25 is the onboard LED on the Pico 2 (no wiring).
#include <stdio.h>

#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include "pico/stdlib.h"

#include "andon_light.h"
#include "andon_version.h"
#include "button.h"
#include "command.h"
#include "fan.h"

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

static const uint LIGHT_PINS[] = {PIN_GREEN, PIN_YELLOW, PIN_RED};
static uint32_t fan_pwm_wrap;

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

static void print_drive(const fan_drive_t *d, andon_light_t light) {
    printf("t_set_us=%llu t_pin_us=%lld pin=%d duty=%u light=%s", (unsigned long long)d->t_set_us,
           (long long)d->t_pin_us, d->pin, d->duty_pct, andon_light_name(light));
}

// Work out the duty the fan should have now and drive it if it changed. Called before
// anything is printed, so a slow USB print can never delay cutting the fan.
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

static fan_result_t run_command(fan_t *fan, command_t cmd, andon_light_t light) {
    switch (cmd.kind) {
    case CMD_FAN_ON:
        return fan_turn_on(fan, light);
    case CMD_FAN_OFF:
        fan_turn_off(fan);
        return FAN_OK;
    case CMD_FAN_SPEED:
        return fan_set_speed(fan, cmd.arg);
    default:
        return FAN_OK;  // status, and invalid lines (reported by the caller)
    }
}

// Read every character waiting on USB serial; handle each complete line.
static void poll_serial(command_reader_t *reader, fan_t *fan, fan_output_t *out,
                        andon_light_t light, uint8_t *driven_pct) {
    int c;
    while ((c = getchar_timeout_us(0)) != PICO_ERROR_TIMEOUT) {
        if (!command_reader_feed(reader, (char)c)) {
            continue;
        }
        uint64_t t_rx_us = time_us_64();
        command_t cmd = command_parse(reader->line);
        if (cmd.kind == CMD_INVALID) {
            printf("error: unknown command \"%s\" (fan on | fan off | fan speed <0-100> | "
                   "status)\n",
                   reader->line);
            continue;
        }
        fan_result_t result = run_command(fan, cmd, light);
        // Always drive and read back, even when nothing changed, so every ack is a
        // measurement of the pin's current level.
        uint32_t now_ms = to_ms_since_boot(get_absolute_time());
        *driven_pct = fan_output_update(out, fan_duty_pct(fan, light), now_ms);
        fan_drive_t d = drive_fan(*driven_pct);
        if (cmd.kind == CMD_STATUS) {
            printf("build=%s\n", ANDON_BUILD);
        }
        printf("ack cmd=\"%s\" result=%s t_rx_us=%llu ", reader->line, fan_result_name(result),
               (unsigned long long)t_rx_us);
        print_drive(&d, light);
        printf(" speed=%u want_on=%d kick=%d\n", fan->speed_pct, fan->want_on, out->kicking);
        if (result == FAN_REFUSED_RED) {
            printf("  the light is red: press the button to reset; the fan restarts if it was "
                   "running\n");
        }
    }
}

int main(void) {
    stdio_init_all();
    init_pins();
    init_fan_pwm();

    andon_light_t light = ANDON_GREEN;
    fan_t fan;
    fan_init(&fan);
    fan_output_t fan_out;
    fan_output_init(&fan_out);
    button_t button;
    button_init(&button, button_is_down(), to_ms_since_boot(get_absolute_time()));
    command_reader_t reader;
    command_reader_init(&reader);
    show_light(light);
    uint8_t duty_pct = 0;  // the PWM starts at 0 % and the fan is off at boot
    printf("andon h1 build=%s: light=%s fan duty=%u pwm=%lu Hz wrap=%lu\n", ANDON_BUILD,
           andon_light_name(light), duty_pct, (unsigned long)FAN_PWM_HZ,
           (unsigned long)fan_pwm_wrap);
    uint32_t next_status_ms = STATUS_MS;

    while (true) {
        uint32_t now_ms = to_ms_since_boot(get_absolute_time());
        heartbeat(now_ms);
        // Safety first: the light and the fan are set before anything is printed, because a
        // USB print can block for a while when the PC holds the port open without reading.
        bool light_changed = false;
        if (button_pressed(&button, button_is_down(), now_ms)) {
            light = andon_light_next(light);
            show_light(light);
            light_changed = true;
        }
        // Red cuts the fan, reset to green restarts it, and a kick-start drops to the set
        // speed here once FAN_KICK_MS is up.
        fan_drive_t d;
        bool fan_changed = update_fan(&fan, &fan_out, light, now_ms, &duty_pct, &d);
        if (light_changed) {
            printf("t=%lu ms light=%s\n", (unsigned long)now_ms, andon_light_name(light));
        }
        if (fan_changed) {
            printf("fan update ");
            print_drive(&d, light);
            printf(" kick=%d\n", fan_out.kicking);
        }
        poll_serial(&reader, &fan, &fan_out, light, &duty_pct);
        // Signed difference, so the check survives the 32-bit ms counter wrapping (~49.7 days).
        if ((int32_t)(now_ms - next_status_ms) >= 0) {
            printf("t=%lu ms alive build=%s light=%s fan duty=%u\n", (unsigned long)now_ms,
                   ANDON_BUILD, andon_light_name(light), duty_pct);
            next_status_ms += STATUS_MS;
        }
        sleep_ms(1);
    }
}
