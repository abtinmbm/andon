// The fan (the machine being watched) and its kill switch. Plain C with no Pico SDK calls,
// so it also builds on the PC.
//
// The fan is switched by a MOSFET whose gate is driven with PWM; this file only decides the
// duty cycle. The andon light has the last word: on red the fan is off whatever was asked
// for, and a serial "fan on" is refused, so only the reset button (red -> green) can start
// it again; it then restarts if it was running before red. The fan is off at boot, so a
// reset or power blip never restarts the machine by itself.
#ifndef FAN_H
#define FAN_H

#include <stdbool.h>
#include <stdint.h>

#include "andon_light.h"

#define FAN_FULL_SPEED_PCT 100u

typedef struct {
    bool want_on;       // what the operator last asked for over serial
    uint8_t speed_pct;  // duty cycle when running, 0 to 100
} fan_t;

typedef enum {
    FAN_OK = 0,
    FAN_REFUSED_RED,  // "fan on" while the light is red
    FAN_BAD_SPEED,    // speed outside 0 to 100
} fan_result_t;

// Off at boot, set to full speed: it starts only on a "fan on" command.
void fan_init(fan_t *fan);

fan_result_t fan_turn_on(fan_t *fan, andon_light_t light);
void fan_turn_off(fan_t *fan);
// Changes the speed it runs at; does not turn it on or off.
fan_result_t fan_set_speed(fan_t *fan, int speed_pct);

// The duty cycle to drive right now, in percent: 0 on red or when turned off.
uint8_t fan_duty_pct(const fan_t *fan, andon_light_t light);

// Kick-start. At a part speed a 2-wire fan keeps turning once it is spinning, but from
// standstill it only jitters (seen on the Sunon at 50 %). So a start from stopped runs at
// full speed for FAN_KICK_MS, then drops to the set speed. Stopping is never delayed.
#define FAN_KICK_MS 1000u

typedef struct {
    uint8_t duty_pct;    // what the gate is driven at now
    bool kicking;        // in the full-speed start
    uint32_t kick_start_ms;
} fan_output_t;

void fan_output_init(fan_output_t *out);

// Call with the wanted duty (fan_duty_pct) on every command, light change and loop pass;
// returns the duty to drive. A wanted duty of 0 always returns 0 at once, even mid-kick.
// now_ms may wrap around; only differences are used.
uint8_t fan_output_update(fan_output_t *out, uint8_t want_pct, uint32_t now_ms);

// PWM compare level for a duty cycle, for a counter that counts 0..wrap. 100 % returns
// wrap + 1, which keeps the output high for the whole period (wrap itself would leave one
// low tick per period).
uint32_t fan_pwm_level(uint8_t duty_pct, uint32_t wrap);

// One word, so the serial log stays easy to parse: ok, refused_red, bad_speed.
const char *fan_result_name(fan_result_t result);

#endif
