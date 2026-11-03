// Debounced pushbutton. Plain C with no Pico SDK calls, so it also builds on the PC.
//
// A mechanical button bounces for a few milliseconds when pressed or released, which would
// read as several presses. The raw level has to hold steady for BUTTON_DEBOUNCE_MS before
// the button counts as pressed or released.
#ifndef BUTTON_H
#define BUTTON_H

#include <stdbool.h>
#include <stdint.h>

#define BUTTON_DEBOUNCE_MS 20u

typedef struct {
    bool stable_down;        // debounced state
    bool raw_down;           // last raw reading
    uint32_t raw_since_ms;   // when the raw reading last changed
} button_t;

// Start from the current level, so a button already held at boot doesn't count as a press.
void button_init(button_t *b, bool is_down, uint32_t now_ms);

// Feed one raw reading; returns true exactly once per debounced press (on the way down).
// now_ms may wrap around; only differences between readings are used.
bool button_pressed(button_t *b, bool is_down, uint32_t now_ms);

#endif
