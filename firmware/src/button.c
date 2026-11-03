#include "button.h"

void button_init(button_t *b, bool is_down, uint32_t now_ms) {
    b->stable_down = is_down;
    b->raw_down = is_down;
    b->raw_since_ms = now_ms;
}

bool button_pressed(button_t *b, bool is_down, uint32_t now_ms) {
    if (is_down != b->raw_down) {
        b->raw_down = is_down;
        b->raw_since_ms = now_ms;
        return false;
    }
    // Unsigned subtraction stays correct across the 32-bit millisecond wraparound.
    if (b->raw_down != b->stable_down && now_ms - b->raw_since_ms >= BUTTON_DEBOUNCE_MS) {
        b->stable_down = b->raw_down;
        return b->stable_down;
    }
    return false;
}
