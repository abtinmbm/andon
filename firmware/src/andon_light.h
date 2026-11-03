// The andon light's three states. Plain C with no Pico SDK calls, so it also builds on the PC.
#ifndef ANDON_LIGHT_H
#define ANDON_LIGHT_H

typedef enum {
    ANDON_GREEN = 0,
    ANDON_YELLOW = 1,
    ANDON_RED = 2,
} andon_light_t;

// Phase H0 only: each button press steps green -> yellow -> red -> green.
andon_light_t andon_light_next(andon_light_t light);

const char *andon_light_name(andon_light_t light);

#endif
