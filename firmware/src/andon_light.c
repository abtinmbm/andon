#include "andon_light.h"

andon_light_t andon_light_next(andon_light_t light) {
    switch (light) {
    case ANDON_GREEN:
        return ANDON_YELLOW;
    case ANDON_YELLOW:
        return ANDON_RED;
    default:
        // Red and any corrupted value go back to green, so the light can't get stuck.
        return ANDON_GREEN;
    }
}

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
