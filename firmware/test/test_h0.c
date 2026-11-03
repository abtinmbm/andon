// Host test for the Phase H0 logic (andon light cycle, button debounce). Built and run on
// the PC by tests/test_firmware.py; prints each failure and exits non-zero if any failed.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "andon_light.h"
#include "button.h"

static int failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                  \
        }                                                                \
    } while (0)

// Feed a constant level for `ms` milliseconds, one reading per ms (as the main loop does).
// Returns how many presses were reported.
static int hold(button_t *b, int is_down, uint32_t *now_ms, uint32_t ms) {
    int presses = 0;
    for (uint32_t i = 0; i < ms; i++) {
        presses += button_pressed(b, is_down, (*now_ms)++);
    }
    return presses;
}

static void test_light_cycles_green_yellow_red(void) {
    CHECK(andon_light_next(ANDON_GREEN) == ANDON_YELLOW);
    CHECK(andon_light_next(ANDON_YELLOW) == ANDON_RED);
    CHECK(andon_light_next(ANDON_RED) == ANDON_GREEN);
    CHECK(andon_light_next((andon_light_t)7) == ANDON_GREEN);
}

static void test_light_names(void) {
    CHECK(strcmp(andon_light_name(ANDON_GREEN), "green") == 0);
    CHECK(strcmp(andon_light_name(ANDON_YELLOW), "yellow") == 0);
    CHECK(strcmp(andon_light_name(ANDON_RED), "red") == 0);
    CHECK(strcmp(andon_light_name((andon_light_t)7), "invalid") == 0);
}

static void test_clean_press_counts_once(void) {
    button_t b;
    uint32_t t = 0;
    button_init(&b, 0, t);
    CHECK(hold(&b, 0, &t, 100) == 0);
    CHECK(hold(&b, 1, &t, BUTTON_DEBOUNCE_MS) == 0);  // not yet steady long enough
    CHECK(hold(&b, 1, &t, 2) == 1);
    CHECK(hold(&b, 1, &t, 1000) == 0);  // holding it doesn't repeat
    CHECK(hold(&b, 0, &t, 100) == 0);   // release is not a press
    CHECK(hold(&b, 1, &t, 100) == 1);   // second press
}

static void test_bounce_is_ignored(void) {
    button_t b;
    uint32_t t = 0;
    button_init(&b, 0, t);
    int presses = 0;
    for (int i = 0; i < 10; i++) {  // contacts chattering for 40 ms
        presses += hold(&b, 1, &t, 2);
        presses += hold(&b, 0, &t, 2);
    }
    presses += hold(&b, 1, &t, 100);
    for (int i = 0; i < 10; i++) {  // and again on release
        presses += hold(&b, 0, &t, 2);
        presses += hold(&b, 1, &t, 2);
    }
    presses += hold(&b, 0, &t, 100);
    CHECK(presses == 1);
}

static void test_held_at_boot_is_not_a_press(void) {
    button_t b;
    uint32_t t = 0;
    button_init(&b, 1, t);
    CHECK(hold(&b, 1, &t, 1000) == 0);
    CHECK(hold(&b, 0, &t, 100) == 0);
    CHECK(hold(&b, 1, &t, 100) == 1);
}

static void test_press_across_millisecond_wraparound(void) {
    button_t b;
    uint32_t t = UINT32_MAX - 5;
    button_init(&b, 0, t);
    CHECK(hold(&b, 1, &t, 100) == 1);
}

int main(void) {
    test_light_cycles_green_yellow_red();
    test_light_names();
    test_clean_press_counts_once();
    test_bounce_is_ignored();
    test_held_at_boot_is_not_a_press();
    test_press_across_millisecond_wraparound();
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
