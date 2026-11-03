// Phase H0, first light: a heartbeat on the onboard LED, a log over USB serial, and a
// button that steps the andon LEDs through green, yellow and red.
//
// Wiring (Pico 2 physical pin in brackets):
//   GP10 [14] -> 220 ohm -> green LED  -> GND [13]
//   GP11 [15] -> 220 ohm -> yellow LED -> GND [13]
//   GP12 [16] -> 220 ohm -> red LED    -> GND [13]
//   GP14 [19] -> pushbutton -> GND [18]   (internal pull-up, so pressed reads low)
//   GP25 is the onboard LED on the Pico 2 (no wiring).
#include <stdio.h>

#include "pico/stdlib.h"

#include "andon_light.h"
#include "button.h"

#define PIN_GREEN 10
#define PIN_YELLOW 11
#define PIN_RED 12
#define PIN_BUTTON 14
#define HEARTBEAT_MS 500u

static const uint LIGHT_PINS[] = {PIN_GREEN, PIN_YELLOW, PIN_RED};

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

static bool button_is_down(void) { return !gpio_get(PIN_BUTTON); }

static void show_light(andon_light_t light) {
    for (size_t i = 0; i < count_of(LIGHT_PINS); i++) {
        gpio_put(LIGHT_PINS[i], (size_t)light == i);
    }
}

static void heartbeat(uint32_t now_ms) { gpio_put(PICO_DEFAULT_LED_PIN, (now_ms / HEARTBEAT_MS) % 2); }

int main(void) {
    stdio_init_all();
    init_pins();

    andon_light_t light = ANDON_GREEN;
    button_t button;
    button_init(&button, button_is_down(), to_ms_since_boot(get_absolute_time()));
    show_light(light);
    printf("andon h0: light=%s\n", andon_light_name(light));

    while (true) {
        uint32_t now_ms = to_ms_since_boot(get_absolute_time());
        heartbeat(now_ms);
        if (button_pressed(&button, button_is_down(), now_ms)) {
            light = andon_light_next(light);
            show_light(light);
            printf("t=%lu ms light=%s\n", (unsigned long)now_ms, andon_light_name(light));
        }
        sleep_ms(1);
    }
}
