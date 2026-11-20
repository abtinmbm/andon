#include "say.h"

#include <stdarg.h>
#include <stdio.h>

#include "pico/stdlib.h"
#include "tusb.h"

static char line[SAY_MAX_LEN + 1];
static uint32_t dropped, truncated;

void say(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(line, sizeof line, fmt, args);
    va_end(args);
    if (len < 0) {
        dropped++;
        return;
    }
    if (len > SAY_MAX_LEN) {
        truncated++;
        len = SAY_MAX_LEN;
        line[len - 1] = '\n';  // keep it a line, so the PC still splits the log correctly
    }
    if (!tud_cdc_connected() || tud_cdc_write_available() < (uint32_t)len) {
        dropped++;
        return;
    }
    stdio_put_string(line, len, false, false);
}

uint32_t say_dropped(void) { return dropped; }
uint32_t say_truncated(void) { return truncated; }
