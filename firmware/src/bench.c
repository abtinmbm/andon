// andon_bench.uf2: runs the end-to-end self-test (src/selftest.c) on the Pico 2 and prints
// the result over USB serial every few seconds, so it is there whenever the port is opened.
// Uses no GPIO pins: only the USB cable.
#include <stdio.h>

#include "hardware/clocks.h"
#include "pico/stdlib.h"
#include "selftest.h"

static uint64_t board_now_us(void) { return time_us_64(); }

int main(void) {
    stdio_init_all();
    for (int run = 1;; run++) {
        selftest_result_t result;
        selftest_run(board_now_us, &result);
        printf("\n== andon bench, run %d, clk_sys %lu Hz\n", run,
               (unsigned long)clock_get_hz(clk_sys));
        selftest_print(&result);
        sleep_ms(5000);
    }
}
