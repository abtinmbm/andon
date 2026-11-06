// Host run of the end-to-end self-test (src/selftest.c), the same code andon_bench.uf2 runs
// on the board: test audio -> C log-mel -> int8 CNN -> score, against Python.
#include <stdio.h>
#include <time.h>

#include "selftest.h"

// Host timings are only a sanity check; the board's time_us_64() numbers are the real ones.
static uint64_t host_now_us(void) {
    return (uint64_t)((double)clock() * 1e6 / CLOCKS_PER_SEC);
}

int main(void) {
    selftest_result_t r;
    const int pass = selftest_run(host_now_us, &r);
    selftest_print(&r);
    // The audio patches come from audio, so they must exist; a silent zero would pass.
    if (!pass || r.frames == 0 || r.patches == 0) {
        printf("self-test failed\n");
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
