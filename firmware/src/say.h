// Text lines to the PC that never wait on it. Device only.
//
// printf over USB blocks for up to PICO_STDIO_USB_STDOUT_TIMEOUT_US (500 ms) when the PC
// holds the port open without reading. With the detector and the watchdog in the loop that
// is too long: the mic ring holds 256 ms, and the watchdog would reboot the board. So a line
// is written only when the USB buffer has room for all of it (4096 B, CMakeLists.txt), and is
// otherwise dropped and counted, as the recorder drops mic frames. A line is never split, so
// text and the recorder's binary frames never interleave mid-line.
#ifndef SAY_H
#define SAY_H

#include <stdint.h>

#define SAY_MAX_LEN 240  // longer lines are cut short (and counted in say_truncated)

// printf-style; the caller ends the line with '\n'.
void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

uint32_t say_dropped(void);
uint32_t say_truncated(void);

#endif
