// Serial commands, read one character at a time. Plain C with no Pico SDK calls, so it also
// builds on the PC.
//
// Commands are one per line, ended by '\n' or '\r' (terminals send either):
//   fan on | fan off | fan speed <0-100> | status | rec start <gain 0-8> | rec stop
#ifndef COMMAND_H
#define COMMAND_H

#include <stdbool.h>
#include <stddef.h>

#define COMMAND_MAX_LEN 31u

typedef enum {
    CMD_INVALID = 0,
    CMD_FAN_ON,
    CMD_FAN_OFF,
    CMD_FAN_SPEED,
    CMD_STATUS,
    CMD_REC_START,  // stream the mic over USB at a gain shift of arg (mic_stream.h)
    CMD_REC_STOP,
} command_kind_t;

typedef struct {
    command_kind_t kind;
    int arg;  // the speed for CMD_FAN_SPEED, the gain shift for CMD_REC_START, else 0
} command_t;

typedef struct {
    char line[COMMAND_MAX_LEN + 1];
    size_t len;
    bool overflow;  // the current line was longer than COMMAND_MAX_LEN
} command_reader_t;

void command_reader_init(command_reader_t *r);

// Feed one received character. Returns true when a line is complete; it is then in r->line
// (NUL-terminated) until the next call. Empty lines are skipped. A line that was too long
// comes back as the empty string, which parses as CMD_INVALID, so it is reported instead of
// being cut short into a different, valid command.
bool command_reader_feed(command_reader_t *r, char c);

// Parse one line. Words may be separated by any number of spaces or tabs.
command_t command_parse(const char *line);

#endif
