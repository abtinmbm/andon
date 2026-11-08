#include "command.h"

#include <string.h>

#include "mic_stream.h"

#define MAX_WORDS 3

void command_reader_init(command_reader_t *r) {
    r->len = 0;
    r->overflow = false;
    r->line[0] = '\0';
}

bool command_reader_feed(command_reader_t *r, char c) {
    if (c == '\n' || c == '\r') {
        bool overflow = r->overflow;
        size_t len = r->len;
        r->line[overflow ? 0 : len] = '\0';
        r->len = 0;
        r->overflow = false;
        // "\r\n" ends one line, not two: the second character finds an empty line.
        return overflow || len > 0;
    }
    if (r->len < COMMAND_MAX_LEN) {
        r->line[r->len++] = c;
    } else {
        r->overflow = true;
    }
    return false;
}

static bool is_space(char c) { return c == ' ' || c == '\t'; }

// Finds the words without copying them: each word is a (start, length) pair.
static int split_words(const char *line, const char *starts[], size_t lens[]) {
    int n = 0;
    const char *p = line;
    while (*p != '\0') {
        while (is_space(*p)) {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        if (n == MAX_WORDS) {
            return MAX_WORDS + 1;  // too many words
        }
        starts[n] = p;
        while (*p != '\0' && !is_space(*p)) {
            p++;
        }
        lens[n] = (size_t)(p - starts[n]);
        n++;
    }
    return n;
}

static bool word_is(const char *start, size_t len, const char *word) {
    return len == strlen(word) && strncmp(start, word, len) == 0;
}

// Digits only, 1 to 3 of them, so "50%", "-1", "+5" and "1000000000000" are all rejected.
static bool parse_small_uint(const char *start, size_t len, int *out) {
    if (len == 0 || len > 3) {
        return false;
    }
    int v = 0;
    for (size_t i = 0; i < len; i++) {
        if (start[i] < '0' || start[i] > '9') {
            return false;
        }
        v = v * 10 + (start[i] - '0');
    }
    *out = v;
    return true;
}

command_t command_parse(const char *line) {
    command_t cmd = {CMD_INVALID, 0};
    const char *w[MAX_WORDS];
    size_t n[MAX_WORDS];
    int count = split_words(line, w, n);

    if (count == 1 && word_is(w[0], n[0], "status")) {
        cmd.kind = CMD_STATUS;
    } else if (count == 2 && word_is(w[0], n[0], "fan") && word_is(w[1], n[1], "on")) {
        cmd.kind = CMD_FAN_ON;
    } else if (count == 2 && word_is(w[0], n[0], "fan") && word_is(w[1], n[1], "off")) {
        cmd.kind = CMD_FAN_OFF;
    } else if (count == 3 && word_is(w[0], n[0], "fan") && word_is(w[1], n[1], "speed") &&
               parse_small_uint(w[2], n[2], &cmd.arg)) {
        cmd.kind = CMD_FAN_SPEED;
    } else if (count == 3 && word_is(w[0], n[0], "rec") && word_is(w[1], n[1], "start") &&
               parse_small_uint(w[2], n[2], &cmd.arg) && cmd.arg <= (int)MIC_GAIN_SHIFT_MAX) {
        cmd.kind = CMD_REC_START;
    } else if (count == 2 && word_is(w[0], n[0], "rec") && word_is(w[1], n[1], "stop")) {
        cmd.kind = CMD_REC_STOP;
    }
    if (cmd.kind != CMD_FAN_SPEED && cmd.kind != CMD_REC_START) {
        cmd.arg = 0;
    }
    return cmd;
}
