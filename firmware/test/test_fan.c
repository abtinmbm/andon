// Host test for the Phase H1 logic (fan state, kill switch on red, serial commands). Built and
// run on the PC by tests/test_firmware.py; prints each failure and exits non-zero if any failed.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "andon_light.h"
#include "command.h"
#include "fan.h"

static int failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                  \
        }                                                                \
    } while (0)

// The device's PWM: 150 MHz / (5999 + 1) = 25 kHz.
#define WRAP 5999u

static void test_fan_is_off_at_boot_until_turned_on(void) {
    // A reset or power blip must never restart the machine by itself.
    fan_t fan;
    fan_init(&fan);
    CHECK(!fan.want_on);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 0);
    CHECK(fan_turn_on(&fan, ANDON_GREEN) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 100);  // full speed unless set otherwise
    CHECK(fan_duty_pct(&fan, ANDON_YELLOW) == 100);
}

static void test_red_always_stops_the_fan(void) {
    fan_t fan;
    fan_init(&fan);
    CHECK(fan_turn_on(&fan, ANDON_GREEN) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_RED) == 0);
    CHECK(fan_duty_pct(&fan, (andon_light_t)7) == 0);  // corrupted light value fails safe
    CHECK(fan_set_speed(&fan, 50) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_RED) == 0);
}

static void test_fan_on_is_refused_on_red(void) {
    fan_t fan;
    fan_init(&fan);
    fan_turn_off(&fan);
    CHECK(fan_turn_on(&fan, ANDON_RED) == FAN_REFUSED_RED);
    CHECK(!fan.want_on);
    // The reset button takes the light back to green; the refused "on" must not linger.
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 0);
}

static void test_reset_restarts_a_fan_that_was_running(void) {
    // Running, then red cuts it; pressing reset (red -> green) starts it again.
    fan_t fan;
    fan_init(&fan);
    CHECK(fan_turn_on(&fan, ANDON_GREEN) == FAN_OK);
    andon_light_t light = ANDON_RED;
    CHECK(fan_duty_pct(&fan, light) == 0);
    light = andon_light_next(light);
    CHECK(light == ANDON_GREEN);
    CHECK(fan_duty_pct(&fan, light) == 100);
}

static void test_off_on_and_speed(void) {
    fan_t fan;
    fan_init(&fan);
    fan_turn_off(&fan);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 0);
    fan_turn_off(&fan);  // twice is harmless
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 0);
    CHECK(fan_set_speed(&fan, 30) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 0);  // speed alone doesn't turn it on
    CHECK(fan_turn_on(&fan, ANDON_GREEN) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 30);
    CHECK(fan_turn_on(&fan, ANDON_YELLOW) == FAN_OK);  // yellow is a warning, not a stop
    CHECK(fan_set_speed(&fan, 0) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 0);
    CHECK(fan_set_speed(&fan, 100) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 100);
}

static void test_bad_speed_keeps_the_old_one(void) {
    fan_t fan;
    fan_init(&fan);
    CHECK(fan_set_speed(&fan, 101) == FAN_BAD_SPEED);
    CHECK(fan_set_speed(&fan, -1) == FAN_BAD_SPEED);
    CHECK(fan_set_speed(&fan, 999) == FAN_BAD_SPEED);
    CHECK(fan_turn_on(&fan, ANDON_GREEN) == FAN_OK);
    CHECK(fan_duty_pct(&fan, ANDON_GREEN) == 100);
}

static void test_kick_start_from_stopped_at_part_speed(void) {
    fan_output_t out;
    fan_output_init(&out);
    CHECK(fan_output_update(&out, 50, 1000) == 100);  // starts at full speed
    CHECK(out.kicking);
    CHECK(fan_output_update(&out, 50, 1000 + FAN_KICK_MS - 1) == 100);
    CHECK(fan_output_update(&out, 50, 1000 + FAN_KICK_MS) == 50);  // then the set speed
    CHECK(!out.kicking);
    CHECK(fan_output_update(&out, 50, 5000) == 50);
}

static void test_no_kick_at_full_speed_or_while_running(void) {
    fan_output_t out;
    fan_output_init(&out);
    CHECK(fan_output_update(&out, 100, 0) == 100);
    CHECK(!out.kicking);
    CHECK(fan_output_update(&out, 30, 10) == 30);  // slowing a spinning fan: no kick
    CHECK(!out.kicking);
    CHECK(fan_output_update(&out, 70, 20) == 70);
}

static void test_off_is_immediate_even_mid_kick(void) {
    fan_output_t out;
    fan_output_init(&out);
    CHECK(fan_output_update(&out, 50, 0) == 100);
    CHECK(fan_output_update(&out, 0, 10) == 0);  // the kill switch never waits
    CHECK(!out.kicking);
    // Starting again kicks again, with a fresh full second.
    CHECK(fan_output_update(&out, 50, 20) == 100);
    CHECK(fan_output_update(&out, 50, 20 + FAN_KICK_MS - 1) == 100);
    CHECK(fan_output_update(&out, 50, 20 + FAN_KICK_MS) == 50);
}

static void test_speed_change_during_kick_lands_on_the_new_speed(void) {
    fan_output_t out;
    fan_output_init(&out);
    CHECK(fan_output_update(&out, 50, 0) == 100);
    CHECK(fan_output_update(&out, 30, 500) == 100);  // still kicking
    CHECK(fan_output_update(&out, 30, FAN_KICK_MS) == 30);
}

static void test_kick_across_millisecond_wraparound(void) {
    fan_output_t out;
    fan_output_init(&out);
    uint32_t start = 0xFFFFFFFFu - 100u;
    CHECK(fan_output_update(&out, 50, start) == 100);
    CHECK(fan_output_update(&out, 50, start + 500u) == 100);  // wrapped, still kicking
    CHECK(fan_output_update(&out, 50, start + FAN_KICK_MS) == 50);
}

static void test_red_light_through_the_output_stops_at_once(void) {
    fan_t fan;
    fan_output_t out;
    fan_init(&fan);
    fan_output_init(&out);
    CHECK(fan_set_speed(&fan, 50) == FAN_OK);
    CHECK(fan_turn_on(&fan, ANDON_GREEN) == FAN_OK);
    CHECK(fan_output_update(&out, fan_duty_pct(&fan, ANDON_GREEN), 0) == 100);
    CHECK(fan_output_update(&out, fan_duty_pct(&fan, ANDON_RED), 1) == 0);
}

static void test_pwm_level(void) {
    CHECK(fan_pwm_level(0, WRAP) == 0);
    CHECK(fan_pwm_level(100, WRAP) == WRAP + 1);  // high for the whole period
    CHECK(fan_pwm_level(50, WRAP) == 3000);
    CHECK(fan_pwm_level(30, WRAP) == 1800);
    CHECK(fan_pwm_level(1, WRAP) == 60);
    CHECK(fan_pwm_level(255, WRAP) == WRAP + 1);
    CHECK(fan_pwm_level(50, 0xFFFFFFFEu) == 0x7FFFFFFFu + 1u);  // no 32-bit overflow
}

static void test_result_names(void) {
    CHECK(strcmp(fan_result_name(FAN_OK), "ok") == 0);
    CHECK(strcmp(fan_result_name(FAN_REFUSED_RED), "refused_red") == 0);
    CHECK(strcmp(fan_result_name(FAN_BAD_SPEED), "bad_speed") == 0);
    CHECK(strcmp(fan_result_name((fan_result_t)9), "invalid") == 0);
}

// Feed a whole string; returns how many lines completed and keeps the last one in `last`.
static int feed(command_reader_t *r, const char *s, char *last) {
    int lines = 0;
    for (; *s != '\0'; s++) {
        if (command_reader_feed(r, *s)) {
            lines++;
            strcpy(last, r->line);
        }
    }
    return lines;
}

static void test_reader_line_endings(void) {
    command_reader_t r;
    char last[COMMAND_MAX_LEN + 1] = "";
    command_reader_init(&r);
    CHECK(feed(&r, "fan off\n", last) == 1 && strcmp(last, "fan off") == 0);
    CHECK(feed(&r, "fan on\r", last) == 1 && strcmp(last, "fan on") == 0);
    CHECK(feed(&r, "status\r\n", last) == 1 && strcmp(last, "status") == 0);  // one line
    CHECK(feed(&r, "\n\r\n\r", last) == 0);                                   // empty lines
    CHECK(feed(&r, "fan", last) == 0);  // incomplete: nothing yet
    CHECK(feed(&r, " off\n", last) == 1 && strcmp(last, "fan off") == 0);
}

static void test_reader_overflow_is_reported_not_truncated(void) {
    command_reader_t r;
    char last[COMMAND_MAX_LEN + 1] = "x";
    command_reader_init(&r);
    // "fan off" followed by padding: cut at 31 chars it would still read as a command.
    CHECK(feed(&r, "fan off                                 x\n", last) == 1);
    CHECK(strcmp(last, "") == 0);
    CHECK(command_parse(last).kind == CMD_INVALID);
    // The reader recovers on the next line.
    CHECK(feed(&r, "fan on\n", last) == 1 && strcmp(last, "fan on") == 0);
    // Exactly the maximum length still fits.
    char max_line[COMMAND_MAX_LEN + 2];
    memset(max_line, 'a', COMMAND_MAX_LEN);
    max_line[COMMAND_MAX_LEN] = '\n';
    max_line[COMMAND_MAX_LEN + 1] = '\0';
    CHECK(feed(&r, max_line, last) == 1 && strlen(last) == COMMAND_MAX_LEN);
}

static void test_parse_commands(void) {
    CHECK(command_parse("fan on").kind == CMD_FAN_ON);
    CHECK(command_parse("fan off").kind == CMD_FAN_OFF);
    CHECK(command_parse("  fan \t off  ").kind == CMD_FAN_OFF);
    CHECK(command_parse("status").kind == CMD_STATUS);
    command_t c = command_parse("fan speed 50");
    CHECK(c.kind == CMD_FAN_SPEED && c.arg == 50);
    c = command_parse("fan speed 0");
    CHECK(c.kind == CMD_FAN_SPEED && c.arg == 0);
    c = command_parse("fan speed 999");  // parsed; fan_set_speed rejects it
    CHECK(c.kind == CMD_FAN_SPEED && c.arg == 999);
}

static void test_parse_rejects(void) {
    const char *bad[] = {
        "",           "fan",          "fan of",         "fan offf",      "fanoff",
        "FAN OFF",    "fan off now",  "fan on 1",       "status 1",      "fan speed",
        "fan speed -1", "fan speed 50%", "fan speed +5", "fan speed 1000", "fan speed 5 5",
        "fan speed x", "speed 50",    "off",
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        command_t c = command_parse(bad[i]);
        if (c.kind != CMD_INVALID || c.arg != 0) {
            printf("FAIL: \"%s\" parsed as a command\n", bad[i]);
            failures++;
        }
    }
}

int main(void) {
    test_fan_is_off_at_boot_until_turned_on();
    test_red_always_stops_the_fan();
    test_fan_on_is_refused_on_red();
    test_reset_restarts_a_fan_that_was_running();
    test_off_on_and_speed();
    test_bad_speed_keeps_the_old_one();
    test_kick_start_from_stopped_at_part_speed();
    test_no_kick_at_full_speed_or_while_running();
    test_off_is_immediate_even_mid_kick();
    test_speed_change_during_kick_lands_on_the_new_speed();
    test_kick_across_millisecond_wraparound();
    test_red_light_through_the_output_stops_at_once();
    test_pwm_level();
    test_result_names();
    test_reader_line_endings();
    test_reader_overflow_is_reported_not_truncated();
    test_parse_commands();
    test_parse_rejects();
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
