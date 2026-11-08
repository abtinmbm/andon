// Host test for the H2 microphone stream: 24-bit I2S words to int16 samples, and the frame
// layout that andon/record.py parses. Built and run on the PC by tests/test_firmware.py.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "command.h"
#include "mic_stream.h"

static int failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                  \
        }                                                                \
    } while (0)

// A 24-bit sample as the I2S program delivers it: in the top 24 bits, junk in the low 8.
static uint32_t word_of(int32_t sample24, uint8_t junk) {
    return ((uint32_t)sample24 << 8) | junk;
}

static uint16_t u16_at(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static void test_gain_0_keeps_the_top_16_bits(void) {
    bool clipped = false;
    CHECK(mic_word_to_sample(word_of(0x7FFFFF, 0xFF), 0, &clipped) == 32767);
    CHECK(mic_word_to_sample(word_of(-0x800000, 0xFF), 0, &clipped) == -32768);
    CHECK(mic_word_to_sample(word_of(0x000100, 0x00), 0, &clipped) == 1);
    CHECK(mic_word_to_sample(word_of(0x0000FF, 0xAB), 0, &clipped) == 0);  // below one LSB
    // Rounds toward minus infinity, like taking the top bits of a two's-complement number.
    CHECK(mic_word_to_sample(word_of(-1, 0x00), 0, &clipped) == -1);
    CHECK(mic_word_to_sample(word_of(-0x000100, 0x00), 0, &clipped) == -1);
    CHECK(mic_word_to_sample(word_of(-0x000101, 0x00), 0, &clipped) == -2);
    CHECK(!clipped);  // full scale never clips at gain 0
}

static void test_low_byte_is_ignored(void) {
    bool clipped = false;
    for (unsigned junk = 0; junk < 256; junk++) {
        CHECK(mic_word_to_sample(word_of(12345 * 256, (uint8_t)junk), 0, &clipped) == 12345);
        CHECK(mic_word_to_sample(word_of(-1234 * 256, (uint8_t)junk), 3, &clipped) ==
              -1234 * 8);
    }
}

static void test_gain_steps_are_6_db_and_clip(void) {
    bool clipped = false;
    CHECK(mic_word_to_sample(word_of(1000 * 256, 0), 1, &clipped) == 2000);
    CHECK(mic_word_to_sample(word_of(1000 * 256, 0), 4, &clipped) == 16000);
    CHECK(mic_word_to_sample(word_of(1000, 0), 8, &clipped) == 1000);  // all 24 bits
    CHECK(!clipped);
    CHECK(mic_word_to_sample(word_of(0x4000 * 256, 0), 1, &clipped) == 32767);
    CHECK(clipped);
    clipped = false;
    CHECK(mic_word_to_sample(word_of(-0x4001 * 256, 0), 1, &clipped) == -32768);
    CHECK(clipped);
    clipped = false;
    CHECK(mic_word_to_sample(word_of(-0x4000 * 256, 0), 1, &clipped) == -32768);  // exact fit
    CHECK(!clipped);
    // A gain above the maximum is held at the maximum rather than shifting by a negative.
    CHECK(mic_word_to_sample(word_of(1000, 0), 20, &clipped) == 1000);
}

static void test_crc_matches_the_standard_check_value(void) {
    // CRC-16/CCITT-FALSE of "123456789" is 0x29B1 (the published check value); Python's
    // binascii.crc_hqx(data, 0xFFFF) computes the same CRC.
    CHECK(mic_crc16((const uint8_t *)"123456789", 9) == 0x29B1u);
}

static void test_frames_have_the_documented_layout(void) {
    static mic_framer_t f;
    mic_framer_init(&f);
    int full = 0;
    for (unsigned i = 0; i < MIC_FRAME_SAMPLES; i++) {
        full += mic_framer_add(&f, word_of((int32_t)i * 256, 0x5A), 0);
    }
    CHECK(full == 1);
    CHECK(MIC_FRAME_BYTES == 526u);
    CHECK(memcmp(f.bytes, "AMIC", 4) == 0);
    CHECK(f.bytes[4] == 0 && f.bytes[5] == 0 && f.bytes[6] == 0 && f.bytes[7] == 0);
    CHECK(f.bytes[8] == 0);
    CHECK(f.bytes[9] == 0);
    CHECK(u16_at(&f.bytes[10]) == MIC_FRAME_SAMPLES);
    CHECK((int16_t)u16_at(&f.bytes[MIC_HEADER_BYTES]) == 0);
    CHECK((int16_t)u16_at(&f.bytes[MIC_HEADER_BYTES + 2 * 255]) == 255);
    CHECK(u16_at(&f.bytes[MIC_FRAME_BYTES - 2]) == mic_crc16(f.bytes, MIC_FRAME_BYTES - 2));
}

static void test_sequence_flags_and_reset_between_frames(void) {
    static mic_framer_t f;
    mic_framer_init(&f);
    mic_framer_mark_overrun(&f);
    for (unsigned i = 0; i < MIC_FRAME_SAMPLES; i++) {
        int32_t s = i == 7 ? 0x7FFFFF : 0;  // one sample clips at gain 2
        mic_framer_add(&f, word_of(s, 0), 2);
    }
    CHECK(f.bytes[8] == 2);
    CHECK(f.bytes[9] == (MIC_FLAG_OVERRUN | MIC_FLAG_CLIPPED));
    // The next frame starts clean and counts on.
    for (unsigned i = 0; i < MIC_FRAME_SAMPLES; i++) {
        mic_framer_add(&f, word_of(0, 0), 2);
    }
    CHECK(f.bytes[4] == 1);
    CHECK(f.bytes[9] == 0);
}

static void test_an_overrun_drops_the_partial_frame(void) {
    static mic_framer_t f;
    mic_framer_init(&f);
    for (unsigned i = 0; i < 100; i++) {
        mic_framer_add(&f, word_of(0x7FFFFF, 0), 2);  // clipped, then lost in the gap
    }
    mic_framer_mark_overrun(&f);
    int full = 0;
    for (unsigned i = 0; i < MIC_FRAME_SAMPLES; i++) {
        full += mic_framer_add(&f, word_of((int32_t)i * 256, 0), 0);
    }
    CHECK(full == 1);
    CHECK(f.bytes[4] == 1);                  // frame 0 was dropped, so the PC sees a gap
    CHECK(f.bytes[9] == MIC_FLAG_OVERRUN);   // and not clipped: those samples were dropped
    CHECK((int16_t)u16_at(&f.bytes[MIC_HEADER_BYTES]) == 0);  // starts with the new data
    // An overrun right after a full frame drops nothing: the next frame just carries the flag.
    mic_framer_mark_overrun(&f);
    for (unsigned i = 0; i < MIC_FRAME_SAMPLES; i++) {
        mic_framer_add(&f, word_of(0, 0), 0);
    }
    CHECK(f.bytes[4] == 2);
    CHECK(f.bytes[9] == MIC_FLAG_OVERRUN);
}

static void test_rec_commands_parse(void) {
    command_t c = command_parse("rec start 3");
    CHECK(c.kind == CMD_REC_START && c.arg == 3);
    c = command_parse("rec  start\t0");
    CHECK(c.kind == CMD_REC_START && c.arg == 0);
    c = command_parse("rec start 8");
    CHECK(c.kind == CMD_REC_START && c.arg == 8);
    CHECK(command_parse("rec start 9").kind == CMD_INVALID);  // above MIC_GAIN_SHIFT_MAX
    CHECK(command_parse("rec start").kind == CMD_INVALID);    // the gain is always explicit
    CHECK(command_parse("rec start -1").kind == CMD_INVALID);
    c = command_parse("rec stop");
    CHECK(c.kind == CMD_REC_STOP && c.arg == 0);
    CHECK(command_parse("rec stop 1").kind == CMD_INVALID);
    CHECK(command_parse("rec").kind == CMD_INVALID);
}

int main(void) {
    test_gain_0_keeps_the_top_16_bits();
    test_low_byte_is_ignored();
    test_gain_steps_are_6_db_and_clip();
    test_crc_matches_the_standard_check_value();
    test_frames_have_the_documented_layout();
    test_sequence_flags_and_reset_between_frames();
    test_an_overrun_drops_the_partial_frame();
    test_rec_commands_parse();
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("test_mic: all checks passed\n");
    return 0;
}
