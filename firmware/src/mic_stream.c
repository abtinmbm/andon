#include "mic_stream.h"

#include <string.h>

int16_t mic_word_to_sample(uint32_t word, unsigned gain_shift, bool *clipped) {
    if (gain_shift > MIC_GAIN_SHIFT_MAX) {
        gain_shift = MIC_GAIN_SHIFT_MAX;
    }
    // Keep the top (16 + gain_shift) bits of the 24-bit sample: the word's top 24 bits are
    // the sample, so shift out 16 - gain_shift bits in all. The shift of a negative int is
    // arithmetic (sign-filling) on GCC, the compiler for both the board and the host tests,
    // so this is exactly "take the top bits", rounding toward minus infinity.
    int32_t v = (int32_t)word >> (16u - gain_shift);
    if (v > INT16_MAX) {
        *clipped = true;
        return INT16_MAX;
    }
    if (v < INT16_MIN) {
        *clipped = true;
        return INT16_MIN;
    }
    return (int16_t)v;
}

uint16_t mic_crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)(v >> 8);
}

static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        p[i] = (uint8_t)(v >> (8 * i));
    }
}

void mic_framer_init(mic_framer_t *f) {
    memset(f, 0, sizeof(*f));
}

void mic_framer_mark_overrun(mic_framer_t *f) {
    // A full frame was already returned and is simply finished here; a partial one is lost.
    // Either way its sequence number is used up.
    if (f->count > 0) {
        f->count = 0;
        f->clipped = false;
        f->seq++;
    }
    f->overrun = true;
}

bool mic_framer_add(mic_framer_t *f, uint32_t word, unsigned gain_shift) {
    if (f->count == MIC_FRAME_SAMPLES) {
        // The previous call returned a full frame; it has been sent or dropped by now.
        f->count = 0;
        f->clipped = false;
        f->overrun = false;
        f->seq++;
    }
    int16_t s = mic_word_to_sample(word, gain_shift, &f->clipped);
    put_u16(&f->bytes[MIC_HEADER_BYTES + 2u * f->count], (uint16_t)s);
    f->count++;
    if (f->count < MIC_FRAME_SAMPLES) {
        return false;
    }
    memcpy(f->bytes, "AMIC", 4);
    put_u32(&f->bytes[4], f->seq);
    f->bytes[8] = (uint8_t)(gain_shift > MIC_GAIN_SHIFT_MAX ? MIC_GAIN_SHIFT_MAX : gain_shift);
    f->bytes[9] = (uint8_t)((f->overrun ? MIC_FLAG_OVERRUN : 0u) |
                            (f->clipped ? MIC_FLAG_CLIPPED : 0u));
    put_u16(&f->bytes[10], (uint16_t)MIC_FRAME_SAMPLES);
    size_t body = MIC_FRAME_BYTES - 2u;
    put_u16(&f->bytes[body], mic_crc16(f->bytes, body));
    return true;
}
