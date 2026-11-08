// Turning the microphone's I2S words into the 16-bit samples the front end sees, and packing
// them into frames for the USB stream. Plain C with no Pico SDK calls, so it also builds on
// the PC (firmware/test/test_mic.c), and andon/record.py parses the same frame layout.
//
// Frame layout (little-endian, MIC_FRAME_BYTES in all):
//   0   4  magic "AMIC"
//   4   4  sequence number, +1 per frame, counting frames that could not be sent too, so the
//          PC sees a gap wherever one was dropped
//   8   1  gain shift used for this frame (see mic_word_to_sample)
//   9   1  flags (MIC_FLAG_*)
//   10  2  number of samples (MIC_FRAME_SAMPLES)
//   12  2n samples, int16
//   ... 2  CRC-16/CCITT-FALSE over every byte before it
#ifndef MIC_STREAM_H
#define MIC_STREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MIC_SAMPLE_RATE_HZ 16000u
#define MIC_FRAME_SAMPLES 256u  // 16 ms at 16 kHz
#define MIC_HEADER_BYTES 12u
#define MIC_FRAME_BYTES (MIC_HEADER_BYTES + 2u * MIC_FRAME_SAMPLES + 2u)
#define MIC_GAIN_SHIFT_MAX 8u

// The samples before this frame were lost on the board (the loop fell behind the DMA ring),
// so this frame does not continue the previous one.
#define MIC_FLAG_OVERRUN 0x01u
// At least one sample in this frame was clipped to the int16 range by the gain.
#define MIC_FLAG_CLIPPED 0x02u

// One 32-bit word from the I2S program holds the mic's 24-bit two's-complement sample in its
// top 24 bits; the low 8 bits are the slot after the sample and are ignored. gain_shift
// (0 to MIC_GAIN_SHIFT_MAX) is a digital gain of 2^gain_shift: 0 keeps the top 16 of the 24
// bits, which is the full scale of the mic, and each step adds 6 dB. Samples beyond the int16
// range are clipped, and *clipped is set (it is never cleared here).
int16_t mic_word_to_sample(uint32_t word, unsigned gain_shift, bool *clipped);

uint16_t mic_crc16(const uint8_t *data, size_t len);

typedef struct {
    uint8_t bytes[MIC_FRAME_BYTES];
    size_t count;  // samples added to the frame being built
    bool clipped;
    bool overrun;
    uint32_t seq;  // sequence number of the frame being built
} mic_framer_t;

void mic_framer_init(mic_framer_t *f);

// Samples were lost before the next one added. A partly built frame is dropped (its sequence
// number is skipped), so no frame ever spans a gap, and the next frame carries
// MIC_FLAG_OVERRUN.
void mic_framer_mark_overrun(mic_framer_t *f);

// Add one I2S word. Returns true when the frame is full: f->bytes then holds a complete
// frame until the next call, and the framer starts the next frame on that call.
bool mic_framer_add(mic_framer_t *f, uint32_t word, unsigned gain_shift);

#endif
