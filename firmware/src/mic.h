// The ICS-43434 I2S microphone, read by a PIO program into a DMA ring (device only).
//
// Wiring (Pico 2 physical pin in brackets), Adafruit breakout pin names:
//   GP18 [24] -> BCLK    GP19 [25] -> LRCLK (WS)    GP20 [26] <- DOUT
//   3V3(OUT) [36] -> 3V    GND [38] -> GND    SEL -> GND (left channel)
// The mic runs from boot, so its 20 ms wake-up is long over before anything is recorded.
#ifndef MIC_H
#define MIC_H

#include <stdbool.h>
#include <stdint.h>

#define MIC_PIN_BCLK 18u  // WS is the next pin, GP19
#define MIC_PIN_DOUT 20u

// Start the PIO program and the DMA ring at MIC_SAMPLE_RATE_HZ.
void mic_init(void);

// Copy out the I2S words that arrived since the last call, at most max_words of them.
// Returns how many were copied. *overrun is set when the caller fell so far behind that the
// ring was overwritten: the words before the ones returned are then lost, and the reading
// restarts from the newest data.
uint32_t mic_read(uint32_t *out, uint32_t max_words, bool *overrun);

// Bytes of static RAM the ring takes (printed at startup).
uint32_t mic_ring_bytes(void);

#endif
