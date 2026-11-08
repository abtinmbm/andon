#include "mic.h"

#include "hardware/dma.h"
#include "hardware/pio.h"
#include "pico/stdlib.h"

#include "mic.pio.h"
#include "mic_stream.h"

// One word per sample. 2^12 words = 256 ms at 16 kHz; the DMA wraps its write address
// within this ring, which needs the buffer aligned to its own size.
#define RING_LOG2_BYTES 14u
#define RING_WORDS ((1u << RING_LOG2_BYTES) / 4u)
// A reader that comes back later than this may have been lapped, so it is told it overran.
// (The ring holds 256 ms; the margin covers the reading of the write pointer itself.)
#define OVERRUN_US 200000u

static uint32_t ring[RING_WORDS] __attribute__((aligned(1u << RING_LOG2_BYTES)));
static int dma_chan = -1;
static uint32_t read_idx;
static uint64_t last_read_us;

static uint32_t write_idx(void) {
    uintptr_t addr = (uintptr_t)dma_channel_hw_addr((uint)dma_chan)->write_addr;
    return (uint32_t)((addr - (uintptr_t)ring) / 4u) % RING_WORDS;
}

void mic_init(void) {
    PIO pio = pio0;
    uint sm = pio_claim_unused_sm(pio, true);
    uint offset = pio_add_program(pio, &i2s_mic_program);

    dma_chan = dma_claim_unused_channel(true);
    dma_channel_config c = dma_channel_get_default_config((uint)dma_chan);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_read_increment(&c, false);
    channel_config_set_write_increment(&c, true);
    channel_config_set_ring(&c, true, RING_LOG2_BYTES);  // wrap the write address
    channel_config_set_dreq(&c, pio_get_dreq(pio, sm, false));
    // Endless mode (RP2350): the channel never finishes, so nothing has to re-arm it.
    dma_channel_configure((uint)dma_chan, &c, ring, &pio->rxf[sm],
                          dma_encode_endless_transfer_count(), true);

    i2s_mic_program_init(pio, sm, offset, MIC_PIN_BCLK, MIC_PIN_DOUT, MIC_SAMPLE_RATE_HZ);
    read_idx = write_idx();
    last_read_us = time_us_64();
}

uint32_t mic_read(uint32_t *out, uint32_t max_words, bool *overrun) {
    uint64_t now = time_us_64();
    uint32_t w = write_idx();
    if (now - last_read_us > OVERRUN_US) {
        *overrun = true;
        read_idx = w;  // the ring may hold a mix of old and new data: skip all of it
    }
    last_read_us = now;
    uint32_t n = 0;
    while (read_idx != w && n < max_words) {
        out[n++] = ring[read_idx];
        read_idx = (read_idx + 1u) % RING_WORDS;
    }
    return n;
}

uint32_t mic_ring_bytes(void) { return (uint32_t)sizeof(ring); }
