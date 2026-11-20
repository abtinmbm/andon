#include "recorder.h"

#include "pico/stdlib.h"
#include "tusb.h"

#include "mic_stream.h"

static mic_framer_t framer;
static recorder_stats_t stats;

void recorder_init(void) { mic_framer_init(&framer); }

static void send_frame(void) {
    if (framer.bytes[9] & MIC_FLAG_CLIPPED) {
        stats.frames_clipped++;
    }
    // Only write when the whole frame fits, so the write never waits on the PC.
    if (tud_cdc_connected() && tud_cdc_write_available() >= MIC_FRAME_BYTES) {
        stdio_put_string((const char *)framer.bytes, (int)MIC_FRAME_BYTES, false, false);
        stats.frames_sent++;
    } else {
        stats.frames_dropped++;
    }
}

void recorder_feed(const uint32_t *words, uint32_t n, bool overrun) {
    if (overrun) {
        stats.overruns++;
        mic_framer_mark_overrun(&framer);
    }
    for (uint32_t i = 0; i < n; i++) {
        // Frames are built even when not streaming, so a recording starts at once.
        if (mic_framer_add(&framer, words[i], stats.gain_shift) && stats.streaming) {
            send_frame();
        }
    }
}

void recorder_start(unsigned gain_shift) {
    mic_framer_init(&framer);  // sequence numbers restart at 0 for each recording
    stats = (recorder_stats_t){0};
    stats.gain_shift = gain_shift;
    stats.streaming = true;
}

void recorder_stop(void) { stats.streaming = false; }

const recorder_stats_t *recorder_stats(void) { return &stats; }
