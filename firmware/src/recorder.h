// Streams the microphone to the PC over USB serial as binary frames (mic_stream.h), between
// "rec start <gain>" and "rec stop". Device only.
//
// The stream never blocks the main loop: a frame is written only when the USB buffer has
// room for all of it, and otherwise dropped and counted (its sequence number is still used,
// so the PC sees the gap). That keeps the button and the kill switch as fast while recording
// as without it. Text lines (acks, the alive line) can arrive between frames; the PC skips
// them using the frame's magic bytes and CRC.
#ifndef RECORDER_H
#define RECORDER_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    bool streaming;
    unsigned gain_shift;
    uint32_t frames_sent;
    uint32_t frames_dropped;  // no room in the USB buffer
    uint32_t overruns;        // the loop fell behind the DMA ring
    uint32_t frames_clipped;
} recorder_stats_t;

void recorder_init(void);

// Drain the mic's DMA ring; while streaming, send each full frame. Call every loop pass.
void recorder_poll(void);

void recorder_start(unsigned gain_shift);
void recorder_stop(void);
const recorder_stats_t *recorder_stats(void);

#endif
