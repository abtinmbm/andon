// Log-mel front end: one 1024-sample frame of audio in, one log-mel frame in dB out.
// Plain C with no Pico SDK calls, so the same file is tested on the PC against Python.
//
// It reproduces andon/features/logmel.py frame by frame: periodic Hann window, 1024-point
// FFT, power spectrum, the exported mel filterbank, then 10 * log10(mel + eps). All float32,
// as the RP2350's FPU is single precision. Framing (and Python's reflect padding of the first
// and last frames) is the caller's job.
#ifndef LOGMEL_H
#define LOGMEL_H

#include <stdint.h>

#define LOGMEL_N_FFT 1024
#define LOGMEL_N_BINS (LOGMEL_N_FFT / 2 + 1)

// Mel filterbank stored sparse, as andon.export_c writes it: band m covers FFT bins
// start[m] to start[m] + len[m] - 1, and its weights come right after band m - 1's.
typedef struct {
    int n_mels;
    const int16_t *start;
    const int16_t *len;
    const float *weight;
} mel_bank_t;

// Builds the window, twiddle and bit-reversal tables. Call once before logmel_frame.
void logmel_init(void);

// frame: LOGMEL_N_FFT samples scaled as sample / 32768.0f. out: bank->n_mels values in dB.
// Uses static scratch buffers, so it is not reentrant (call it from one core only).
void logmel_frame(const mel_bank_t *bank, const float *frame, float *out);

// Bytes of static RAM the front end uses, printed at startup on the device.
uint32_t logmel_ram_bytes(void);

#endif  // LOGMEL_H
