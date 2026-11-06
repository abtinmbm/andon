#include "logmel.h"

#include <math.h>

#define N LOGMEL_N_FFT
#define LOG2_N 10

// Python adds sys.float_info.epsilon to the mel energies; in float32 it is the same value.
#define LOG_EPS ((float)2.220446049250313e-16)

_Static_assert((1 << LOG2_N) == N, "LOG2_N must match LOGMEL_N_FFT");

static float window[N];
static float twiddle_re[N / 2];  // e^(-2 pi i k / N)
static float twiddle_im[N / 2];
static uint16_t bit_reversed[N];
static float buf_re[N];
static float buf_im[N];

void logmel_init(void) {
    // Tables in double, rounded once to float, the way NumPy builds its float32 window.
    const double two_pi = 2.0 * 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
        window[i] = (float)(0.5 - 0.5 * cos(two_pi * i / N));
        unsigned r = 0;
        for (int b = 0; b < LOG2_N; b++) {
            r |= (unsigned)((i >> b) & 1) << (LOG2_N - 1 - b);
        }
        bit_reversed[i] = (uint16_t)r;
    }
    for (int k = 0; k < N / 2; k++) {
        twiddle_re[k] = (float)cos(two_pi * k / N);
        twiddle_im[k] = (float)-sin(two_pi * k / N);
    }
}

// In-place iterative radix-2 FFT of buf_re/buf_im (already in bit-reversed order).
static void fft(void) {
    for (int size = 2; size <= N; size <<= 1) {
        const int half = size / 2;
        const int step = N / size;
        for (int start = 0; start < N; start += size) {
            for (int k = 0; k < half; k++) {
                const float wr = twiddle_re[k * step];
                const float wi = twiddle_im[k * step];
                const int a = start + k;
                const int b = a + half;
                const float tr = buf_re[b] * wr - buf_im[b] * wi;
                const float ti = buf_re[b] * wi + buf_im[b] * wr;
                buf_re[b] = buf_re[a] - tr;
                buf_im[b] = buf_im[a] - ti;
                buf_re[a] += tr;
                buf_im[a] += ti;
            }
        }
    }
}

void logmel_frame(const mel_bank_t *bank, const float *frame, float *out) {
    for (int i = 0; i < N; i++) {
        buf_re[bit_reversed[i]] = frame[i] * window[i];
        buf_im[bit_reversed[i]] = 0.0f;
    }
    fft();
    // Power spectrum in place (bins 0..N/2 only; the rest mirror them for real input).
    for (int k = 0; k < LOGMEL_N_BINS; k++) {
        buf_re[k] = buf_re[k] * buf_re[k] + buf_im[k] * buf_im[k];
    }
    const float *w = bank->weight;
    for (int m = 0; m < bank->n_mels; m++) {
        const float *power = &buf_re[bank->start[m]];
        float energy = 0.0f;
        for (int j = 0; j < bank->len[m]; j++) {
            energy += w[j] * power[j];
        }
        w += bank->len[m];
        out[m] = 10.0f * log10f(energy + LOG_EPS);
    }
}

uint32_t logmel_ram_bytes(void) {
    return (uint32_t)(sizeof window + sizeof twiddle_re + sizeof twiddle_im +
                      sizeof bit_reversed + sizeof buf_re + sizeof buf_im);
}
