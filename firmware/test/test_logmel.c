// Host test for the C log-mel front end (Phase H4): every frame of the exported test audio,
// framed and reflect-padded as andon/features/logmel.py does, must match Python's log-mel.
#include <math.h>
#include <stdio.h>

#include "andon_test_vectors.h"
#include "logmel.h"
#include "model_params.h"

// Largest allowed |C - Python| in dB. Both sides are float32 but sum in different orders
// (NumPy's FFT and BLAS against a textbook radix-2 FFT), so the last bits differ; the
// measured worst case (1.5e-5 dB on the real fan vectors) is printed on every run.
#define TOL_DB 0.001f

static int failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                  \
        }                                                                \
    } while (0)

// np.pad(mode="reflect"): mirror about the first and last sample, without repeating them.
static float padded_sample(long j) {
    const long len = ANDON_TV_AUDIO_LEN;
    long i = j - LOGMEL_N_FFT / 2;
    if (i < 0) i = -i;
    if (i >= len) i = 2 * (len - 1) - i;
    return andon_tv_audio[i] / 32768.0f;
}

static void test_frames_match_python(void) {
    static float frame[LOGMEL_N_FFT];
    static float mels[ANDON_TV_N_MELS];
    float worst = 0.0f;
    int worst_frame = 0, worst_mel = 0;
    for (int f = 0; f < ANDON_TV_FRAMES; f++) {
        for (int i = 0; i < LOGMEL_N_FFT; i++) {
            frame[i] = padded_sample((long)f * ANDON_TV_HOP_LENGTH + i);
        }
        logmel_frame(&andon_mel_bank, frame, mels);
        for (int m = 0; m < ANDON_TV_N_MELS; m++) {
            const float err = fabsf(mels[m] - andon_tv_logmel[f * ANDON_TV_N_MELS + m]);
            if (err > worst) {
                worst = err;
                worst_frame = f;
                worst_mel = m;
            }
        }
    }
    printf("log-mel: %d frames x %d mels, worst |C - Python| = %.6f dB (frame %d, mel %d),"
           " tolerance %.3f dB\n",
           ANDON_TV_FRAMES, ANDON_TV_N_MELS, worst, worst_frame, worst_mel, TOL_DB);
    CHECK(worst <= TOL_DB);
}

static void test_silence_gives_log_eps(void) {
    static float frame[LOGMEL_N_FFT];  // zeros
    static float mels[ANDON_TV_N_MELS];
    logmel_frame(&andon_mel_bank, frame, mels);
    const float expected = 10.0f * log10f((float)2.220446049250313e-16);  // about -156.5 dB
    for (int m = 0; m < ANDON_TV_N_MELS; m++) {
        CHECK(mels[m] == expected);
    }
}

// A full-scale tone puts its energy in the mel band whose filter peaks at its frequency.
static void test_tone_lands_in_its_band(void) {
    static float frame[LOGMEL_N_FFT];
    static float mels[ANDON_TV_N_MELS];
    const int bin = 100;  // 100 * 16000 / 1024 = 1562.5 Hz
    for (int i = 0; i < LOGMEL_N_FFT; i++) {
        frame[i] = (float)sin(2.0 * 3.14159265358979323846 * bin * i / LOGMEL_N_FFT);
    }
    logmel_frame(&andon_mel_bank, frame, mels);
    int loudest = 0;
    for (int m = 1; m < ANDON_TV_N_MELS; m++) {
        if (mels[m] > mels[loudest]) loudest = m;
    }
    const int first = andon_mel_bank.start[loudest];
    CHECK(first <= bin && bin < first + andon_mel_bank.len[loudest]);
}

int main(void) {
    logmel_init();
    test_frames_match_python();
    test_silence_gives_log_eps();
    test_tone_lands_in_its_band();
    printf("front end static RAM: %u bytes\n", (unsigned)logmel_ram_bytes());
    if (failures) {
        printf("%d check(s) failed\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
