// Per-fan Gaussian mixture scorer (the real-fan model H3 chose): the mean log-mel spectrum
// of a window of frames, scored by its negative log-likelihood under the fan's mixture.
// Plain C with no Pico SDK calls, so the same file is tested on the PC against Python.
//
// It reproduces andon/export_gmm.py in float32, as the RP2350's FPU is single precision:
// y = (x - mean) P for each component, log p = log_const - |y|^2 / 2, and the score is
// -logsumexp over the components. P is the upper-triangular Cholesky factor of the precision
// matrix, packed column by column (column j is P[0..j][j]).
#ifndef GMM_H
#define GMM_H

#include <stdint.h>

#define GMM_MAX_DIM 128
#define GMM_MAX_COMPONENTS 8

typedef struct {
    int n_components;
    int n_dim;
    const float *mean;       // [component][dim]
    const float *chol;       // [component][dim * (dim + 1) / 2], packed columns
    const float *log_const;  // [component]
} gmm_t;

// Negative log-likelihood of x (n_dim values) under the mixture. Higher = less normal.
// NaN if the model is larger than GMM_MAX_DIM or GMM_MAX_COMPONENTS.
float gmm_score(const gmm_t *g, const float *x);

// Streaming window mean over the log-mel frames: a window of 2 * hop frames every hop
// frames, as h3 scores (frames 0..2h-1, h..3h-1, ...). It keeps two half-window sums
// instead of the frames, so it needs 2 * n_dim floats whatever the window length.
typedef struct {
    int hop;
    int n_dim;
    int in_half;  // frames added to the current half
    int halves;   // halves completed since the reset
    float older[GMM_MAX_DIM];
    float newer[GMM_MAX_DIM];
} gmm_window_t;

// Returns 0, or -1 if hop < 1 or n_dim is out of range.
int gmm_window_reset(gmm_window_t *w, int hop, int n_dim);

// Adds one frame. Returns 1 and writes the window's mean into mean_out when the frame
// completes a window, else 0.
int gmm_window_push(gmm_window_t *w, const float *frame, float *mean_out);

#endif  // GMM_H
