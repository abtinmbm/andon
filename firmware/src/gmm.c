#include "gmm.h"

#include <math.h>
#include <string.h>

float gmm_score(const gmm_t *g, const float *x) {
    float diff[GMM_MAX_DIM];
    float log_p[GMM_MAX_COMPONENTS];
    const int d = g->n_dim;
    // The scratch buffers are sized for the largest model; a bigger one must not overrun them.
    if (d < 1 || d > GMM_MAX_DIM || g->n_components < 1 ||
        g->n_components > GMM_MAX_COMPONENTS) {
        return NAN;
    }
    const int tri = d * (d + 1) / 2;
    float best = -INFINITY;
    for (int k = 0; k < g->n_components; k++) {
        const float *mean = &g->mean[k * d];
        const float *p = &g->chol[k * tri];
        for (int i = 0; i < d; i++) diff[i] = x[i] - mean[i];
        // Column j of P has j + 1 entries (rows 0..j), stored one column after another.
        float sq = 0.0f;
        for (int j = 0; j < d; j++) {
            float y = 0.0f;
            for (int i = 0; i <= j; i++) y += diff[i] * p[i];
            p += j + 1;
            sq += y * y;
        }
        log_p[k] = g->log_const[k] - 0.5f * sq;
        if (log_p[k] > best) best = log_p[k];
    }
    // logsumexp around the largest term, so the exponentials cannot overflow.
    float sum = 0.0f;
    for (int k = 0; k < g->n_components; k++) sum += expf(log_p[k] - best);
    return -(best + logf(sum));
}

int gmm_window_reset(gmm_window_t *w, int hop, int n_dim) {
    if (hop < 1 || n_dim < 1 || n_dim > GMM_MAX_DIM) return -1;
    memset(w, 0, sizeof *w);
    w->hop = hop;
    w->n_dim = n_dim;
    return 0;
}

int gmm_window_push(gmm_window_t *w, const float *frame, float *mean_out) {
    for (int i = 0; i < w->n_dim; i++) w->newer[i] += frame[i];
    if (++w->in_half < w->hop) return 0;
    // A half is complete: with the half before it, it makes a window.
    w->in_half = 0;
    int done = 0;
    if (++w->halves >= 2) {
        const float scale = 1.0f / (2.0f * (float)w->hop);
        for (int i = 0; i < w->n_dim; i++) mean_out[i] = (w->older[i] + w->newer[i]) * scale;
        done = 1;
    }
    memcpy(w->older, w->newer, sizeof w->older);
    memset(w->newer, 0, sizeof w->newer);
    return done;
}
