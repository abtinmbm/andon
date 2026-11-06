#include "model.h"

#include <math.h>
#include <string.h>

#define INT8_LO (-128)
#define INT8_HI 127

// Ping-pong activation buffers plus one for the pooled input of a layer; no malloc.
static int8_t act_a[MODEL_ACT_MAX];
static int8_t act_b[MODEL_ACT_MAX];
static int8_t pooled[MODEL_ACT_MAX / 4];
static int8_t input_buf[MODEL_INPUT_MAX];

static int32_t clamp_int8(int64_t v) {
    return (int32_t)(v < INT8_LO ? INT8_LO : (v > INT8_HI ? INT8_HI : v));
}

void model_quantize_input(const model_t *m, const float *patch, int8_t *out) {
    for (int f = 0; f < m->patch_frames; f++) {
        for (int b = 0; b < m->n_mels; b++) {
            const float x = (patch[f * m->n_mels + b] - m->band_mean[b]) / m->band_std[b];
            // rint rounds half to even (default rounding mode), as np.round does.
            double q = rint((double)x / m->input_scale) + m->input_zero;
            q = q < INT8_LO ? INT8_LO : (q > INT8_HI ? INT8_HI : q);  // clamp before the cast
            out[b * m->patch_frames + f] = (int8_t)q;
        }
    }
}

// 2x2 max pool, floor on odd sizes (nn.MaxPool2d(2)). Exact on int8.
static void maxpool2(const int8_t *in, int ch, int h, int w, int8_t *out) {
    const int oh = h / 2, ow = w / 2;
    for (int c = 0; c < ch; c++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                const int8_t *p = &in[(c * h + 2 * y) * w + 2 * x];
                int8_t v = p[0];
                if (p[1] > v) v = p[1];
                if (p[w] > v) v = p[w];
                if (p[w + 1] > v) v = p[w + 1];
                out[(c * oh + y) * ow + x] = v;
            }
        }
    }
}

// acc = bias + sum(w * (q_in - in_zero)) with zero padding, then the fixed-point rescale
// q = clamp(((acc * mult + 2^(shift-1)) >> shift) + out_zero).
static void conv3x3(const conv_layer_t *L, int32_t in_zero, const int8_t *in, int h, int w,
                    int8_t *out, int oh, int ow) {
    for (int o = 0; o < L->out_ch; o++) {
        const int8_t *wo = &L->weight[o * L->in_ch * 9];
        const int shift = L->shift[o];
        const int64_t rounding = (int64_t)1 << (shift - 1);
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                // |sum| <= 127 * 255 * 9 * 64 < 2^25, so int32 cannot overflow here.
                int32_t sum = 0;
                for (int i = 0; i < L->in_ch; i++) {
                    const int8_t *wi = &wo[i * 9];
                    const int8_t *plane = &in[i * h * w];
                    for (int ky = 0; ky < 3; ky++) {
                        const int iy = y * L->stride + ky - 1;
                        if (iy < 0 || iy >= h) continue;  // padding is real zero
                        for (int kx = 0; kx < 3; kx++) {
                            const int ix = x * L->stride + kx - 1;
                            if (ix < 0 || ix >= w) continue;
                            sum += wi[ky * 3 + kx] * (plane[iy * w + ix] - in_zero);
                        }
                    }
                }
                // The bias can be large, so add it and rescale in 64 bits. >> on a negative
                // int64 is an arithmetic shift (floor) with GCC, as in the Python reference.
                const int64_t acc = (int64_t)sum + L->bias[o];
                const int64_t v = (acc * L->mult[o] + rounding) >> shift;
                out[(o * oh + y) * ow + x] = (int8_t)clamp_int8(v + L->out_zero);
            }
        }
    }
}

int model_head_acc(const model_t *m, const int8_t *input, int32_t *head_acc,
                   int8_t *const *layer_out) {
    if (m->n_conv > MODEL_MAX_CONV || m->n_classes < 1 || m->n_classes > MODEL_MAX_CLASSES) {
        return -1;
    }
    int ch = 1, h = m->n_mels, w = m->patch_frames;
    int32_t zero = m->input_zero;
    const int8_t *in = input;
    int8_t *bufs[2] = {act_a, act_b};
    for (int i = 0; i < m->n_conv; i++) {
        const conv_layer_t *L = &m->conv[i];
        if (L->in_ch != ch) return -1;
        if (L->pool_before) {
            if (ch * (h / 2) * (w / 2) > (int)sizeof pooled) return -1;
            maxpool2(in, ch, h, w, pooled);
            h /= 2;
            w /= 2;
            in = pooled;
        }
        const int oh = (h - 1) / L->stride + 1;  // (h + 2 - 3) / stride + 1
        const int ow = (w - 1) / L->stride + 1;
        if (L->out_ch * oh * ow > MODEL_ACT_MAX) return -1;
        int8_t *out = bufs[i % 2];
        conv3x3(L, zero, in, h, w, out, oh, ow);
        if (layer_out != NULL) memcpy(layer_out[i], out, (size_t)L->out_ch * oh * ow);
        in = out;
        ch = L->out_ch;
        h = oh;
        w = ow;
        zero = L->out_zero;
    }
    if (h * w != m->pool_size) return -1;
    // Global pool: the sum of (q - zero) over positions; the 1/HW is folded into head_scale.
    // |pooled| <= 255 * pool_size and |w . pooled| <= 127 * 255 * 32 * 128: fits int32.
    for (int k = 0; k < m->n_classes; k++) {
        int64_t acc = m->head_bias[k];
        for (int c = 0; c < ch; c++) {
            int32_t s = 0;
            const int8_t *plane = &in[c * h * w];
            for (int p = 0; p < h * w; p++) s += plane[p] - zero;
            acc += (int64_t)m->head_weight[k * ch + c] * s;
        }
        head_acc[k] = (int32_t)acc;
    }
    return 0;
}

void model_logits(const model_t *m, const int32_t *head_acc, float *logits) {
    for (int k = 0; k < m->n_classes; k++) logits[k] = (float)head_acc[k] * m->head_scale[k];
}

float model_score(int n_classes, const float *logits, int true_id) {
    // -log softmax = logsumexp(logits) - logits[true_id], shifted by the max for stability.
    float top = logits[0];
    for (int k = 1; k < n_classes; k++) {
        if (logits[k] > top) top = logits[k];
    }
    float total = 0.0f;
    for (int k = 0; k < n_classes; k++) total += expf(logits[k] - top);
    return top + logf(total) - logits[true_id];
}

float model_score_patch(const model_t *m, const float *patch, int true_id) {
    int32_t acc[MODEL_MAX_CLASSES];
    float logits[MODEL_MAX_CLASSES] = {0};
    if (m->n_mels * m->patch_frames > (int)sizeof input_buf) return NAN;
    model_quantize_input(m, patch, input_buf);
    if (model_head_acc(m, input_buf, acc, NULL) != 0) return NAN;
    model_logits(m, acc, logits);
    return model_score(m->n_classes, logits, true_id);
}

uint32_t model_ram_bytes(void) {
    return (uint32_t)(sizeof act_a + sizeof act_b + sizeof pooled + sizeof input_buf);
}
