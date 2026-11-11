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
// The 3x3 neighbourhoods of a strip of output positions (see conv3x3): 16 KB, which holds
// 14 positions of the widest layer (64 input channels) and 910 of the first.
#define MODEL_PATCH_BUF 8192
static int16_t patches[MODEL_PATCH_BUF];
_Static_assert(MODEL_PATCH_BUF >= MODEL_MAX_IN_CH * 9, "one neighbourhood must fit");

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

// The 3x3 neighbourhood of one output position across every input channel, as
// (q_in - in_zero), in the weights' order [in][ky][kx]. A tap that falls in the padding is
// stored as 0, which is what "padding is real zero" means once the zero point is taken off.
static void gather_patch(const int8_t *in, int in_ch, int h, int w, int cy, int cx,
                         int32_t in_zero, int16_t *dst) {
    for (int i = 0; i < in_ch; i++) {
        const int8_t *plane = &in[i * h * w];
        for (int ky = 0; ky < 3; ky++) {
            const int iy = cy + ky - 1;
            for (int kx = 0; kx < 3; kx++) {
                const int ix = cx + kx - 1;
                const int inside = iy >= 0 && iy < h && ix >= 0 && ix < w;
                *dst++ = inside ? (int16_t)(plane[iy * w + ix] - in_zero) : 0;
            }
        }
    }
}

// Dot product of one output channel's weights with one gathered neighbourhood; n is a
// multiple of 9. Written out nine taps at a time so the loop overhead is paid once per
// input channel, not once per multiply.
static int32_t dot9(const int8_t *w, const int16_t *p, int n) {
    // |sum| <= 127 * 255 * 9 * 64 < 2^25, so int32 cannot overflow here.
    int32_t sum = 0;
    for (int j = 0; j < n; j += 9, w += 9, p += 9) {
        sum += w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + w[3] * p[3] + w[4] * p[4] +
               w[5] * p[5] + w[6] * p[6] + w[7] * p[7] + w[8] * p[8];
    }
    return sum;
}

// acc = bias + sum(w * (q_in - in_zero)) with zero padding, then the fixed-point rescale
// q = clamp(((acc * mult + 2^(shift-1)) >> shift) + out_zero).
//
// The first version checked the padding on every tap and measured 886 ms per window on the
// Pico 2. Here the neighbourhoods of a strip of output positions are gathered once, and
// each output channel then runs a plain dot product over the strip. Keeping the channel
// loop outside the strip's positions means one channel's weights (at most 576 bytes) are
// read many times in a row, which matters because they sit in flash behind a 16 KB cache.
// The integer sums are the same in any order, so the result is still bit-exact.
static void conv3x3(const conv_layer_t *L, int32_t in_zero, const int8_t *in, int h, int w,
                    int8_t *out, int oh, int ow) {
    const int n = L->in_ch * 9;
    const int positions = oh * ow;
    const int strip = MODEL_PATCH_BUF / n;  // at least 1: in_ch <= MODEL_MAX_IN_CH
    for (int first = 0; first < positions; first += strip) {
        const int count = positions - first < strip ? positions - first : strip;
        for (int k = 0; k < count; k++) {
            const int y = (first + k) / ow, x = (first + k) % ow;
            gather_patch(in, L->in_ch, h, w, y * L->stride, x * L->stride, in_zero,
                         &patches[k * n]);
        }
        for (int o = 0; o < L->out_ch; o++) {
            const int8_t *wo = &L->weight[o * n];
            const int shift = L->shift[o];
            const int64_t rounding = (int64_t)1 << (shift - 1);
            const int64_t bias = L->bias[o], mult = L->mult[o];
            int8_t *dst = &out[o * positions + first];
            for (int k = 0; k < count; k++) {
                // The bias can be large, so add it and rescale in 64 bits. >> on a negative
                // int64 is an arithmetic shift (floor) with GCC, as in the Python reference.
                const int64_t acc = (int64_t)dot9(wo, &patches[k * n], n) + bias;
                const int64_t v = (acc * mult + rounding) >> shift;
                dst[k] = (int8_t)clamp_int8(v + L->out_zero);
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
        if (L->in_ch != ch || ch > MODEL_MAX_IN_CH) return -1;
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
    return (uint32_t)(sizeof act_a + sizeof act_b + sizeof pooled + sizeof input_buf +
                      sizeof patches);
}
