// Int8 machine-ID CNN (Phase H5 kernels). Plain C with no Pico SDK calls, tested on the PC
// against the vectors andon.export_c writes. Integer arithmetic matches andon/quant.py bit
// for bit up to the head's int32 sums; only the logits and the score are float.
//
// Tensor layout is channel-major, [channel][mel][frame], as the CNN sees a patch.
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#define MODEL_MAX_CONV 4
#define MODEL_MAX_CLASSES 16
// Largest int8 activation the static buffers hold: the first conv's output, 16 x 64 x 32.
#define MODEL_ACT_MAX (16 * 64 * 32)
// The int8 input patch, 128 mels x 64 frames.
#define MODEL_INPUT_MAX (128 * 64)

// One 3x3 conv (padding 1) with batch norm and ReLU folded in, int8 in, int8 out.
typedef struct {
    int in_ch;
    int out_ch;
    int stride;
    int pool_before;  // 2x2 max pool on the input first
    int32_t out_zero;
    const int8_t *weight;  // [out][in][3][3]
    const int32_t *bias;   // in the accumulator's scale
    const int32_t *mult;   // fixed-point multiplier per output channel
    const int8_t *shift;
} conv_layer_t;

typedef struct {
    int n_mels;
    int patch_frames;
    int n_classes;
    int n_conv;
    int pool_size;  // positions summed by the global pool
    const float *band_mean;
    const float *band_std;
    double input_scale;
    int32_t input_zero;
    conv_layer_t conv[MODEL_MAX_CONV];
    const int8_t *head_weight;  // [class][channel]
    const int32_t *head_bias;
    const float *head_scale;  // logit = head_scale * accumulator
} model_t;

// Log-mel patch [frame][mel] (float dB) to int8 input [mel][frame]: per-band normalisation
// in float32, then rint(x / scale) + zero in double, clamped to int8.
void model_quantize_input(const model_t *m, const float *patch, int8_t *out);

// Int8 input to the head's int32 sums, one per class. If layer_out is not NULL, each conv
// layer's int8 output is copied to layer_out[i] (for the tests). Returns 0, or -1 if the
// model does not fit the static buffers or the patch shape is wrong.
int model_head_acc(const model_t *m, const int8_t *input, int32_t *head_acc,
                   int8_t *const *layer_out);

// Head sums to logits, and the anomaly score -log softmax(logits)[true_id].
void model_logits(const model_t *m, const int32_t *head_acc, float *logits);
float model_score(int n_classes, const float *logits, int true_id);

// Whole patch to score: quantize, run, score. Returns NAN if model_head_acc fails.
float model_score_patch(const model_t *m, const float *patch, int true_id);

// Bytes of static RAM the kernels use, printed at startup on the device.
uint32_t model_ram_bytes(void);

#endif  // MODEL_H
