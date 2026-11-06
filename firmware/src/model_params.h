// The exported model's parameters, wrapped in the structs the C kernels take. Only
// model_params.c includes the generated andon_model.h, so its arrays exist once in flash.
#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "logmel.h"
#include "model.h"

extern const mel_bank_t andon_mel_bank;
extern const model_t andon_model;

#endif  // MODEL_PARAMS_H
