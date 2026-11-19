// The exported model's parameters, wrapped in the structs the C kernels take. Only
// model_params.c includes the generated andon_model.h, so its arrays exist once in flash.
#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "model.h"

extern const model_t andon_model;

#endif  // MODEL_PARAMS_H
