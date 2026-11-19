// The exported mel filterbank, wrapped in the struct logmel_frame takes. Only
// frontend_params.c includes the generated andon_frontend.h, so its arrays exist once in
// flash. Both detectors (the int8 CNN and the per-fan GMM) use this one front end.
#ifndef FRONTEND_PARAMS_H
#define FRONTEND_PARAMS_H

#include "logmel.h"

extern const mel_bank_t andon_mel_bank;

#endif  // FRONTEND_PARAMS_H
