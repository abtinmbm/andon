// The exported per-fan GMMs, wrapped in the structs gmm_score takes, one per fan. Only
// gmm_params.c includes the generated andon_gmm.h, so its arrays exist once in flash.
#ifndef GMM_PARAMS_H
#define GMM_PARAMS_H

#include "detector.h"
#include "gmm.h"

extern const int andon_gmm_n_fans;
extern const int andon_gmm_window_hop;  // frames; a window is twice this
extern const gmm_t andon_gmm[];
extern const char *const *const andon_gmm_fan_ids;  // fan index -> machine ID
extern const char *const andon_gmm_model_source;    // which training run, for the logs

// The detector's settings from the export (andon_gmm.h), with the front end's mel bank.
void andon_detector_config(detector_config_t *out);

#endif  // GMM_PARAMS_H
