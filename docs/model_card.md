# Model card: machine-ID CNN (`phase3_idcnn_mixup`)

The final Stage 1 anomaly detector. Every figure below comes from [results/results_log.csv](../results/results_log.csv) and is the mean ± sample std over seeds 0, 1, 2; the full tables are in [results/phase3_comparison.md](../results/phase3_comparison.md) and [results/phase5_simulator.md](../results/phase5_simulator.md).

## Model

- **Task.** Unsupervised anomaly detection in machine sound: score how abnormal a recording sounds, having seen only normal recordings.
- **Input.** Log-mel spectrogram of 16 kHz mono audio: 128 mel bands, 1024-point FFT, a frame every 32 ms. A window is 64 frames (about 2 s), each band normalised with statistics from the training clips.
- **Architecture.** One CNN per machine type, about 98k weights: a stride-2 convolution stem, then 3 pooled stages (16, 32, 64, 128 channels), and a classifier over that type's machine IDs (6 or 7).
- **Anomaly score.** `-log p(true machine ID)` for a window. A clip's score is the mean over windows that overlap by half. The machine ID comes from the file name, as it would from the wiring of a real sensor.
- **Training.** Normal clips only, of the development and evaluation machine IDs (development train set plus the additional training set). 100 epochs of one random window per clip, batch 128, Adam at 0.001 with cosine decay, mixup with α = 0.2. Seeds 0, 1, 2, on a CPU. The config is [configs/phase3_idcnn_mixup.yaml](../configs/phase3_idcnn_mixup.yaml).
- **Released weights.** 18 float32 checkpoints (6 types × 3 seeds), about 412 KB each, in the `checkpoints-v1` release. `uv run python -m andon.reproduce` downloads and checks them.

## Data

DCASE 2020 Task 2 (ToyADMOS and MIMII, CC BY-NC-SA 4.0): ToyCar, ToyConveyor, fan, pump, slider and valve, recorded in a lab with factory noise mixed in. Clips are about 10 s long.

- **Development test** (same machine IDs as training): every model choice was made here.
- **Evaluation test** (machine IDs no choice was tuned on): scored once, after all choices were fixed.
- No test clip of either set is used for training.

## Metrics

| | Development test | Evaluation (scored once) |
| --- | ---: | ---: |
| AUC, mean of six types (%) | 86.64 ± 0.57 | 92.13 ± 0.16 |
| pAUC (false-positive rate up to 10 %, DCASE standardised: 50 % is guessing), mean of six types (%) | 79.24 ± 0.71 | 85.44 ± 0.24 |

On eval, per type, AUC runs from 80.38 ± 2.00 (ToyConveyor) to 98.25 ± 0.40 (slider). The DCASE autoencoder baseline scores 79.70 ± 0.39 on eval, and a Gaussian mixture per machine 83.32 ± 0.27.

**As an andon light on streamed audio** (`phase5_andon`; limits at mean + 2σ and 3σ of window scores on half of each machine's normal test clips): on eval, 5.53 ± 1.79 false alarms per hour of normal audio, 65.57 ± 1.02 % of 30 s faults caught, 10.57 ± 0.24 s mean time to red.

## Intended use

A research baseline and the reference model for the Stage 2 device, which monitors one known machine. It is not validated for stopping real equipment: Stage 2 adds a separate measurement on a real fan.

## Known limits

- **Closed set of machines.** It can only score machine IDs it was trained on. A new machine needs its normal recordings added and the model retrained.
- **ToyConveyor.** It is the weakest type on both sets; on eval the autoencoder beats it (88.31 ± 0.38). A per-machine Gaussian mixture also beats it on eval on ToyCar, ToyConveyor and fan.
- **Clip AUC is not catch rate.** Valve has an eval AUC of 94.64 ± 0.23 but the light catches only 21.75 ± 2.22 % of valve faults. Single 2 s windows rarely cross the control limit 3 times in a row there.
- **False alarm rates rest on little audio.** For fan, pump, slider and valve each rate comes from 0.42 to 0.56 h of normal audio per type, so one false alarm moves it by about 2 per hour. Pump's eval rate is 14.38 ± 7.19 per hour.
- **Eval above dev.** Eval AUC is higher than dev AUC (92.13 against 86.64). This was not investigated: the two sets have different machine IDs, so they are not directly comparable.
- **Recording conditions.** Lab recordings of toys and single industrial machines with noise mixed in; there are no results yet on a real line, a different microphone, or a machine that drifts as it wears.
- **Size.** Float32 on a PC. Int8 quantization and the microcontroller budget are Phase 4.

## Licence

Code MIT. The data, and the released weights trained on it, are CC BY-NC-SA 4.0.
