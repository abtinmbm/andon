# Andon

Machine-sound anomaly detection. Andon learns what a healthy machine sounds like from recordings of normal running only, then raises a green / yellow / red andon signal when the machine starts to sound wrong. No recordings of faults are needed to train it.

Stage 1 (this repository today) builds and evaluates the detector on the public DCASE 2020 Task 2 factory-sound benchmark and plays it back as a live andon light. Stage 2 will run the same pipeline on a Raspberry Pi Pico 2 that cuts power to a real fan when it hears a fault. See [PLAN.md](PLAN.md).

![Andon light replay: a fan runs normally for two minutes, then develops a fault and the light goes from green to red](docs/andon_demo.gif)

*Replay of one fan (machine `id_06`, development test clips, model seed 0, shown at 8x real time). About two minutes of normal running, then a fault. The yellow light partway through is a real warning that clears by itself without stopping the line. This machine was chosen to show how the light works; the typical rates are in the tables below.*

## Results

**How well it separates faulty from normal sound.** AUC is the chance that a randomly picked faulty clip scores as more abnormal than a randomly picked normal one: 50 % is guessing, 100 % is perfect. These are the evaluation machines, scored once. Their normal recordings are part of training, as the DCASE protocol intends, but no model choice was tuned on their test clips:

| Evaluation AUC (%) | Autoencoder (DCASE baseline) | Gaussian mixture per machine | **Machine-ID CNN (final)** |
| --- | ---: | ---: | ---: |
| ToyCar | 79.65 ± 1.17 | 92.63 ± 0.80 | 90.29 ± 0.27 |
| ToyConveyor | 88.31 ± 0.38 | 91.64 ± 0.19 | 80.38 ± 2.00 |
| fan | 86.10 ± 0.82 | 95.58 ± 0.22 | 94.47 ± 0.86 |
| pump | 84.51 ± 0.57 | 86.46 ± 0.95 | 94.73 ± 0.85 |
| slider | 81.73 ± 0.41 | 79.34 ± 0.24 | 98.25 ± 0.40 |
| valve | 57.93 ± 0.19 | 54.27 ± 0.26 | 94.64 ± 0.23 |
| **Mean of six types** | 79.70 ± 0.39 | 83.32 ± 0.27 | **92.13 ± 0.16** |

Mean ± sample std over seeds 0, 1, 2; configs `phase3_baseline_eval`, `phase2_gmm_id` and `phase3_idcnn_mixup`. Every figure is computed from [results/results_log.csv](results/results_log.csv); the full dev and eval tables, with pAUC, are in [results/phase3_comparison.md](results/phase3_comparison.md).

The CNN beats the autoencoder baseline on 5 of the 6 machine types. It loses on ToyConveyor, and a much simpler model, a Gaussian mixture on each clip's average spectrum, still beats it on ToyCar, ToyConveyor and fan.

**How the andon light does on a continuous stream** (config `phase5_andon`, mean of the six types, mean ± std over model seeds 0, 1, 2):

| | Development test | Evaluation (scored once) |
| --- | --- | --- |
| False alarms per hour of normal audio | 4.49 ± 0.56 | 5.53 ± 1.79 |
| Faults caught before they end | 52.96 ± 0.99 % | 65.57 ± 1.02 % |
| Mean time from fault start to red, faults caught | 12.02 ± 0.32 s | 10.57 ± 0.24 s |

Per-type tables: [results/phase5_simulator.md](results/phase5_simulator.md). The limits of the model and of these numbers are in the [model card](docs/model_card.md).

**Small enough for a microcontroller.** The Stage 2 budget is at most 256 KB of model and under 20 ms of compute for every 32 ms of audio. With every weight and activation stored as an 8-bit integer (int8), the CNN fits with room to spare and scores almost the same:

| Machine-ID CNN | Float32 (PC) | Int8 (microcontroller) |
| --- | ---: | ---: |
| Model size | 383.0 KB | 98.7 KB |
| Estimated compute per 32 ms of audio | | 3.27 ms |
| Development AUC, mean of six types (%) | 86.64 ± 0.57 | 86.64 ± 0.60 |
| Evaluation AUC, mean of six types (%), scored once | 92.13 ± 0.16 | 92.08 ± 0.17 |

Configs `phase3_idcnn_mixup` and `phase4_idcnn_int8`, seeds 0, 1, 2. The int8 version is the same trained model, converted without retraining; on eval it changes AUC by −0.04 ± 0.02 points, a small, consistent drop that is smaller than the spread between seeds. Compute is estimated from the count of multiply-adds, assuming 0.5 per clock cycle on the 150 MHz Cortex-M33; Stage 2 will measure it. `andon.export_c` writes the int8 model as C headers, with test vectors that the C version must match bit for bit up to the last layer's integer sums (the float logits and scores within a small tolerance); CI runs it on every push with a seeded synthetic model of the same shape. All three models, with size, compute and AUC before and after int8: [results/phase4_mcu.md](results/phase4_mcu.md).

## Reproduce the headline number

Requires [uv](https://docs.astral.sh/uv/) and about 5 GB of free disk (2.3 GB of evaluation audio, 1.2 GB of cached features, and the Python environment). After cloning:

```
uv sync
uv run python -m andon.reproduce
```

This downloads the evaluation test clips and labels from Zenodo (MD5-checked) and the 18 trained CNN checkpoints from this repository's [`checkpoints-v1` release](https://github.com/abtinmbm/andon/releases/tag/checkpoints-v1) (SHA-256-checked). It then scores the evaluation set and checks every machine type and seed against its row in the results log. It ends with:

```
Eval AUC, mean of 6 machine types, seeds 0, 1, 2: 92.13 +/- 0.16 % (logged: 92.13 +/- 0.16 %)
Largest difference from the log: 0.0000 (tolerance 0.0005). Reproduced.
```

It adds nothing to the results log: the evaluation set is scored once per model, and this is a re-check of that run. `--train` retrains the checkpoints that are missing instead of downloading them (ones already in `checkpoints/` are scored as they are). That needs the full development and additional training sets (about 15 GB of audio) and many hours on a CPU. CPU training is seeded, but it is not guaranteed to give bit-identical weights on another machine, so a retrained run can land outside the tolerance and report "NOT reproduced".

## How it works

1. **Features.** Each 10 s clip becomes a log-mel spectrogram: 128 frequency bands, a new frame every 32 ms.
2. **Model.** One small CNN per machine type (about 98k weights) learns to tell that type's individual machines apart (fan `id_00` from fan `id_02`, and so on), using normal recordings only. Training blends pairs of examples (mixup), which keeps the classifier from becoming overconfident.
3. **Score.** A 2 s window scores high when the CNN can't recognise which machine it is hearing: the score is `-log p(correct machine ID)`. A clip's score is the mean over its windows. A faulty machine sounds unlike its normal self, so its own ID becomes less likely.
4. **Limits, set like a control chart.** For each machine, half of its normal test clips set a warning limit (mean + 2σ of the window scores) and a control limit (mean + 3σ). The other half is the normal running the light is judged on.
5. **The light.** A window is scored about once a second. Yellow comes on after 2 windows in a row above the warning limit, and goes off after 3 in a row below it, so a score hovering at the limit doesn't flicker. Red needs 3 windows in a row above the control limit and stays on until someone resets it. In the simulator a fault is 3 anomalous clips of the same machine (about 30 s), played after 20 s of normal running.

The development test set was used for every choice. The evaluation set (other machines of the same types, whose normal recordings are also in training) was scored once per final model. Every run is a YAML config in [configs/](configs/) and one row per machine type and seed in the results log, and [DECISIONS.md](DECISIONS.md) records what was tried and why.

**On the development set**, the PyTorch reproduction of the DCASE 2020 autoencoder baseline scores 73.80 ± 0.14 mean AUC, against the published 73.51. The per-machine Gaussian mixture scores 83.25 ± 0.22, the plain machine-ID CNN 84.33 ± 0.67 and the final CNN with mixup 86.64 ± 0.57 ([results/phase2_comparison.md](results/phase2_comparison.md), [results/phase3_comparison.md](results/phase3_comparison.md)).

## What didn't work

- **ToyConveyor.** It is the final CNN's weakest type on both sets. On eval it scores 80.38 ± 2.00 against the autoencoder's 88.31 ± 0.38; on dev, 76.18 ± 0.65 against the Gaussian mixture's 88.18 ± 0.84.
- **Spectrogram masking** on top of mixup moved the mean dev AUC from 86.64 ± 0.57 to 86.79 ± 0.19. That is below the seed spread, and it pulled valve down while pushing ToyConveyor up, so it was dropped (`phase3_idcnn_mixup_mask`).
- **Isolation forest** was the weakest detector: 57.98 ± 0.33 mean dev AUC per machine type, 63.26 ± 0.42 per machine, below the autoencoder baseline.
- **Valve with average-spectrum models.** Every classic detector scored 50.06 to 59.43 on valve on dev: close to guessing. The CNN, which looks at 2 s windows, reaches 87.63 ± 1.14 there.
- **Control limits on log scores.** On dev they gave 0.30 ± 0.30 false alarms per hour but caught only 24.87 ± 0.72 % of faults, and almost none for ToyCar or valve (`phase5_andon_log`).
- **The light misses many faults.** Even with a clip AUC above 90 % on eval, the light catches 65.57 ± 1.02 % of eval faults, and only 21.75 ± 2.22 % for valve. A single 2 s window is weak evidence, and limits set for few false alarms let many faulty windows through. With 10 s faults instead of 30 s, the dev catch rate falls to 32.79 ± 0.41 % (`phase5_andon_1clip`).
- **The autoencoder baseline doesn't fit the microcontroller.** Even at int8 it needs 272.1 KB, over the 256 KB budget.
- **Very loud faults can clip the int8 input.** The int8 input range is set from normal audio. In a test with a synthetic fault far louder than any normal clip, the input clipped and the int8 model scored the fault far lower than the float model did. On the DCASE evaluation set at most 0.003 % of input values clipped, all in anomalous pump clips, but a real fan fault could be much louder, so Stage 2 checks this.
- **False alarms.** 5.53 ± 1.79 false alarms per hour on eval, and 14.38 ± 7.19 for pump, would be too many on a real line. The limits trade false stops against missed faults; Stage 2 will pick that trade-off for one real machine.

## Run everything

```
uv sync                                                 # Python 3.12 and dependencies
uv run pytest -q                                        # tests (synthetic data, under a minute)
uv run python -m andon.data.download all                # development set, all six types (8.1 GB)
uv run python -m andon.data.download all --dataset eval_data   # additional training set, evaluation set, labels

uv run python -u -m andon.train --config configs/phase1_baseline.yaml --seed 0       # autoencoder baseline; repeat for seeds 1, 2
uv run python -m andon.classic --config configs/phase2_gmm_id.yaml --seed 0 1 2      # classic detectors
uv run python -u -m andon.train_idcnn --config configs/phase3_idcnn_mixup.yaml --seed 0 --threads 6   # final CNN; repeat for seeds 1, 2
uv run python -m andon.evaluate --compare phase1_baseline phase2_gmm_id phase3_idcnn_mixup   # tables

uv run python -u -m andon.quantize --config configs/phase4_idcnn_int8.yaml --seed 0 1 2 --threads 6   # int8 CNN, dev test
uv run python -m andon.quantize --config configs/phase4_baseline_int8.yaml --seed 0 1 2    # int8 autoencoder
uv run python -m andon.quantize --config configs/phase4_gmm_id_int8.yaml --seed 0 1 2      # int8 Gaussian mixture
uv run python -m andon.quantize --summary --config phase4_idcnn_int8 phase4_baseline_int8 phase4_gmm_id_int8 --split dev_test eval --out results/phase4_mcu.md
uv run python -m andon.export_c --config configs/phase4_idcnn_int8.yaml --machine fan --out firmware/model   # C headers

uv run python -m andon.simulate --seed 0 1 2 --split dev_test       # andon light on streamed audio
uv sync --all-groups                                                # adds the dashboard packages
uv run --group dashboard streamlit run dashboard/app.py             # live dashboard, plays the audio being scored
uv run --group dashboard python -m andon.demo --gif docs/andon_demo.gif
```

Each training and scoring command appends its rows to `results/results_log.csv` and skips rows that are already there, so the evaluation set can't be scored twice for the same config. The committed log already holds every run, so on a fresh clone these commands skip everything; to rerun from scratch, move `results/results_log.csv` aside in your copy first. Evaluation is scored with `--split eval` on `andon.score`, `andon.classic` and `andon.simulate`, and on `andon.quantize` only for a config marked `final: true`. Training is CPU-only; the autoencoder took about 6.5 h for 18 runs on a laptop.

```
andon/          Python package: data download and loading, log-mel features, models,
                training, scoring, the andon light, the streaming simulator,
                int8 quantization and the C export
configs/        one YAML file per experiment
results/        results_log.csv (one row per run) and the generated tables
dashboard/      Streamlit dashboard
docs/           model card and the demo GIF
tests/          pytest, on tiny synthetic data
```

## Data and licence

DCASE 2020 Task 2 data, built from ToyADMOS and MIMII, licensed CC BY-NC-SA 4.0: the development dataset (https://zenodo.org/records/3678171), the additional training dataset (https://zenodo.org/records/3727685) and the evaluation dataset (https://zenodo.org/records/3841772), with the evaluation ground truth (https://zenodo.org/records/3951620).

Code: MIT. The released checkpoints are trained on that data and are shared under the same CC BY-NC-SA 4.0 licence.
