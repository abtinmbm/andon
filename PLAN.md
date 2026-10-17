# Andon: Project Plan

Andon is a machine-sound anomaly detector. It learns what a healthy machine sounds like from normal recordings only, then flags the machine, and in Stage 2 stops it, when it starts to sound wrong.

The name comes from lean manufacturing. An *andon* is the signal used to stop a production line when something is wrong. *Jidoka* means a machine stops itself when it detects an abnormality, and a person restarts it after checking.

The project has two stages. Stage 1 is a complete project on its own.

1. **Stage 1: software.** Train and evaluate anomaly detectors on the public DCASE 2020 Task 2 factory-sound benchmark. Reproduce the published baseline first, beat it on held-out machines, then shrink the model so it fits on a microcontroller. Finish with a streaming simulator and a dashboard that shows green, yellow and red andon states.
2. **Stage 2: hardware.** Run the same pipeline on a Raspberry Pi Pico 2 with a microphone, listening to a real 5 V fan. When a fault is injected (for example, tape on a blade or a blocked intake), the andon light turns red and the device cuts power to the fan.

---

## How it works

Two loops share one pipeline:

1. **Learn what normal sounds like (offline, on the PC).** Take normal clips only (no fault examples are needed), compute log-mel features (128 mel bands, 64 ms frames), and train the model (an autoencoder or a machine-ID CNN). Then set the yellow and red limits from the model's scores on normal audio, the same way Statistical Process Control (SPC) sets control limits.
2. **Watch live.** Audio comes in (replayed DCASE clips in Stage 1, the Pico's microphone in Stage 2). The same log-mel features are computed for each new frame, every 32 ms (Python in Stage 1, C on the Pico in Stage 2). The model scores each frame, where a high score means the frame sounds unlike normal. Each score updates the andon state, and each state change triggers an action: the dashboard light in Stage 1, the LEDs and a MOSFET that cuts the fan in Stage 2.

Red latches. Only a person pressing reset returns the state to green.

---

## Scope

| | In scope | Out of scope (for now) |
| --- | --- | --- |
| Data | DCASE 2020 Task 2: all 6 machine types; own fan recordings in Stage 2 | Later DCASE years (2021 to 2025, with domain shift); private factory data |
| ML | Reproduce the autoencoder baseline; classic baselines; one stronger model; int8 quantization | Large pretrained audio models in the final system (allowed only as a comparison) |
| Software | Training and evaluation pipeline, streaming simulator, local dashboard, tests, CI | Cloud hosting, user accounts, mobile app |
| Hardware | Raspberry Pi Pico 2, one microphone, a 5 V fan, andon LEDs, MOSFET auto-stop, reset button | Mains-powered machines, PLCs, motor drivers, custom PCB |
| Firmware | C with the Pico SDK: mic capture, log-mel features, int8 inference, andon state machine | An RTOS, over-the-air updates, Wi-Fi |

**Stretch goals**, in priority order:

1. **Learn normal on the device:** press a button, the device listens to a new machine for 5 minutes, sets its own thresholds, then starts watching. No PC needed.
2. **Vibration as a second sensor:** add an ADXL345 accelerometer and fuse the sound and vibration scores.
3. **Wireless dashboard:** switch to a Pico 2 W and send readings to the dashboard over Wi-Fi.
4. **Domain shift:** test on the DCASE 2023 "first-shot" data, where some machines have very little training data.

---

## Stage 1: software

Do the phases in this order: 0, 1, 2, 3, 5, 6, then 4. Phase 4 only matters once the hardware work starts, so it waits until the write-up is done.

### Phase 0: Setup

- Repo, Python 3.12 with `uv`, PyTorch, pytest, GitHub Actions. Development is on Windows.
- A download script for the DCASE 2020 Task 2 development data. Start with fan and ToyConveyor only (3.3 GB of the 8.1 GB total). Data stays out of git.
- A log-mel feature extractor, with unit tests on a synthetic sine wave.
- **Done when:** CI is green, and one command (`uv run python -m andon.data.download fan ToyConveyor`) downloads the data and prints the number of clips per machine ID.
- Scaffolded in advance: `pyproject.toml`, CI, `andon/features/logmel.py`, `andon/data/download.py`, `andon/data/dataset.py` and their tests. What's left is running it all on the dev machine and on GitHub.

### Phase 1: Reproduce the baseline

- Re-implement the official autoencoder baseline in PyTorch. The original is TensorFlow 1 / Keras.
  - Inputs: 128 mel bands, 5 stacked frames, so 640 values.
  - Layers: dense layers of 128 units with batch norm and ReLU, around an 8-unit bottleneck.
  - Training: 100 epochs, batch size 512, Adam with a learning rate of 0.001.
  - Anomaly score: the reconstruction error.
- Write the AUC and pAUC (FPR range 0 to 0.1) code by hand, and check it against scikit-learn on toy data. First confirm which pAUC definition the official baseline code uses (raw partial area / 0.1, or scikit-learn's McClish-standardized `max_fpr`), match it, and record the answer in DECISIONS.md.
- Run all 6 machine types with 3 random seeds.
- **Done when:** the mean AUC for every machine type is within 2 points of the published development-set baseline.

| Machine type | Published AUC | Published pAUC |
| --- | --- | --- |
| ToyCar | 78.77% | 67.58% |
| ToyConveyor | 72.53% | 60.43% |
| fan | 65.83% | 52.45% |
| pump | 72.89% | 59.99% |
| slider | 84.76% | 66.53% |
| valve | 66.28% | 50.98% |

### Phase 2: Classic baselines

- Three cheap detectors on simple features:
  - Mahalanobis distance on mean log-mel features
  - a Gaussian mixture model
  - an isolation forest
- **Done when:** one results table compares all three with the autoencoder. If they win on some machines, that is a finding to report, not a failure. They also serve as fallback models for the microcontroller.

### Phase 3: Beat the baseline

- Implement the approach used by the top DCASE 2020 systems: train a small CNN to tell machine IDs apart, using normal audio only. The anomaly score is how unsure the CNN is about the correct machine ID. The winning ensembles reached an average AUC of 94.7 on the evaluation set, against 80.0 for the baseline.
- Add mixup and spectrogram augmentation one at a time. Keep each one only if it helps on dev test.
- **Done when:** the eval set has been run once, and the model beats the baseline on at least 5 of the 6 machine types. The aim is to close about half the gap to the winners; matching a team ensemble is not required.

### Phase 4: Make it microcontroller-sized (do this after Phase 6)

- Hardware budget:
  - model weights at most 256 KB in int8
  - under 20 ms of compute per new audio frame (one arrives every 32 ms) on a 150 MHz Cortex-M33
- Estimate compute from multiply-accumulate counts for now. It gets measured for real in Stage 2.
- Pick the best model that fits the budget, apply post-training int8 quantization, and report the AUC lost.
- Export the weights and test vectors (input features and expected scores) to C header files, so Stage 2 can check the C code against Python.
- **Done when:** a table shows each model's size, estimated compute, and AUC before and after quantization, and the export script runs in CI.

### Phase 5: Andon simulator and dashboard

- A streaming simulator that plays test clips back to back (a long stretch of normal audio, then an anomaly) and scores them window by window, the way a real device would.
- An andon state machine:
  - green while scores look normal
  - yellow above the warning limit
  - red above the control limit, held for several windows in a row
  - hysteresis so the state doesn't flicker
  - red latches until reset
- A small local dashboard in Streamlit with a live score chart and the andon light. Record a GIF of it.
- **Done when:** the simulator reports false alarms per hour of normal audio and the average detection delay in seconds, and the GIF is in the README.

### Phase 6: Write-up

- A README covering the problem, the results tables, the GIF, how to run the project, and what didn't work.
- A one-page model card: data, metrics, known limits.
- **Done when:** someone else can clone the repo and reproduce the headline number with two commands.

---

## Stage 2: hardware

The owner has no hardware experience, so Stage 2 starts slowly. All firmware is C with the Pico SDK, not MicroPython or Arduino.

### Phase H0: First light

- Install the Pico SDK (the official VS Code extension works on Windows), flash a blinking LED, and print over USB serial.
- Wire three LEDs (green, yellow and red) and a button on a breadboard.
- **Done when:** pressing the button cycles through green, yellow and red.

### Phase H1: The machine and the kill switch

- A 5 V fan, switched through a logic-level MOSFET module with a flyback diode. The Pico's pin only ever drives the MOSFET, never the fan directly.
- The fan turns on and off from serial commands, then from the andon state.
- **Done when:** a serial command turns the fan off within 100 ms, confirmed with a timestamp log.

### Phase H2: Microphone and dataset

- A PDM MEMS microphone, using Arm's open-source microphone-library-for-pico (C, Pico SDK). Capture at 16 kHz, the same rate as the DCASE data.
- Stream raw audio to the PC over USB. A Python recorder on the PC saves labelled WAV files.
- Record about 30 minutes of normal running at two fan speeds.
- Record anomalies created safely:
  - tape or a small washer on one blade (imbalance)
  - a card partly blocking the intake (obstruction)
  - a loosened mount (rattle)
- **Done when:** the dataset has a written labelling protocol and the recordings play back cleanly.

### Phase H3: Retrain for the real fan

- A model trained on the DCASE fans will not transfer to a desk fan, because the sound, the mic and the background noise are all different. Calibrating on the machine being watched is normal practice.
- Train the Phase 4 model on the fan's normal recordings only, and test it on the labelled anomalies.
- **Done when:** there is an AUC on the real-fan data, produced under the same split rules as Stage 1.

### Phase H4: Features in C

- Port the log-mel feature extractor to C: windowing, FFT, mel filterbank, log.
- Check it against Python with the Phase 4 test vectors. Agreement within a small tolerance is the pass bar.
- **Done when:** a test on the PC compiles the C code, runs the test vectors and passes in CI.

### Phase H5: Inference in C

- Run the int8 model on the Pico. There are two options:
  - Preferred: hand-written C kernels for the layers.
  - Fallback: Raspberry Pi's pico-tflmicro port. It only documents the original Pico (RP2040), so it may need work on the Pico 2.
- Measure on the device: milliseconds per frame, RAM used, flash used.
- **Done when:** the on-device scores match Python's on the test vectors, and the timing fits the Phase 4 budget.

### Phase H6: Jidoka

- The full loop runs on the device: listen, score, update the andon state, and cut the fan on red.
- Only pressing the reset button restarts the fan.
- Send scores to the Stage 1 dashboard over USB serial.
- **Done when:** the fan stops by itself after a blade weight is added, and a log records the time from fault to stop.

### Phase H7: Measure and show

- Live trials:
  - run normal for an hour and count the false stops
  - inject each fault type 10 times, and record the detection rate and delay
- A 60-second demo video: the fan running, a weight added, the light going red, the fan stopping.
- **Done when:** the README has a hardware results table and the video.

### Parts

| Part | Job |
| --- | --- |
| Raspberry Pi Pico 2 (or Pico 2 H, which has headers pre-soldered) | Microcontroller (RP2350, 150 MHz Cortex-M33, 520 KB SRAM) |
| Adafruit PDM MEMS microphone breakout | Microphone |
| MAX9814 analog microphone module | Backup microphone, read with the ADC |
| 5 V DC fan, 40 to 60 mm, 2-wire | The machine being watched |
| Logic-level MOSFET module, or a MOSFET and a 1N4001 flyback diode | Kill switch |
| Red, yellow and green LEDs, 220 Ω resistors, a pushbutton | Andon light and reset button |
| Breadboard, jumper wires, micro-USB data cable | Wiring and programming |
| ADXL345 accelerometer (stretch goal) | Vibration sensing |

---

## ML and evaluation plan

The core rule: train on normal audio only, tune on development data, and touch the held-out set once.

**Data.** The DCASE 2020 Task 2 data combines ToyADMOS (toy car, toy conveyor) and MIMII (industrial fan, pump, valve and slide rail).

- Clips are about 10 seconds long, mono, 16 kHz, with background noise mixed in.
- Each machine ID has about 1,000 normal training clips.
- Licence: CC BY-NC-SA 4.0. Credit it in the README.
- The eval ground truth (`eval_data_list.csv`) is on Zenodo. The official evaluator script is y-kawagu/dcase2020_task2_evaluator.

**Splits.**

| Split | What it holds | Used for |
| --- | --- | --- |
| Dev train | Normal clips only, dev machine IDs | Training |
| Dev test | Labelled normal and anomalous clips, same IDs | All model choices and tuning |
| Eval (additional train + eval test) | Different machine IDs | One final run per model, reported as is |
| Real fan (Stage 2) | Own recordings | Training on normal clips; testing on labelled faults |

**Metrics.**

- **AUC** and **pAUC** (FPR range 0 to 0.1) for each machine type. These are the official DCASE metrics.
- **Operating metrics:**
  - detection rate at 1% false alarms
  - false alarms per hour of normal running
  - seconds from fault to red
- **On-device metrics:**
  - milliseconds per frame
  - KB of RAM and flash
  - AUC lost to int8

**Thresholds, set like SPC.** Score a long stretch of normal audio and fit the score distribution. Yellow is a warning limit, for example 2 standard deviations above the mean of normal scores. Red is the control limit, for example 3 standard deviations, and it must hold for several windows in a row.

**Experiment rules:**

1. Reproduce the published baseline before changing anything.
2. Every run is a config file plus one row in `results/results_log.csv`, recording the git commit, config, seed and metrics.
3. Report the mean and spread over 3 seeds.
4. Change one thing at a time, and keep it only if it helps on dev test.
5. Run the eval set once per final model, and report what comes out.
6. Keep `DECISIONS.md` up to date: what was tried, what happened, and why it was kept or dropped.

---

## Headline results to produce

These are the numbers the project exists to measure. Fill them with real results only.

- How close the PyTorch baseline gets to the published AUC, for each of the 6 machine types
- Mean AUC on held-out machines: the baseline compared with the improved model
- Model size after int8, and the AUC lost to quantization
- From the streaming simulator: detection delay in seconds and false alarms per hour
- Stage 2:
  - how closely the C output matches Python
  - milliseconds per frame and KB of RAM on the Pico
  - time from fault to fan stop
  - detection rate per fault type
  - false stops per hour

---

## Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| The PyTorch baseline doesn't match the 2020 numbers | Accept within 2 AUC points over 3 seeds. If a gap remains, document the likely cause (for example, a different normalization) instead of chasing it for weeks. |
| Training is slow on a laptop CPU | The autoencoder is small. For the Phase 3 CNN, use free GPU time on Colab or Kaggle, or shrink the model. |
| 8.1 GB of data is too much disk | Start with fan and ToyConveyor, and add the other machine types one at a time. |
| The PDM mic library misbehaves on the Pico 2 | Switch to the analog MAX9814 microphone on the ADC, or use an original Pico. |
| The pico-tflmicro port doesn't build for the Pico 2 | Hand-write the int8 layers in C. |
| Python and C features don't match | Compare one step at a time against the test vectors: window, FFT, mel, log. |
| Background noise ruins the recordings | Record at quiet times with the mic close to the fan, and add noise augmentation during training. |
| A wiring mistake damages the board | 5 V only, never mains. The fan always runs through the MOSFET. Get the wiring checked before the first power-up. |

---

## Repo layout

```
andon/
  PLAN.md                 this document
  DECISIONS.md            what was tried, what happened, why
  README.md
  pyproject.toml          uv project, Python 3.12
  andon/                  Python package
    data/                 download.py, dataset.py
    features/             logmel.py
    models/               autoencoder.py, classic.py, id_classifier.py
    metrics.py            AUC, pAUC, operating metrics
    train.py  evaluate.py  quantize.py  export_c.py
    stream/               simulator.py, andon_state.py
  configs/                one YAML file per experiment
  results/                results_log.csv (one row per run)
  dashboard/              app.py (Streamlit)
  firmware/               Stage 2: Pico SDK project in C
    src/                  main.c, mic.c, logmel.c, model.c, andon.c
    tests/                PC-side tests against exported test vectors
  tools/                  recorder.py (serial audio capture)
  tests/                  pytest
  .github/workflows/ci.yml
```

---

## Sources

- DCASE 2020 Task 2 description and baseline results: https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds
- DCASE 2020 Task 2 challenge results: https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds-results
- Development dataset (Zenodo): https://zenodo.org/records/3678171
- Official baseline: https://github.com/y-kawagu/dcase2020_task2_baseline
- Official evaluator: https://github.com/y-kawagu/dcase2020_task2_evaluator
- Microphone library for Pico: https://github.com/ArmDeveloperEcosystem/microphone-library-for-pico
- TensorFlow Lite Micro for Pico: https://github.com/raspberrypi/pico-tflmicro
