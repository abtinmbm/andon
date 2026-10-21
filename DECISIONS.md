# Decisions log

One entry per experiment or design choice, newest first. Keep each entry short. This log is the source for the README's "what didn't work" section.

## Template

### YYYY-MM-DD: short title (Phase N)
- **Tried:** what was changed, and the config name
- **Result:** the numbers, taken from results/results_log.csv
- **Decision:** kept or dropped
- **Why:** one or two sentences

---

### 2026-10-20: The headline is reproduced from released checkpoints, not by retraining (Phase 6)
- **Tried:** `andon.reproduce`, the second of the two "done when" commands after `uv sync`. It downloads only the evaluation test clips and labels (Zenodo, MD5-checked) and the 18 `phase3_idcnn_mixup` checkpoints (SHA-256 pinned in the code). It re-scores the evaluation set through `andon.score`'s code path and compares every machine type and seed with its logged row.
- **Result:** on this machine, with the data and checkpoints already on disk, all 36 eval AUC and pAUC values matched the log exactly (largest difference 0.0000), and the mean came out at the logged 92.13 ± 0.16. On 2026-10-21 a fresh clone of `e5bff26` in an empty folder, with `uv sync` and `uv run python -m andon.reproduce` only, downloaded the eval test clips and the 18 checkpoints from release `checkpoints-v1` and gave the same result: all 36 values matched (largest difference 0.0000), 92.13 ± 0.16, "Reproduced". The clone's working tree, including `results/results_log.csv`, was unchanged afterwards.
- **Decision:** kept, with these rules:
  - The re-scored rows go to `data/reproduced_eval.csv` (gitignored), never to `results/results_log.csv`.
  - It refuses any config or seed that has no logged eval row, before downloading, training or scoring anything, and it refuses any model other than the machine-ID CNN.
  - Tolerance: 0.0005 per AUC or pAUC.
  - `--train` retrains only the missing checkpoints.
- **Why:**
  - Retraining needs about 15 GB of audio and many hours on a CPU, which isn't a two-command check. Scoring fixed checkpoints is deterministic.
  - The eval-once rule is about choosing a model. Re-scoring the logged model to check it adds no new number to choose from. The guard stops the command from being used to look at eval for a model that was never scored.
  - The tolerance allows for another CPU or PyTorch build rounding floats differently and swapping two near-tied clips.

### 2026-10-20: Checkpoints published as a GitHub release under CC BY-NC-SA 4.0 (Phase 6)
- **Tried:** the 18 float32 checkpoints (about 412 KB each) as assets of release `checkpoints-v1`. Each checkpoint holds the weights, the config, the machine type, the seed, the ID list and the training history, and nothing machine-specific.
- **Decision:** kept (PM decision at the Phase 6 plan).
- **Why:** `.pt` files stay out of git. A release is a fixed URL next to the code, and the pinned hashes catch a replaced file. The weights are trained on CC BY-NC-SA 4.0 data, so they are shared under the same licence.

### 2026-10-20: README and model card figures are tested against the generated tables (Phase 6)
- **Tried:** `tests/test_docs.py` fails if `README.md` or `docs/model_card.md` quotes a two-decimal figure that is in none of `results/*.md`. It also fails on a mean paired with a spread that no table pairs it with, and on a relative link to a missing file. The only figures allowed from outside the tables are the published DCASE baseline values in `andon.evaluate`.
- **Decision:** kept.
- **Why:** the tables are generated from `results/results_log.csv`, so a hand-typed or stale figure fails CI. The test can't tell whether a figure is attributed to the right config or split; that was checked by hand in the Phase 6 review.

### 2026-10-20: pillow declared; CI actions moved to Node 24 releases (Phase 6)
- **Tried:**
  - `pillow>=12.3.0` added to the `dashboard` group. It was already installed through matplotlib, but `andon/demo.py` imports it directly.
  - `actions/checkout@v4` → `@v7` and `astral-sh/setup-uv@v6` → `@v10.3.0`, both on Node 24. setup-uv has no floating major tag after v7, so it is pinned exactly.
- **Decision:** kept (PM decisions at the Phase 6 plan).
- **Why:** a direct import should be declared, not rely on another package's dependencies. CI had warned that Node 20 actions are deprecated.

### 2026-10-20: Detection rate at 1 % false alarms deferred to H7 (Phase 6)
- **Tried:** nothing. It is the one PLAN.md operating metric that has not been computed.
- **Decision:** deferred to Phase H7 (PM decision).
- **Why:** on eval it would mean re-scoring the final model to read off a new number. In H7 it is measured on the real fan, where the operating point is chosen anyway.

### 2026-10-20: Evaluation set, scored once: the andon light catches 65.57 % of faults at 5.53 false alarms per hour (Phase 5)
- **Tried:** `phase5_andon` with every setting frozen after the dev run, seeds 0, 1, 2, `--split eval`, run once. Rows are tagged `4c3956c`. Tables are in `results/phase5_simulator.md`.
- **Result (mean of six types, mean ± sample std over seeds):**
  - Eval: 5.53 ± 1.79 false alarms per hour of normal audio, 65.57 ± 1.02 % of faults caught, 10.57 ± 0.24 s mean delay to red.
  - Dev: 4.49 ± 0.56, 52.96 ± 0.99 % and 12.02 ± 0.32 s.
  - On eval, caught rates run from 21.75 % (valve) and 35.40 % (ToyConveyor) to 94.27 % (fan). False alarms per hour run from 0.00 (valve) to 14.38 ± 7.19 (pump).
  - On dev, slider has the most false alarms (17.37 ± 5.78 per hour), and most of them come from `id_02` (8 of 11 on seed 0).
- **Decision:** reported as is.
- **Why it looks this way:** the CNN's clip AUC is high, but one 2 s window is weak evidence, and a limit set for few false alarms lets many faulty windows through. Valve has a high clip AUC (93.84 to 94.23 % on eval, simulation clips) but the lowest caught rate, so its fault windows rarely cross the control limit three times in a row; a likely cause is that valve sound is intermittent (open and close events), but this was not investigated. The false-alarm rates rest on little normal audio for the MIMII types (0.42 to 0.56 h per type), so one false alarm moves the rate by about 2 per hour; their seed spread shows it.

### 2026-10-20: Fault = three anomaly clips in a row (Phase 5)
- **Tried:** `phase5_andon_1clip`: `phase5_andon` with each fault a single 10 s anomaly clip (`fault_clips: 1`), dev test, seeds 0, 1, 2. Never run on eval.
- **Result (mean of six types):** 32.79 ± 0.41 % of faults caught, against 52.96 ± 0.99 % with three-clip faults. The false alarms per hour are the same 4.49 ± 0.56, since the limits and normal stream don't change. The mean delay is 5.44 ± 0.07 s against 12.02 ± 0.32 s, because a 10 s fault can only be caught quickly; the delay counts caught faults only.
- **Decision:** three clips (about 30 s) kept.
- **Why:** a failing machine stays faulty, and with 10 s faults two thirds are over before the light can act. The three clips are consecutive anomaly files of the same machine ID in name order, which may be different fault types; the leftover clips that don't fill a fault are not played.

### 2026-10-20: Limits on raw scores, not log scores (Phase 5)
- **Tried:** the first Phase 5 plan fitted mean + 2σ and 3σ on log window scores, to undo their right skew. A seed-0 dev check before any row was logged caught no ToyCar or valve fault. The variant is logged as `phase5_andon_log` (dev test, seeds 0, 1, 2; never run on eval).
- **Result (mean of six types):** log-scale limits give 0.30 ± 0.30 false alarms per hour but catch only 24.87 ± 0.72 % of faults: 0.09 ± 0.16 % of ToyCar's and 0.00 ± 0.00 % of valve's. Raw-score limits (`phase5_andon`) give 4.49 ± 0.56 and 52.96 ± 0.99 %.
- **Decision:** raw scores (`log_scale: false`), a plain Shewhart chart as PLAN.md describes it.
- **Why:** windows the classifier is very sure of score near 0, and their logs form a long left tail that inflates σ, so the 3σ control limit lands above almost every fault window. Raw limits were chosen knowing they cost more false alarms. Probability limits (quantiles of normal scores) were looked at in the same unlogged seed-0 check; no result is claimed for them, since they have no config or rows.

### 2026-10-20: Andon simulator protocol (Phase 5)
- **Tried:** `andon.simulate` plays each machine ID's test clips back to back and scores a 64-frame window every 32 frames with the Phase 3 CNN (`phase3_idcnn_mixup` checkpoints, nothing retrained). Per ID, the normal clips in name order alternate between calibration (they fit the limits and are used for nothing else) and simulation. False alarms are counted on the simulation normals back to back, with the light reset after each red. Each fault plays after two simulation normals (about 20 s). Audio time is counted as frames x 32 ms. The light: yellow after 2 windows above the warning limit and off after 3 below it; red after 3 above the control limit, latched.
- **Decision:** kept. Settings were fixed before the logged runs, with no sweep. The two changes (raw-score limits, three-clip faults) were made on an unlogged seed-0 dev check before any row was logged. The `phase5_andon_log` and `phase5_andon_1clip` rows were logged afterwards (after the eval run) to record that comparison with 3 seeds; they changed no setting.
- **How false alarms are counted:** the light is reset right after each red, so one long normal excursion above the control limit counts once per 3 windows it stays there. Windows that straddle the join between two unrelated clips are scored like any other; a join covers at most 2 consecutive windows, so it can't make a red alone.
- **Why:** half the normal test clips is the only normal audio the model never trained on, so limits fitted on training clips would be too tight. Each row's AUC and pAUC are the Phase 3 clip scores on the simulation normals and every anomaly clip, so they tie back to Phase 3.
- **Rows tagged `4c3956c-dirty`:** the 36 rows of `phase5_andon_log` and `phase5_andon_1clip`. They ran before their configs and the demo code were committed. `andon/simulate.py` and `andon/light.py` then matched `4c3956c` except for the summary-table printing, which was changed during the run and doesn't affect the rows. The configs are in the commit that follows.

### 2026-10-20: Demo replay and dashboard share one drawing; the GIF is rendered, not screen-recorded (Phase 5)
- **Tried:** `andon/demo.py` builds one stream with exactly the simulator's limits and windows: fan `id_06`, dev test clips, model seed 0, the first 12 simulation normals, then that ID's first fault (`configs/phase5_demo.yaml`). `dashboard/app.py` (Streamlit) replays it live with a reset button; `python -m andon.demo --gif` draws the same frames with matplotlib into `docs/andon_demo.gif`. The score axis is logarithmic because normal windows sit near 0.01 and fault windows near 10.
- **Result:** no results rows; the demo is an illustration. The ID was chosen because the light catches its faults (fan `id_06` was caught in every fault on seed 0), so the caption says it was picked and points at the rate table.
- **Decision:** kept.
- **Why:** a rendered GIF regenerates from one command and matches the dashboard exactly; a screen recording can't be reproduced. The dashboard and GIF packages live in a `dashboard` dependency group, and CI installs all groups so the drawing code is tested.

### 2026-10-20: pyarrow capped below 26 (Phase 5)
- **Tried:** `streamlit` needs `pyarrow`; Windows Smart App Control blocked pyarrow 26.0.0's DLL ("An Application Control policy has blocked this file"). Versions 20.0.0 to 25.0.0 each loaded in a clean environment.
- **Decision:** `pyarrow<26` in the `dashboard` group (25.0.1 installed), as with the numpy cap.
- **Why:** Smart App Control stays on; the cap can be lifted once the newer DLLs gain reputation.

### 2026-10-20: Evaluation set, scored once: the mixup CNN beats the autoencoder on 5 of 6 types (Phase 3)
- **Tried:** the three final models, each scored once on the evaluation test clips with seeds 0, 1, 2. All choices were already made on dev test.
  - `phase3_idcnn_mixup`, from its checkpoints, with no retraining.
  - `phase3_baseline_eval`, the autoencoder trained on the additional training set.
  - `phase2_gmm_id`, refitted per evaluation ID on the additional training set.
  - Tables are in `results/phase3_comparison.md`.
- **Result (eval AUC %, mean ± sample std over 3 seeds):**
  - Mean of the six types: CNN 92.13 ± 0.16, autoencoder 79.70 ± 0.39, per-ID GMM 83.32 ± 0.27. Mean eval pAUC: 85.44 ± 0.24, 63.21 ± 0.18 and 68.47 ± 0.63.
  - Against the autoencoder, the CNN is higher on ToyCar (90.29 against 79.65), fan (94.47 against 86.10), pump (94.73 against 84.51), slider (98.25 against 81.73) and valve (94.64 against 57.93).
  - It is lower on ToyConveyor (80.38 ± 2.00 against 88.31 ± 0.38).
  - The per-ID GMM stays ahead on ToyCar (92.63), ToyConveyor (91.64) and fan (95.58).
- **Decision:** reported as is. The Phase 3 target (beat the baseline on at least 5 of 6 types) is met on the evaluation set, as it was on dev.
- **Why it looks this way:** ToyConveyor is the CNN's weak type on both sets, and its seed spread is the largest there (2.00 on eval). The autoencoder's ToyConveyor score rose from 73.20 on dev to 88.31 on eval, so that gap belongs partly to the evaluation IDs, not only to the model. The CNN scoring higher on eval than on dev (92.13 against 86.64) was not investigated; the two sets have different machine IDs, so they are not directly comparable.

### 2026-10-20: Six CNN eval rows are tagged `4db225f-dirty`; five are in commit `1617472` (Phase 3)
- **Tried:** the eval job started while the eval-table code was still uncommitted, and the commit was made while the job was running.
- **Result:**
  - The six `phase3_idcnn_mixup` eval rows for ToyCar and ToyConveyor say `4db225f-dirty`, and five of them were picked up by commit `1617472`. Every later eval row says `1617472`.
  - The uncommitted files were the comparison-table code in `andon/evaluate.py`, the GMM eval path in `andon/classic.py`, the tests and this log. CNN scoring (`andon/score.py`, `andon/train_idcnn.py`, `andon/models/idcnn.py`) and `evaluate.log_scores` were the same as in `4db225f`.
- **Decision:** rows left as written. Eval is not re-run, the log is not edited and the pushed commit is not rewritten.
- **Why:** the scores come from fixed checkpoints, and the code that computed them is identical in both commits. Re-running eval to get a cleaner tag would break the run-once rule.

### 2026-10-20: Per-ID GMM on the evaluation set is refitted, not retrained under a new config (Phase 3)
- **Tried:** for eval, `andon.classic --split eval` fits `phase2_gmm_id` on the additional training set (the normal clips of the evaluation IDs, one GMM per ID) and scores the evaluation test clips. The rows are logged under `phase2_gmm_id` with split `eval`. The autoencoder got its own config (`phase3_baseline_eval`) because its eval model is a different training run with its own checkpoints. A per-ID GMM is the same procedure on any ID's normal clips, so it needs neither a new config nor a checkpoint. Eval tables come from `andon.evaluate --compare ... --split dev_test eval`, which keeps dev and eval in separate tables. `--beats` gives the per-type head to head and refuses partial or duplicated runs.
- **Decision:** kept.
- **Why:** it is the official protocol for the evaluation set (fit on its IDs' normal training clips), and the GMM fits in under a second per type.

### 2026-10-20: Spectrogram masking dropped; mixup CNN is the final Phase 3 model (Phase 3)
- **Tried:** `phase3_idcnn_mixup_mask`: `phase3_idcnn_mixup` plus SpecAugment-style masking of each training patch (2 bands of up to 12 mels, 2 spans of up to 6 frames). The sizes were fixed up front with no sweep. Seeds 0, 1, 2, dev test. The rule set before the run: keep it only if the mean dev AUC beats 86.64 by more than the seed spread. It was set in the step plan before the run, but no committed file had it until this entry. From now on, keep rules go in the config's comment, which is committed before the run.
- **Result (AUC %, mean ± sample std over 3 seeds):** the mean of the six types went from 86.64 ± 0.57 to 86.79 ± 0.19 (+0.15). Per-seed means went from 86.48 / 86.17 / 87.28 to 86.79 / 86.59 / 86.98. By type: ToyCar 86.69 → 87.05, ToyConveyor 76.18 → 80.15, fan 84.72 → 84.61, pump 87.12 → 88.20, slider 97.52 → 96.64, valve 87.63 → 84.07. Mean pAUC went from 79.24 to 79.44.
- **Decision:** dropped. `phase3_idcnn_mixup` is the final model and the only CNN scored on the evaluation set.
- **Why:** +0.15 is about a quarter of mixup's seed spread (0.57), so it fails the rule. Masking moved the types in opposite directions (ToyConveyor +3.97, valve −3.56), which a mean hides; with no clear net gain, the simpler model stays.

### 2026-10-20: Mixup kept for the machine-ID CNN (Phase 3)
- **Tried:** `phase3_idcnn_mixup`, identical to `phase3_idcnn` except mixup: each training patch is blended with another from the batch, weight lam ~ Beta(0.2, 0.2), with the same blend of the two ID losses. Alpha fixed up front, no sweep. Seeds 0, 1, 2, dev test.
- **Result (AUC %, mean ± sample std over 3 seeds):** mean of the six types 84.33 ± 0.67 → 86.64 ± 0.57 (+2.31); per-seed means 83.64 / 84.38 / 84.97 → 86.48 / 86.17 / 87.28. Higher on all six types: ToyCar 86.48 → 86.69, ToyConveyor 70.64 → 76.18, fan 82.04 → 84.72, pump 85.91 → 87.12, slider 93.94 → 97.52, valve 86.97 → 87.63. Mean pAUC 77.24 → 79.24, but lower on ToyCar (83.94 → 81.98) and valve (81.54 → 78.38).
- **Decision:** kept; step 4 (masking) builds on it.
- **Why:** the gain is over three times either config's seed spread and the worst mixup seed beats the best plain one. ToyConveyor gained most: without mixup its classifier reached 100% train accuracy (loss 0.0002), i.e. it was certain about every patch, which leaves little room for a fault to lower p(true ID); mixup's soft targets keep it less certain. That explanation fits the numbers but was not tested separately.

### 2026-10-20: Machine-ID CNN beats the autoencoder on 5 of 6 types on dev test (Phase 3)
- **Tried:** `phase3_idcnn`: one CNN per machine type (stride-2 stem, 3 pooled stages, 16-32-64-128 channels, about 98k weights) classifies 64-frame log-mel patches by machine ID. Trained on normal clips only, 100 epochs of one random patch per clip, Adam 1e-3 with cosine decay, last epoch kept; inputs normalised per mel band with training statistics stored in the checkpoint. Score: mean over patches (hop 32, last patch flush with the clip end) of -log p(true ID). Seeds 0, 1, 2, dev test.
- **Result (AUC %):** mean 84.33 ± 0.67 against the autoencoder's 73.80 ± 0.14 and per-ID GMM's 83.25 ± 0.22. Valve 86.97 ± 1.52 (autoencoder 65.96, GMM 59.43), slider 93.94 ± 1.14. Below the autoencoder only on ToyConveyor (70.64 ± 2.40 against 73.20 ± 0.47); below the GMM on ToyCar, ToyConveyor, fan and pump. Mean pAUC 77.24 ± 1.08 (autoencoder 59.93, GMM 65.95).
- **Decision:** kept as the Phase 3 base model.
- **Why:** the size was chosen by timing on this laptop's CPU before any training: the same widths with two convolutions per stage trained 2 to 9 times slower (612 patches/s with the stride-2 stem, 139 without, against 1240), too slow for 18-run jobs. No architecture search was done.

### 2026-10-20: One ID classifier per type on dev and evaluation IDs; autoencoder retrained for eval (Phase 3)
- **Tried:** the CNN learns every machine ID of a type: dev train plus the additional training set (normal clips of the evaluation IDs; 6 or 7 IDs per type). The autoencoder's evaluation-set run (`phase3_baseline_eval`) is `phase1_baseline` trained on the additional training set only, as the official baseline does; it is only checkpointed, never scored on dev test.
- **Result:** one CNN checkpoint scores both the dev test and the evaluation set, so eval is a single scoring pass (`andon.score --split eval`) with no retraining. The dev numbers come from a 6-to-7-ID problem rather than 3-to-4.
- **Decision:** kept (decided at the Phase 3 plan).
- **Why:** an ID classifier can only score IDs it was trained on; this is how the top DCASE 2020 systems trained. No test clip of either set is used in training; `all_train_clips` refuses an ID present in both sets.

### 2026-10-20: First 9 `phase3_idcnn` rows are tagged `fc2291e-dirty` (Phase 3)
- **Tried:** the training job was started before the code was committed. Rows for ToyCar, ToyConveyor and fan (all seeds) were logged with commit `fc2291e-dirty`; the remaining rows after the commit carry `a9746aa`.
- **Result:** no code file changed after 2026-10-19 05:33Z (file modification times), the first row was logged at 05:48Z, and the working tree matched `a9746aa` when it was committed. So all 18 rows ran `a9746aa`'s code.
- **Decision:** rows left as written (the log is never edited by hand); no re-run.
- **Why:** re-training would cost about an hour and reproduce the same code.

### 2026-10-18: GMM and Mahalanobis on the mean log-mel spectrum beat the autoencoder on 5 of 6 types (Phase 2)
- **Tried:** Mahalanobis distance, a Gaussian mixture and an isolation forest, each on one 128-value vector per clip (the mean of its log-mel frames), fitted on normal train clips only. Configs `phase2_{mahalanobis,gmm,iforest}_{type,id}`, seeds 0, 1, 2, dev test. Table: `results/phase2_comparison.md`.
- **Result (AUC %, mean ± sample std over 3 seeds; mean of the six types in the last figure):** per-ID GMM ToyCar 93.80 ± 0.84, ToyConveyor 88.18 ± 0.84, fan 84.01 ± 0.19, pump 86.53 ± 0.99, slider 87.54 ± 0.75, valve 59.43 ± 0.05; mean 83.25 ± 0.22 against the autoencoder's 73.80 ± 0.14. Per-type GMM: mean 81.72 ± 0.08 (best on pump 86.65 and slider 91.18). Per-ID Mahalanobis: mean 80.34. The autoencoder still wins valve (65.96 ± 0.82); every classic detector is at 50.06 to 59.43 there. Isolation forest is the weakest (mean 57.98 per type, 63.26 per ID), below the autoencoder on every type except pump per ID (75.39). On pAUC the per-type GMM has the best mean (66.05 ± 0.07), level with per-ID GMM (65.95 ± 0.82); the autoencoder's is 59.93 ± 0.31.
- **Decision:** reported as a finding. Per-ID GMM and per-ID Mahalanobis are the strongest cheap models, so they are the fallback candidates for the microcontroller (Phase 4) and the bar Phase 3 has to clear, not just the autoencoder.
- **Why:** a 10 s average keeps the steady spectral shape, which is where most of these anomalies show. Valve faults are likely short clicks that the average washes out, which a frame-level model can still see. The isolation forest's random axis-aligned splits likely suit 128 strongly correlated bands poorly. Neither explanation was tested.

### 2026-10-18: Fit classic detectors both per machine type and per machine ID (Phase 2)
- **Tried:** every detector twice: one model on all IDs of a type pooled (as the autoencoder is trained), and one model per ID, scoring each test clip with its own ID's model (the ID is in every file name).
- **Result (mean AUC % over the six types, per type → per ID):** Mahalanobis 74.96 → 80.34, GMM 81.72 → 83.25, isolation forest 57.98 → 63.26. Per ID is not better everywhere: GMM on slider 91.18 → 87.54. Per-ID GMM seeds spread more (pump pAUC ± 3.37 against ± 0.09 pooled), since each mixture sees a third to a quarter of the data (3 or 4 IDs per type).
- **Decision:** both kept in the table. Per type is the like-for-like comparison with the autoencoder; per ID matches the hardware stage, which monitors one fan.
- **Why:** reporting per ID alone would mix two changes (the model and the grouping) in one comparison.

### 2026-10-18: Classic detector settings fixed up front, no sweep (Phase 2)
- **Tried:** Mahalanobis with a Ledoit-Wolf shrunk covariance (about 1,000 clips per ID for 128 bands make the plain sample covariance noisy in the directions the distance weights most); GMM with 4 full-covariance components and `reg_covar` 0.01 dB²; isolation forest with 100 trees and scikit-learn's default 256 samples per tree. Score = squared distance, negative log-likelihood, and negated `score_samples` respectively, so higher means more anomalous.
- **Result:** Mahalanobis is deterministic, so its three seeds give identical rows (std 0.00); they are logged anyway so every config has 3 seeds. Fits take under 1.5 s per machine type; all 108 runs (6 configs × 6 types × 3 seeds) took about 3 minutes, almost all of it loading the feature cache.
- **Decision:** kept as the Phase 2 reference settings. Tuning them (more GMM components, adding the per-band std) is left to Phase 3, one change at a time on dev test.
- **Why:** Phase 2 is a reference point. Tuning six configs before there was a first table would have been scope the plan didn't ask for.

### 2026-10-18: scikit-learn becomes a runtime dependency (Phase 2)
- **Tried:** moved `scikit-learn>=1.9.1` from the dev group to the runtime dependencies. The lock is unchanged at 1.9.1, which already loaded under Smart App Control in the Phase 1 metric tests.
- **Decision:** kept.
- **Why:** writing a GMM and an isolation forest by hand would have been slower and riskier. They need porting to C only if one is chosen as the microcontroller fallback.

### 2026-10-18: Uncommitted new code also marks a run "-dirty" (Phase 2)
- **Tried:** `git_commit()` only looked at tracked files. The first Phase 2 job ran before its code was committed, so its 108 rows read `1d40af1-dirty`, a commit without the classic detectors. With only the new, untracked files added, the rows would have shown a clean hash that doesn't contain them.
- **Result:** we committed the code as `0b53cf9`, restored the log with git (no hand edit) and re-ran all six configs. The 108 new rows read `0b53cf9`, and every AUC, pAUC and per-ID value is identical to the first job's, so the seeded runs reproduce exactly.
- **Decision:** kept: untracked files under `andon/` and `configs/` now count as dirty too. Untracked output files elsewhere (generated tables) still don't. Covered by a test.
- **Why:** the tag must identify the code that produced a row.

### 2026-10-18: Comparison table built only from the results log (Phase 2)
- **Tried:** `uv run python -m andon.evaluate --compare <config names> --out <file>` prints AUC and pAUC tables (mean +/- sample std over seeds, in %) and writes them as Markdown. The "mean" row averages the six types within each seed, then takes the mean and std over seeds. It shows `n/a` unless every type has the same seeds. Duplicate rows raise an error. Only `dev_test` rows are used.
- **Decision:** kept; `results/phase2_comparison.md` is regenerated by the command in its first line.
- **Why:** every figure in the docs has to come from `results_log.csv`, and a generated table can't drift from it.

### 2026-10-18: PyTorch baseline reproduces the published DCASE 2020 numbers (Phase 1)
- **Tried:** `configs/phase1_baseline.yaml`, 6 machine types × seeds 0, 1, 2, dev test, commit `e2db074`.
- **Result (AUC %, mean ± sample std over 3 seeds, published in brackets):** ToyCar 79.65 ± 1.12 (78.77), ToyConveyor 73.20 ± 0.47 (72.53), fan 66.24 ± 1.38 (65.83), pump 73.12 ± 0.55 (72.89), slider 84.67 ± 0.52 (84.76), valve 65.96 ± 0.82 (66.28). Largest AUC gap 0.88 points (ToyCar); largest pAUC gap 1.18 (ToyCar). Mean of the six: 73.80 vs 73.51 published.
- **Decision:** kept as the reference model for every later comparison. Phase 1 "done when" (every type within 2 AUC points) is met.
- **Why:** with the Keras details below matched, no gap was left to explain.

### 2026-10-17: pAUC is scikit-learn's McClish-standardized value (Phase 1)
- **Tried:** read the official baseline (`01_test.py`) and evaluator (`y-kawagu/dcase2020_task2_evaluator`, `evaluator.py`). Both call `roc_auc_score(y_true, y_pred, max_fpr=0.1)`, which standardizes the partial area so that random = 0.5 and perfect = 1.0. It is not the raw partial area divided by 0.1.
- **Result:** `andon/metrics.py` computes the same thing by hand; tests match scikit-learn to 1e-12 on random scores, heavy ties and several `max_fpr` values. The raw area is kept as `partial_auc()`.
- **Decision:** kept. Per-ID AUC and pAUC are averaged unweighted per machine type, as both official scripts do.
- **Why:** our numbers must be comparable with the published table and with other DCASE 2020 reports.

### 2026-10-17: Match the Keras defaults the baseline relied on (Phase 1)
- **Tried:** reproduced what PyTorch does differently from Keras 2.1: Glorot-uniform weights and zero biases; batch norm momentum 0.99 (0.01 in PyTorch's convention) and epsilon 1e-3; Adam epsilon 1e-7; and `validation_split=0.1`, which holds out the last 10% of the unshuffled training windows (mostly the highest machine ID) rather than a random 10%. Like the baseline, one model per machine type is trained on the train clips of all its IDs, with no input normalization, and the last epoch is used (the held-out 10% only reports a loss).
- **Result:** see the reproduction entry above; no ablation of each detail was run.
- **Decision:** kept, all set in the config before the runs.
- **Why:** a few lines of config remove every obvious excuse if the numbers had missed; finding the cause of a gap afterwards would have cost far more training time.

### 2026-10-17: Fused Adam and 3 seeds in parallel to fit training on a laptop CPU (Phase 1)
- **Tried:** the default PyTorch Adam loops over the 36 parameter tensors one at a time: 4.8 ms per step against 0.9 ms for `fused=True`, which applies the same update in one kernel. With 8 threads a batch-512 step took about 10 ms, so one epoch on fan took 20 s and 18 runs one after another would have taken about 10 h. Three runs at 6 threads each ran at about 15 ms per step each, so about twice the throughput.
- **Result:** the full job (18 runs) finished in about 6.5 h overnight (ToyCar 84 min, slider 53 min per run).
- **Decision:** kept `fused=True`; `--threads` flag on `andon.train`; per-process temp files and a lock on `results_log.csv` so parallel seeds can't collide. Logs need `python -u`, otherwise piped output is buffered for hours.
- **Why:** keeps the CPU-only setup (no GPU, no data upload) without changing the training algorithm.

### 2026-10-17: Cache log-mel frames, stack windows per batch (Phase 1)
- **Tried:** the baseline builds every 640-value vector up front, about 3 GB of float32 per machine type. We cache only the 128-band frames (`data/cache/`, keyed by clip list and front-end settings) and stack 5-frame windows when each batch is built.
- **Result:** a test checks our stacking against the baseline's own loop, element by element, so the vectors are identical.
- **Decision:** kept.
- **Why:** 5x less memory, so three seeds fit in RAM at once.

### 2026-10-18: Results-log commit tags read "-dirty" because of the log itself (Phase 1)
- **Tried:** `git_commit()` marked a row `-dirty` whenever any tracked file had changes. `results_log.csv` is tracked, so after the first row every later row looked dirty.
- **Result:** 17 of the 18 Phase 1 rows read `e2db074-dirty`. During the runs `git status` showed `results/results_log.csv` as the only changed tracked file, so all 18 rows came from the code at `e2db074`. The later code diff touches only the summary print and this check, and re-scoring the saved fan and valve seed-0 checkpoints into a scratch log reproduced their logged AUC, pAUC and per-ID values exactly.
- **Decision:** rows left as written (the log is never edited by hand); the dirty check now ignores the log, with a test.
- **Why:** the tag exists to catch uncommitted code changes, not new results.

### 2026-10-17: Reflect padding at the clip edges, as in the baseline (Phase 0)
- **Tried:** the first version zero-padded the centered STFT frames, matching librosa 1.0's default. The baseline pinned librosa 0.6.0, whose `stft` defaults to `pad_mode='reflect'`.
- **Result:** zero padding moved only the first and last frame of each clip, by several dB; every other frame was unchanged (checked by `test_pad_mode_changes_only_the_edge_frames`). With reflect padding the librosa cross-check passes on every frame.
- **Decision:** kept `pad_mode="reflect"` as the `LogMelConfig` default; the librosa cross-check now passes `pad_mode="reflect"` explicitly.
- **Why:** Phase 1 has to reproduce the baseline exactly. The setting stays configurable because a streaming C front end has no "edge" to reflect.

### 2026-10-17: Download resumes even when the partial file is already complete (Phase 0)
- **Tried:** if a run stopped after the last byte but before the rename, the next run asked Zenodo for bytes past the end, got HTTP 416, and failed on every retry.
- **Decision:** kept: a 416 on a resume skips straight to the MD5 check, which deletes the file if it's bad. Tests cover both the 206 resume and the 416 case with a fake server.
- **Why:** the zips are 1 to 2 GB; a stuck download shouldn't need someone to delete files by hand.

### 2026-10-17: Cap numpy below 2.5 for Windows Smart App Control (Phase 0)
- **Tried:** `uv sync` resolved numpy 2.5.4; on the Windows dev machine Smart App Control refused to load its `_multiarray_umath` DLL, so nothing that imports numpy (torch, scipy, librosa) worked. numpy 2.3.5 and 2.4.6 load fine with the rest of the locked stack unchanged.
- **Decision:** kept `numpy>=2.0,<2.5`; locked to 2.4.6.
- **Why:** Smart App Control only trusts unsigned binaries once they have cloud reputation, which a brand-new release lacks. The cap can be lifted once a 2.5.x release loads on Windows.

### 2026-10-17: Log-mel in plain NumPy, cross-checked against librosa (Phase 0)
- **Tried:** wrote the baseline front end (1024-point FFT, hop 512, 128 Slaney mel bands, power 2, `10·log10(mel + eps)`) by hand instead of calling librosa.
- **Result:** unit tests on sine waves pass; a test compares it against librosa's `melspectrogram` within 0.01 dB.
- **Decision:** kept. librosa stays a dev-only dependency for that test.
- **Why:** the same steps are ported to C in Phase H4, so the Python reference must be simple and fully understood, and still match the baseline exactly.

### 2026-10-17: CPU-only PyTorch, stdlib WAV reader (Phase 0)
- **Tried:** pinned PyTorch to the CPU wheel index for all platforms; read WAVs with the standard `wave` module.
- **Decision:** kept.
- **Why:** keeps installs and CI small and identical on Windows and Linux; GPU training happens on Colab/Kaggle. DCASE clips are 16-bit mono PCM, so no native audio library is needed.

### 2026-10-17: Resumable download with MD5 check (Phase 0)
- **Tried:** the downloader resumes `.part` files with HTTP Range requests and checks each zip against the MD5 on the Zenodo record.
- **Decision:** kept.
- **Why:** the zips are 1 to 2 GB each; a dropped connection shouldn't mean starting over, and a corrupt zip should fail loudly before any training.
