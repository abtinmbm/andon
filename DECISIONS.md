# Decisions log

One entry per experiment or design choice, newest first. Keep each entry short. This log is the source for the README's "what didn't work" section.

## Template

### YYYY-MM-DD: short title (Phase N)
- **Tried:** what was changed, and the config name
- **Result:** the numbers, taken from results/results_log.csv
- **Decision:** kept or dropped
- **Why:** one or two sentences

---

### 2026-10-18: GMM and Mahalanobis on the mean log-mel spectrum beat the autoencoder on 5 of 6 types (Phase 2)
- **Tried:** Mahalanobis distance, a Gaussian mixture and an isolation forest, each on one 128-value vector per clip (the mean of its log-mel frames), fitted on normal train clips only. Configs `phase2_{mahalanobis,gmm,iforest}_{type,id}`, seeds 0, 1, 2, dev test. Table: `results/phase2_comparison.md`.
- **Result (AUC %, mean ± sample std over 3 seeds; mean of the six types in the last figure):** per-ID GMM ToyCar 93.80 ± 0.84, ToyConveyor 88.18 ± 0.84, fan 84.01 ± 0.19, pump 86.53 ± 0.99, slider 87.54 ± 0.75, valve 59.43 ± 0.05; mean 83.25 ± 0.22 against the autoencoder's 73.80 ± 0.14. Per-type GMM: mean 81.72 ± 0.08 (best on pump 86.65 and slider 91.18). Per-ID Mahalanobis: mean 80.34. The autoencoder still wins valve (65.96 ± 0.82); every classic detector is at 50.06 to 59.43 there. Isolation forest is the weakest (mean 57.98 per type, 63.26 per ID), below the autoencoder on every type except pump per ID (75.39). On pAUC the per-type GMM has the best mean (66.05 ± 0.07), level with per-ID GMM (65.95 ± 0.82); the autoencoder's is 59.93 ± 0.31.
- **Decision:** reported as a finding. Per-ID GMM and per-ID Mahalanobis are the strongest cheap models, so they are the fallback candidates for the microcontroller (Phase 4) and the bar Phase 3 has to clear, not just the autoencoder.
- **Why:** a 10 s average keeps the steady spectral shape, which is where most of these anomalies show. Valve faults are likely short clicks that the average washes out, which a frame-level model can still see. The isolation forest's random axis-aligned splits likely suit 128 strongly correlated bands poorly. Neither explanation was tested.

### 2026-10-18: Fit classic detectors both per machine type and per machine ID (Phase 2)
- **Tried:** every detector twice: one model on all IDs of a type pooled (as the autoencoder is trained), and one model per ID, scoring each test clip with its own ID's model (the ID is in every file name).
- **Result (mean AUC % over the six types, per type → per ID):** Mahalanobis 74.96 → 80.34, GMM 81.72 → 83.25, isolation forest 57.98 → 63.26. Per ID is not better everywhere: GMM on slider 91.18 → 87.54. Per-ID GMM seeds spread more (pump pAUC ± 3.37 against ± 0.09 pooled), since each mixture sees a third to a quarter of the data (3 or 4 IDs per type).
- **Decision:** both kept in the table (PM call). Per type is the like-for-like comparison with the autoencoder; per ID matches the hardware stage, which monitors one fan.
- **Why:** reporting per ID alone would mix two changes (the model and the grouping) in one comparison.

### 2026-10-18: Classic detector settings fixed up front, no sweep (Phase 2)
- **Tried:** Mahalanobis with a Ledoit-Wolf shrunk covariance (about 1,000 clips per ID for 128 bands make the plain sample covariance noisy in the directions the distance weights most); GMM with 4 full-covariance components and `reg_covar` 0.01 dB²; isolation forest with 100 trees and scikit-learn's default 256 samples per tree. Score = squared distance, negative log-likelihood, and negated `score_samples` respectively, so higher means more anomalous.
- **Result:** Mahalanobis is deterministic, so its three seeds give identical rows (std 0.00); they are logged anyway so every config has 3 seeds. Fits take under 1.5 s per machine type; all 108 runs (6 configs × 6 types × 3 seeds) took about 3 minutes, almost all of it loading the feature cache.
- **Decision:** kept as the Phase 2 reference settings. Tuning them (more GMM components, adding the per-band std) is left to Phase 3, one change at a time on dev test.
- **Why:** Phase 2 is a reference point. Tuning six configs before there was a first table would have been scope the plan didn't ask for.

### 2026-10-18: scikit-learn becomes a runtime dependency (Phase 2)
- **Tried:** moved `scikit-learn>=1.9.1` from the dev group to the runtime dependencies (PM approval). The lock is unchanged at 1.9.1, which already loaded under Smart App Control in the Phase 1 metric tests.
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
- **Decision:** kept, all set in the config (PM call before the runs).
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
