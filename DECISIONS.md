# Decisions log

One entry per experiment or design choice, newest first. Keep each entry short. This log is the source for the README's "what didn't work" section.

## Template

### YYYY-MM-DD: short title (Phase N)
- **Tried:** what was changed, and the config name
- **Result:** the numbers, taken from results/results_log.csv
- **Decision:** kept or dropped
- **Why:** one or two sentences

---

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
