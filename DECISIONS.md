# Decisions log

One entry per experiment or design choice, newest first. Keep each entry short. This log is the source for the README's "what didn't work" section.

## Template

### YYYY-MM-DD: short title (Phase N)
- **Tried:** what was changed, and the config name
- **Result:** the numbers, taken from results/results_log.csv
- **Decision:** kept or dropped
- **Why:** one or two sentences

---

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
