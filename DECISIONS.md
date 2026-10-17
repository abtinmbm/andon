# Decisions log

One entry per experiment or design choice, newest first. Keep each entry short. This log is the source for the README's "what didn't work" section.

## Template

### YYYY-MM-DD: short title (Phase N)
- **Tried:** what was changed, and the config name
- **Result:** the numbers, taken from results/results_log.csv
- **Decision:** kept or dropped
- **Why:** one or two sentences

---

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
