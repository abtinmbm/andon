"""Export the per-fan GMM (H3's final model) and its test vectors as C headers (Phase H5).

    uv run python -m andon.export_gmm --config configs/h3_gmm.yaml --seed 0 --out firmware/model
    uv run python -m andon.export_gmm --synthetic --out firmware/build/model   (CI)

``--config`` refits the model that was scored once on the eval session: one GMM per fan,
trained on every dev session's normal clips (``h3.training_clips``, the same clips and the
same code path as ``h3.score``), with the given seed. The eval session is never read. Its
test vectors come from each fan's first training clip of the first dev session, so they
check the C front end on the real microphone's audio as well. Both depend on the recordings,
so they are written locally and not committed. ``--synthetic`` fits the same settings on
synthetic hums; it needs no data, so CI runs it.

Three headers are written, then read back and checked against scikit-learn:

- ``andon_frontend.h``: the log-mel front end, the same file ``andon.export_c`` writes.
- ``andon_gmm.h``: for each fan and component, the float32 mean, the upper-triangular
  Cholesky factor ``P`` of the precision matrix (precision = ``P @ P.T``, scikit-learn's
  ``precisions_cholesky_``) packed column by column, and the constant
  ``log weight + sum(log diag P) - D / 2 * log(2 pi)``.
- ``andon_gmm_vectors.h``: per fan, int16 audio, its log-mel frames, the mean spectrum of
  each 64-frame window every 32 frames, and each window's score under every fan's GMM (a
  window scored by another fan's GMM gives the large scores a fault gives).

The arithmetic the C code must reproduce, in float32:

- window mean: the mean of the window's log-mel frames, band by band
- per component: ``y = (x - mean) @ P``, ``log p_k = const_k - 0.5 * sum(y ** 2)``
- score = ``-logsumexp_k(log p_k)``, the negative log-likelihood scikit-learn's
  ``score_samples`` gives with its sign flipped, as ``h3`` scores it
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from andon import export_c, h3, train_idcnn
from andon.data.audio import read_wav
from andon.data.features import SplitFeatures, load_or_extract
from andon.experiment import Config, load_config
from andon.features.logmel import LogMelConfig, logmel
from andon.models.classic import build_detector

GMM_HEADER = "andon_gmm.h"
VECTORS_HEADER = "andon_gmm_vectors.h"
N_WINDOWS = 3  # test windows per fan
TOOL = "andon.export_gmm"
# Header floats are float32, so a score rebuilt from the header differs from scikit-learn's
# float64 one by rounding alone. Measured worst on the synthetic and real-fan exports:
# printed by ``main``.
SCORE_RTOL = 1e-5
SCORE_ATOL = 1e-3


@dataclass
class GmmBank:
    """One full-covariance mixture per fan, in the form the C scorer uses."""

    fans: list[str]
    mean: np.ndarray  # (fans, components, dims)
    chol: np.ndarray  # (fans, components, dims, dims), upper triangular
    log_const: np.ndarray  # (fans, components)

    @classmethod
    def from_mixtures(cls, mixtures: dict[str, GaussianMixture]) -> GmmBank:
        fans = sorted(mixtures)
        for f in fans:
            if mixtures[f].covariance_type != "full":
                raise ValueError(f"{f}: only full-covariance GMMs are exported")
        mean = np.stack([mixtures[f].means_ for f in fans])
        chol = np.stack([mixtures[f].precisions_cholesky_ for f in fans])
        if np.abs(np.tril(chol, -1)).max() != 0:
            raise ValueError("the precision Cholesky factors are not upper triangular")
        dims = mean.shape[-1]
        log_det = np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
        weights = np.stack([mixtures[f].weights_ for f in fans])
        log_const = np.log(weights) + log_det - 0.5 * dims * math.log(2 * math.pi)
        return cls(fans, mean, chol, log_const)

    def score(self, fan: int, x: np.ndarray) -> np.ndarray:
        """Negative log-likelihood of each row of ``x`` under fan ``fan``'s mixture."""
        x = np.asarray(x, dtype=np.float64)
        diff = x[:, None, :] - self.mean[fan][None]  # (n, k, d)
        y = np.einsum("nkd,kde->nke", diff, self.chol[fan])
        return -logsumexp(self.log_const[fan][None] - 0.5 * (y**2).sum(axis=-1), axis=1)

    def packed_chol(self) -> np.ndarray:
        """``P`` column by column, upper triangle only: column ``j`` is ``P[0..j, j]``."""
        dims = self.mean.shape[-1]
        cols = [self.chol[..., : j + 1, j] for j in range(dims)]
        return np.concatenate(cols, axis=-1)

    @classmethod
    def from_packed(
        cls, fans: list[str], mean: np.ndarray, packed: np.ndarray, log_const: np.ndarray
    ) -> GmmBank:
        dims = mean.shape[-1]
        chol = np.zeros((*mean.shape[:2], dims, dims))
        at = 0
        for j in range(dims):
            chol[..., : j + 1, j] = packed[..., at : at + j + 1]
            at += j + 1
        return cls(fans, mean, chol, log_const)


@dataclass
class GmmExport:
    """Everything that goes into the three headers."""

    bank: GmmBank
    source: str  # where the model came from, for the header comment
    audio: np.ndarray  # (fans, samples) int16; row f is fan f's audio
    logmel_cfg: LogMelConfig
    width: int  # frames per window
    hop: int  # frames between windows


def fit_gmms(
    model_cfg: dict,
    seed: int,
    feats: SplitFeatures,
    clip_fans: list[str],
    width: int,
    hop: int,
) -> dict[str, GaussianMixture]:
    """One GMM per fan on the mean spectrum of its training windows, as ``h3.score`` fits it
    (same windows, same order, same detector)."""
    starts, owner = h3.windows(feats, width, hop)
    x = h3.window_means(feats.frames, starts, width)
    ids = np.asarray(clip_fans)[owner]
    cfg = dict(model_cfg, type="gmm")
    if cfg.get("fit_per") != "machine_id":
        raise ValueError("the device scores one fan: the config must fit one GMM per fan")
    return {f: build_detector(cfg, seed).fit(x[ids == f]).mixture for f in sorted(set(ids))}


def audio_len(n_windows: int, width: int, hop: int, cfg: LogMelConfig) -> int:
    """Samples whose centred log-mel has exactly the frames ``n_windows`` windows need."""
    return (width + (n_windows - 1) * hop - 1) * cfg.hop_length


# --- writing ----------------------------------------------------------------------------


def gmm_header(ex: GmmExport) -> str:
    b = ex.bank
    n_fans, n_comp, dims = b.mean.shape
    if dims != ex.logmel_cfg.n_mels:
        raise ValueError(f"the GMM takes {dims} values, the front end gives {ex.logmel_cfg}")
    if ex.width != 2 * ex.hop:
        raise ValueError("the C window keeps two half-window sums: width must be 2 * hop")
    ids = ", ".join(f'"{f}"' for f in b.fans)
    if any(c in ex.source for c in '"\\\n'):
        raise ValueError(f"the source text cannot go in a C string: {ex.source!r}")
    body = [
        # The bench prints this, so a board log says which model was flashed.
        f'static const char andon_gmm_source[] = "{ex.source}";\n',
        export_c._define("ANDON_GMM_N_FANS", n_fans)
        + export_c._define("ANDON_GMM_N_COMPONENTS", n_comp)
        + export_c._define("ANDON_GMM_N_DIM", dims)
        + export_c._define("ANDON_GMM_N_TRI", dims * (dims + 1) // 2)
        + export_c._define("ANDON_GMM_WINDOW_FRAMES", ex.width)
        + export_c._define("ANDON_GMM_WINDOW_HOP", ex.hop),
        f"/* Fan index -> machine ID. */\nstatic const char *const andon_gmm_fan_id[{n_fans}]"
        f" = {{{ids}}};\n",
        "/* Component k of fan f: mean, then P (precision = P P^T) packed column by column,"
        " column j being P[0..j][j]; log_const = log weight + sum(log diag P)"
        " - D / 2 log(2 pi). */\n"
        + export_c.c_array("float", "andon_gmm_mean", b.mean, "[fan][component][dim]")
        + export_c.c_array("float", "andon_gmm_chol", b.packed_chol(), "[fan][component][tri]")
        + export_c.c_array("float", "andon_gmm_log_const", b.log_const, "[fan][component]"),
    ]
    return export_c._guard("ANDON_GMM_H", f"Per-fan GMM: {ex.source}.", body, TOOL)


@dataclass
class Vectors:
    logmel: np.ndarray  # (fans, frames, mels) float32
    window_mean: np.ndarray  # (fans, windows, mels) float64
    score: np.ndarray  # (fans, windows, model fans) float64


def reference_vectors(ex: GmmExport, mixtures: dict[str, GaussianMixture] | None) -> Vectors:
    """Python's front end, windows and scores for each fan's audio. Scores come from
    scikit-learn when the mixtures are given (an export), else from the bank (a read-back)."""
    frames = np.stack([logmel(a.astype(np.float32) / 32768, ex.logmel_cfg) for a in ex.audio])
    starts = train_idcnn.patch_starts(frames.shape[1], ex.width, ex.hop)
    if len(starts) != N_WINDOWS or starts[-1] != (N_WINDOWS - 1) * ex.hop:
        raise ValueError(f"{frames.shape[1]} frames do not hold exactly {N_WINDOWS} windows")
    means = np.stack([h3.window_means(f, starts, ex.width) for f in frames])
    fans = ex.bank.fans
    scores = np.empty((len(fans), len(starts), len(fans)))
    for m, fan in enumerate(fans):
        for f in range(len(fans)):
            if mixtures is not None:
                scores[f, :, m] = -mixtures[fan].score_samples(means[f])
            else:
                scores[f, :, m] = ex.bank.score(m, means[f])
    return Vectors(frames, means, scores)


def vectors_header(ex: GmmExport, v: Vectors) -> str:
    n_fans, n_frames, n_mels = v.logmel.shape
    d = export_c._define
    body = [
        d("ANDON_GV_N_FANS", n_fans)
        + d("ANDON_GV_SAMPLE_RATE", ex.logmel_cfg.sample_rate)
        + d("ANDON_GV_HOP_LENGTH", ex.logmel_cfg.hop_length)
        + d("ANDON_GV_N_MELS", n_mels)
        + d("ANDON_GV_AUDIO_LEN", ex.audio.shape[1])
        + d("ANDON_GV_FRAMES", n_frames)
        + d("ANDON_GV_N_WINDOWS", v.window_mean.shape[1])
        + d("ANDON_GV_WINDOW_FRAMES", ex.width)
        + d("ANDON_GV_WINDOW_HOP", ex.hop),
        "/* Audio of fan f as int16 PCM; the front end sees sample / 32768.0f. */\n"
        + export_c.c_array("int16_t", "andon_gv_audio", ex.audio, "[fan][sample]"),
        "/* Log-mel of each fan's audio, centred frames, reflect-padded at both ends. */\n"
        + export_c.c_array("float", "andon_gv_logmel", v.logmel, "[fan][frame][mel]"),
        f"/* Window w is frames w * {ex.hop} to w * {ex.hop} + {ex.width - 1}: its mean"
        " log-mel, and its score (negative log-likelihood) under every fan's GMM. */\n"
        + export_c.c_array("float", "andon_gv_window_mean", v.window_mean, "[fan][window][mel]")
        + export_c.c_array("float", "andon_gv_score", v.score, "[fan][window][model fan]"),
    ]
    return export_c._guard("ANDON_GMM_VECTORS_H", f"GMM test vectors for {ex.source}.", body, TOOL)


def bank_from_header(text: str) -> GmmBank:
    a, d = export_c.read_header(text)
    shape = (d["ANDON_GMM_N_FANS"], d["ANDON_GMM_N_COMPONENTS"])
    mean = a["andon_gmm_mean"].astype(np.float64).reshape(*shape, d["ANDON_GMM_N_DIM"])
    packed = a["andon_gmm_chol"].astype(np.float64).reshape(*shape, d["ANDON_GMM_N_TRI"])
    log_const = a["andon_gmm_log_const"].astype(np.float64).reshape(shape)
    start = text.index("andon_gmm_fan_id")
    fans = text[text.index("{", start) + 1 : text.index("}", start)].replace('"', "").split(", ")
    return GmmBank.from_packed(fans, mean, packed, log_const)


def verify(out_dir: Path, cfg: LogMelConfig) -> float:
    """Re-read the headers and check the vectors against the model rebuilt from them.

    The front end must reproduce the vectors' log-mel exactly. The header's floats are
    float32, so the scores get a tolerance. Returns the worst relative score difference.
    """
    export_c.verify_frontend(out_dir, cfg)
    g_text = (out_dir / GMM_HEADER).read_text(encoding="utf-8")
    bank = bank_from_header(g_text)
    _, gd = export_c.read_header(g_text)
    tv, d = export_c.read_header((out_dir / VECTORS_HEADER).read_text(encoding="utf-8"))
    n_fans, n_win, n_mels = d["ANDON_GV_N_FANS"], d["ANDON_GV_N_WINDOWS"], d["ANDON_GV_N_MELS"]
    width, hop = gd["ANDON_GMM_WINDOW_FRAMES"], gd["ANDON_GMM_WINDOW_HOP"]
    audio = tv["andon_gv_audio"].reshape(n_fans, -1)
    back = reference_vectors(GmmExport(bank, "", audio, cfg, width, hop), None)
    np.testing.assert_array_equal(back.logmel.ravel(), tv["andon_gv_logmel"])
    np.testing.assert_array_equal(
        back.window_mean.astype(np.float32).ravel(), tv["andon_gv_window_mean"]
    )
    want = tv["andon_gv_score"].astype(np.float64).reshape(n_fans, n_win, n_fans)
    np.testing.assert_allclose(back.score, want, rtol=SCORE_RTOL, atol=SCORE_ATOL)
    if back.window_mean.shape[-1] != n_mels:
        raise ValueError("the vectors' mel count differs from the windows'")
    # The same relative error the C self-test prints: |difference| / max(|score|, 1).
    return float((np.abs(back.score - want) / np.maximum(np.abs(want), 1.0)).max())


def write(ex: GmmExport, mixtures: dict[str, GaussianMixture], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / export_c.FRONTEND_HEADER, out_dir / GMM_HEADER, out_dir / VECTORS_HEADER]
    texts = [
        export_c.frontend_header(ex.logmel_cfg),
        gmm_header(ex),
        vectors_header(ex, reference_vectors(ex, mixtures)),
    ]
    for path, text in zip(paths, texts, strict=True):
        path.write_text(text, encoding="utf-8", newline="\n")
    verify(out_dir, ex.logmel_cfg)
    return paths


# --- sources ----------------------------------------------------------------------------


def _window(cfg: Config) -> tuple[int, int]:
    return int(cfg.raw["window"]["frames"]), int(cfg.raw["window"]["hop"])


def synthetic_export(seed: int = 0) -> tuple[GmmExport, dict[str, GaussianMixture]]:
    """``h3_gmm``'s settings fitted on three synthetic fans: hums at their own frequency with
    harmonics and noise, 40 s each for training and separate audio for the vectors."""
    src = load_config(export_c.CONFIGS / "h3_gmm.yaml")
    width, hop, cfg = *_window(src), src.logmel
    rng = np.random.default_rng(seed)

    def fan(k: int, seconds: float) -> np.ndarray:
        t = np.arange(int(cfg.sample_rate * seconds)) / cfg.sample_rate
        f0 = 150.0 * (k + 1)
        y = sum(0.2 / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) for h in (1, 2, 3))
        y = y * (1 + 0.1 * np.sin(2 * np.pi * 0.3 * t)) + 0.02 * rng.standard_normal(t.size)
        return np.round(np.clip(y, -1, 1) * 32767).astype(np.int16)

    fans = [f"id_{k:02d}" for k in range(3)]
    train = [logmel(fan(k, 40.0).astype(np.float32) / 32768, cfg) for k in range(3)]
    offsets = np.r_[0, np.cumsum([len(f) for f in train])].astype(np.int64)
    feats = SplitFeatures(frames=np.concatenate(train), offsets=offsets)
    mixtures = fit_gmms(src.raw["model"], seed, feats, fans, width, hop)
    n = audio_len(N_WINDOWS, width, hop, cfg)
    audio = np.stack([fan(k, n / cfg.sample_rate + 1)[:n] for k in range(3)])
    ex = GmmExport(
        GmmBank.from_mixtures(mixtures), f"synthetic, seed {seed}", audio, cfg, width, hop
    )
    return ex, mixtures


def real_export(cfg: Config, seed: int) -> tuple[GmmExport, dict[str, GaussianMixture]]:
    """The eval fold's model (trained on every dev session) for ``seed``, with vectors from
    each fan's first training clip of the first dev session. The eval session is not read."""
    if cfg.model != "gmm" or cfg.raw["model"].get("covariance_type") != "full":
        raise ValueError("only a full-covariance per-fan GMM config can be exported")
    width, hop = _window(cfg)
    recording = h3.read_recording(cfg)
    sessions = tuple(recording["sessions"]["dev"])
    clips = h3.training_clips(cfg, recording, h3.read_manifest(cfg.data_root), sessions)
    feats = load_or_extract(
        [c.clip for c in clips], cfg.logmel, h3.CACHE_DIR, f"realfan_h5_{'+'.join(sessions)}"
    )
    fans = [c.clip.machine_id for c in clips]
    mixtures = fit_gmms(cfg.raw["model"], seed, feats, fans, width, hop)
    n = audio_len(N_WINDOWS, width, hop, cfg.logmel)
    audio, names = [], []
    for fan in sorted(mixtures):
        clip = next(c.clip for c in clips if c.session == sessions[0] and c.clip.machine_id == fan)
        samples, _ = read_wav(clip.path)
        pcm = np.round(samples * 32768).astype(np.int16)
        if not np.array_equal(pcm.astype(np.float32) / 32768, samples.astype(np.float32)):
            raise ValueError(f"{clip.path} is not 16-bit PCM")
        audio.append(pcm[:n])
        names.append(clip.path.relative_to(cfg.data_root).as_posix())
    source = f"{cfg.name}, trained on {'+'.join(sessions)}, seed {seed}; vectors from " + ", ".join(
        names
    )
    bank = GmmBank.from_mixtures(mixtures)
    return GmmExport(bank, source, np.stack(audio), cfg.logmel, width, hop), mixtures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the per-fan GMM as C headers.")
    parser.add_argument("--synthetic", action="store_true", help="synthetic fans, no data")
    parser.add_argument("--config", type=Path, help="a per-fan GMM config (configs/h3_gmm.yaml)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.synthetic == bool(args.config):
        parser.error("give exactly one of --synthetic or --config")
    if args.config:
        ex, mixtures = real_export(load_config(args.config), args.seed)
    else:
        ex, mixtures = synthetic_export(args.seed)
    for path in write(ex, mixtures, args.out):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    worst = verify(args.out, ex.logmel_cfg)
    n_fans, n_comp, dims = ex.bank.mean.shape
    floats = n_fans * n_comp * (dims + dims * (dims + 1) // 2 + 1)
    print(
        f"verified: {n_fans} fans x {N_WINDOWS} windows x {n_fans} models; worst relative"
        f" score difference, float32 header vs scikit-learn: {worst:.2e}"
    )
    print(f"model data: {floats * 4 / 1024:.1f} KB float32 ({ex.source})")


if __name__ == "__main__":
    main()
