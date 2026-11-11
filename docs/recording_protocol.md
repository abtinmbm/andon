# Real-fan recording protocol (Phase H2)

This protocol fixes how the real-fan dataset is recorded and labelled. Every session follows it
exactly, so recordings from different days and of different fans can be pooled, and the last
session can serve as a held-out eval set. The settings that software controls are in
[`configs/h2_recording.yaml`](../configs/h2_recording.yaml). Everything physical is written down
here. Measurements marked *(pilot)* are filled in from the pilot session, before session s1, and
are not changed afterwards.

## What is recorded

| | |
| --- | --- |
| Machines | Three 5 V, 40 mm, 2-wire fans, one per machine ID: `id_00` Sunon MF40100V2, `id_01` and `id_02` ANVISION YDM4010B05. The two ANVISION fans are marked **A** and **B** on the frame, so they can't be swapped by mistake. |
| Speed | 100 % only (the 50 % speed is not recorded). |
| Sessions | `s1` and `s2` are **dev** (H3 trains on their normal clips and makes every choice on their faults). `s3` is **eval**, scored once per final model. `pilot` is for the level check only and is never used for training or scoring. Each session is on a different day. |
| Per fan, per session | 30 min of normal running, then 3 min of each of the two faults, `imbalance` and `obstruction`. From s2 on, then a 3 min **control take** of normal running after the fan is lifted off its tape and put back (see "Running a session"). |
| Totals | 1.5 h of normal running per fan in the main takes plus 6 min of control takes (s2, s3), and 9 min of each fault per fan, across s1–s3. |
| Clips | 10 s, 16 kHz, mono, 16-bit WAV, the same format as the DCASE 2020 Task 2 clips. |

## Signal chain

- **Mic.** Adafruit ICS-43434 breakout (I2S, 24-bit). SEL is tied to GND, so it is the left channel. It is read by a PIO program and DMA on the Pico 2 (`firmware/src/mic.pio`, `mic.c`). BCLK is 1.024 MHz, which is 64 × 16 kHz. At 16 kHz the mic runs in its low-power mode (6.25 to 18.75 kHz), and its datasheet (DS-000069 v1.2) gives the spec at this rate: sensitivity −26 dBFS at 94 dB SPL, SNR 64 dBA, and a high-pass filter at 24 Hz that removes DC.
- **24 bits to 16.** The board keeps the top 16 + `gain_shift` bits of each 24-bit sample and clips to the int16 range (`mic_word_to_sample` in `firmware/src/mic_stream.c`). This is exactly the conversion the C front end will use in H4–H6, so the WAV files hold the samples the device will see. `gain_shift` is **4** (×16, +24 dB), fixed from the pilot: at gain 0 the loudest peak was −39.3 dBFS (obstruction on `id_00`), so gain 4 puts it near −15 dBFS. The ANVISION fans are louder: on `id_01` at gain 4, obstruction peaks at −9.6 dBFS with no clipping, which leaves 9.6 dB of headroom (`results/h2_pilot.md`).
- **Transport.** The samples go over USB serial as frames of 256 samples, each with a sequence number and a CRC (`mic_stream.h`). `andon.record` checks both. A clip is kept only if all of its frames arrived in order. A clip broken by a missing or corrupt frame is discarded, and the take's log counts it.

## Physical setup *(pilot; then fixed)*

| Item | Setting |
| --- | --- |
| Room | A bedroom, with the window and the door closed. |
| Fan mount | Everything stands on one wooden plank. The breadboard is stuck to it by its adhesive back. The fan stands upright at the plank's end, 1 cm past the breadboard's bottom end and lined up with the mic, on a strip of double-sided tape that stays on the plank, so every fan lands on the same spot. Both faces are open, and the exhaust blows off the end of the plank. |
| Fan orientation | The intake (the open side, without the logo) faces the mic, and the air blows away from it. The wires leave at the top. |
| Mic position | On the breadboard, rows 56–61, column c (bench guide, H2 step 2), sound hole facing up. The breadboard is stuck to the same plank by its adhesive back, with a 1 cm gap (measured) between its row 63 end and the fan, so vibration does not travel through the board. |
| Mic to fan distance | About 2.2 cm (measured with a ruler) from the mic's sound hole to the fan's open face. |
| Surface | The plank lies flat on a desk. |
| Background | Normal room noise is allowed, but no music, speech close by or other machines on purpose. Anything unusual goes in the session notes below. |

Before each session, check the setup against this table and compare it with the reference photos, `C:\andon-photos\h2-setuptop.jpg`, `h2-setupside.jpg`, `h2-setupfront.jpg` and `h2-setupback.jpg` (kept out of git).

## Order within a session

To keep the time of day and the room's state from lining up with one fan, the fan order rotates:

| Session | Order |
| --- | --- |
| pilot | id_00 only |
| s1 | id_00, id_01, id_02 |
| s2 | id_01, id_02, id_00 |
| s3 | id_02, id_00, id_01 |

## Running a session

1. Flash `andon.uf2` (H2 build). Check that the light is green and the fan is off.
2. For each fan, in this session's order, run:

   | Session | Fan order |
   | --- | --- |
   | `pilot` | `id_00` only, plus `id_01` for the gain check |
   | `s1` | `id_00`, `id_01`, `id_02` |
   | `s2` | `id_01`, `id_02`, `id_00` |
   | `s3` | `id_02`, `id_00`, `id_01` |

   The first fan rotates, so no fan is always recorded first or last, when the room, the board and the mounting tape are in a different state.


   ```powershell
   uv run python -m andon.record --port COM6 --session s1 --fan id_00
   ```

   The recorder asks you to wire the fan. Then it records the normal take: fan on, 5 s to settle, 30 min, fan off. After that it asks for each fault in turn, records 3 min with the fault in place, and asks you to undo it. The fan is always off while you change anything. Wait for the blades to stop before touching them.
3. **Control take (s2 and s3), straight after that fan's last fault is undone.** With the fan off and the blades stopped, lift the fan off its tape and press it back on the same spot, as when swapping fans. Then record 3 min of normal running:

   ```powershell
   uv run python -m andon.record --port COM6 --session s2 --fan id_01 --condition normal --minutes 3
   ```

   Every fault take follows a stop and handling of the fan, and the main normal take doesn't, so without this take H3 couldn't tell a fault from the fan having been handled and remounted. Its clips are ordinary normal clips of their own take (the take name in the manifest tells them apart). Why it was added: `results/h3_pilot.md` and DECISIONS.md (2026-11-10).
4. Write anything unusual in the session notes below.
5. Afterwards, run the check (step "Checking" below).

## Faults

All faults are applied with the fan off and the blades stopped, and fully undone after their take. Each fault uses the same piece, size and place in every session and on every fan. If a take grinds or scrapes, a blade hits something, or the fan moves on the plank, it is stopped at once with Ctrl+C, which turns the fan off.

| Fault | How | Check | Undo |
| --- | --- | --- | --- |
| `imbalance` | A 5 × 5 mm square of electrical tape stuck flat on the intake-side face of one blade, near the tip, not folded over the blade's edge or tip. A washer is too heavy for a 40 mm fan. | Turn the blades once by hand: nothing catches or scrapes. | Peel it off and check that no residue is left. |
| `obstruction` | A reusable strip of index card, about 4 × 1.3 cm, laid flat on the frame over the bottom third of the intake side, taped to the frame's corners at its two ends only. | Turn the blades by hand: they must not touch the card. | Peel the two pieces of tape and lift the strip off. |

`rattle` (a loose mount: one bottom corner of the fan lifted off the tape by a shim) was tried in the pilot and dropped before s1. With paper, then a dime, then a nickel as the shim, it sounded the same as normal running. Measured below 125 Hz against normal, per 10 s clip: paper +0.8 dB on average, the dime +3.0 to +3.9 dB, and the nickel +2.4 to +17.1 dB, so not steady. Imbalance, for comparison, is +7.0 to +8.7 dB in every clip (`results/h2_pilot.md`). A contact fault (paper ticking the blades) was considered instead and not used, because its strength would depend on exactly where the paper sits each session. The rattle clips stay in `pilot/`, listed under `retired_faults` in the config.

## Labels and files

- **Files.** `data/realfan/<session>/<condition>_<fan_id>_<NNNNNNNN>.wav`, where `condition` is `normal` or `anomaly`, the same pattern as the DCASE file names. The numbering continues across takes and is never reused. The recorder refuses to overwrite a file.
- **Manifest.** `data/realfan/manifest.csv` has one row per clip: `file, session, fan_id, fan_model, condition, fault, speed_pct, gain_shift, firmware_build, take, clip_in_take, recorded_utc`. The fault type is a manifest column, not part of the file name.
- **Take logs.** `data/realfan/<session>/logs/<take>.txt` holds everything the board printed, and a header with the stream statistics.
- **Not in git.** The dataset lives in `data/` (gitignored). Back up the whole `data/realfan` folder after each session. Whether it is published is decided in H7.

## Checking

```powershell
uv run python -m andon.realfan check --listen 2 --out results/h2_dataset.md
```

This checks every clip against the manifest and the config, adds up stream breaks, and measures each clip's level and clipping. It exits with an error if anything is wrong. With `--listen 2` it also lists two random clips per session, fan and condition. Listen to each one and tick it. A normal clip should be a steady fan sound with no clicks, dropouts or buzz, and each fault should sound different from normal.

## Session notes

| Session | Date | Notes |
| --- | --- | --- |
| pilot | | |
| s1 | | |
| s2 | | |
| s3 | | |
