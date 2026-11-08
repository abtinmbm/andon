| Session | Split | Fan | Fault | Clips | Minutes | Median RMS (dBFS) | Max peak (dBFS) | DC (LSB, mean) | Clipped samples |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot | other | id_00 | none | 12 | 2.0 | -63.0 | -47.8 | -0.5 | 0 |
| pilot | other | id_00 | imbalance | 6 | 1.0 | -60.3 | -47.4 | -0.5 | 0 |
| pilot | other | id_00 | obstruction | 6 | 1.0 | -51.8 | -39.3 | -0.5 | 0 |
| pilot | other | id_00 | rattle | 18 | 3.0 | -62.2 | -34.6 | -0.5 | 0 |
| pilot | other | id_01 | none | 6 | 1.0 | -29.7 | -15.0 | -0.7 | 0 |
| pilot | other | id_01 | imbalance | 6 | 1.0 | -29.4 | -16.0 | -0.6 | 0 |
| pilot | other | id_01 | obstruction | 6 | 1.0 | -21.8 | -9.6 | -0.6 | 0 |

Stream, from the take logs (frames received; breaks; frames missing; corrupt; seconds of partial clips discarded):

| Session | Takes | Frames | Breaks | Missing | Corrupt | Discarded (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot | 9 | 37511 | 0 | 0 | 0 | 0.00 |

Fault takes against normal running of the same fan in the same session: mean band level of the take minus that of the normal clips, in dB, at gain 0. Near 0 in every band means the fault does not change the sound. The last column is the range over the take's clips in the band that changed most, to show whether the change is steady or comes from a few clips.

| Session | Fan | Fault | Take | 0-125 Hz | 125-250 Hz | 250-500 Hz | 500-1000 Hz | 1000-2000 Hz | 2000-4000 Hz | 4000-8000 Hz | Per clip, biggest band |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pilot | id_00 | imbalance | pilot-id_00-imbalance-20261107T231520Z | +8.0 | +0.3 | +2.1 | +0.8 | +0.9 | +0.1 | +0.2 | 0-125 Hz: +7.0 to +8.7 |
| pilot | id_00 | obstruction | pilot-id_00-obstruction-20261107T224433Z | +5.0 | +10.3 | +14.1 | +14.1 | +6.2 | +7.0 | +9.8 | 500-1000 Hz: +13.7 to +14.3 |
| pilot | id_00 | rattle | pilot-id_00-rattle-20261107T233213Z | +0.8 | -0.1 | -0.4 | -2.7 | +0.4 | -0.5 | -0.5 | 500-1000 Hz: -3.6 to -2.0 |
| pilot | id_00 | rattle | pilot-id_00-rattle-20261107T234602Z | +3.3 | +0.3 | -0.1 | +0.0 | +1.4 | +0.0 | -0.1 | 0-125 Hz: +3.0 to +3.9 |
| pilot | id_00 | rattle | pilot-id_00-rattle-20261107T235159Z | +10.2 | +1.2 | +0.3 | +1.0 | +3.0 | +1.3 | +1.1 | 0-125 Hz: +2.4 to +17.1 |
| pilot | id_01 | imbalance | pilot-id_01-imbalance-20261108T010739Z | +0.4 | +3.4 | -0.1 | +0.0 | -0.2 | +0.2 | +0.2 | 125-250 Hz: +3.1 to +3.9 |
| pilot | id_01 | obstruction | pilot-id_01-obstruction-20261108T005655Z | +1.0 | +7.4 | +6.7 | +10.4 | +4.0 | +4.5 | +4.2 | 500-1000 Hz: +10.2 to +10.6 |

Listening check (1 random clips per row, seed 0):

- [ ] `pilot/anomaly_id_00_00000009.wav` (imbalance)
- [ ] `pilot/normal_id_00_00000006.wav` (none)
- [ ] `pilot/anomaly_id_00_00000000.wav` (obstruction)
- [ ] `pilot/anomaly_id_00_00000020.wav` (rattle)
- [ ] `pilot/anomaly_id_01_00000010.wav` (imbalance)
- [ ] `pilot/normal_id_01_00000003.wav` (none)
- [ ] `pilot/anomaly_id_01_00000003.wav` (obstruction)

Problems: 0
