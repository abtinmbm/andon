`phase5_andon` on dev_test, seeds 0, 1, 2: mean ± sample std over seeds.

| Machine type | False alarms per hour | Faults caught (%) | Mean delay to red (s) | Normal audio (h) | Faults |
| --- | --- | --- | --- | --- | --- |
| ToyCar | 0.47 ± 0.47 | 41.67 ± 3.71 | 15.44 ± 0.47 | 2.14 | 352 |
| ToyConveyor | 1.30 ± 0.35 | 32.25 ± 1.41 | 14.03 ± 0.54 | 3.34 | 369 |
| fan | 2.40 ± 1.03 | 65.44 ± 1.94 | 7.54 ± 0.10 | 0.56 | 490 |
| pump | 2.99 ± 2.74 | 76.82 ± 2.89 | 9.30 ± 0.43 | 0.56 | 151 |
| slider | 17.37 ± 5.78 | 94.46 ± 0.20 | 8.20 ± 0.29 | 0.56 | 295 |
| valve | 2.40 ± 2.74 | 7.13 ± 1.58 | 17.59 ± 1.92 | 0.56 | 159 |
| **Mean of six types** | 4.49 ± 0.56 | 52.96 ± 0.99 | 12.02 ± 0.32 | | |

`phase5_andon` on eval, seeds 0, 1, 2: mean ± sample std over seeds.

| Machine type | False alarms per hour | Faults caught (%) | Mean delay to red (s) | Normal audio (h) | Faults |
| --- | --- | --- | --- | --- | --- |
| ToyCar | 0.29 ± 0.50 | 57.32 ± 3.27 | 13.40 ± 0.04 | 1.15 | 264 |
| ToyConveyor | 2.80 ± 0.69 | 35.40 ± 2.71 | 14.46 ± 0.65 | 0.83 | 354 |
| fan | 9.34 ± 2.34 | 94.27 ± 1.87 | 6.68 ± 0.14 | 0.43 | 343 |
| pump | 14.38 ± 7.19 | 92.78 ± 4.24 | 7.25 ± 0.51 | 0.42 | 157 |
| slider | 6.39 ± 1.39 | 91.90 ± 3.11 | 7.51 ± 0.43 | 0.42 | 177 |
| valve | 0.00 ± 0.00 | 21.75 ± 2.22 | 14.13 ± 0.73 | 0.42 | 213 |
| **Mean of six types** | 5.53 ± 1.79 | 65.57 ± 1.02 | 10.57 ± 0.24 | | |

`phase5_andon_log` on dev_test, seeds 0, 1, 2: mean ± sample std over seeds.

| Machine type | False alarms per hour | Faults caught (%) | Mean delay to red (s) | Normal audio (h) | Faults |
| --- | --- | --- | --- | --- | --- |
| ToyCar | 0.00 ± 0.00 | 0.09 ± 0.16 | nan ± nan | 2.14 | 352 |
| ToyConveyor | 0.00 ± 0.00 | 16.98 ± 1.13 | 16.20 ± 0.57 | 3.34 | 369 |
| fan | 0.00 ± 0.00 | 62.11 ± 1.25 | 7.72 ± 0.35 | 0.56 | 490 |
| pump | 0.00 ± 0.00 | 43.93 ± 2.02 | 9.22 ± 0.34 | 0.56 | 151 |
| slider | 1.80 ± 1.80 | 26.10 ± 2.90 | 13.77 ± 0.76 | 0.56 | 295 |
| valve | 0.00 ± 0.00 | 0.00 ± 0.00 | nan ± nan | 0.56 | 159 |
| **Mean of six types** | 0.30 ± 0.30 | 24.87 ± 0.72 | nan ± nan | | |

nan: in at least one seed no fault was caught, so there is no delay to average.

`phase5_andon_1clip` on dev_test, seeds 0, 1, 2: mean ± sample std over seeds.

| Machine type | False alarms per hour | Faults caught (%) | Mean delay to red (s) | Normal audio (h) | Faults |
| --- | --- | --- | --- | --- | --- |
| ToyCar | 0.47 ± 0.47 | 19.39 ± 2.56 | 6.59 ± 0.07 | 2.14 | 1059 |
| ToyConveyor | 1.30 ± 0.35 | 13.15 ± 0.39 | 6.01 ± 0.14 | 3.34 | 1110 |
| fan | 2.40 ± 1.03 | 47.71 ± 1.42 | 4.06 ± 0.06 | 0.56 | 1475 |
| pump | 2.99 ± 2.74 | 49.49 ± 1.08 | 4.33 ± 0.06 | 0.56 | 456 |
| slider | 17.37 ± 5.78 | 64.72 ± 2.11 | 4.57 ± 0.04 | 0.56 | 890 |
| valve | 2.40 ± 2.74 | 2.30 ± 0.91 | 7.10 ± 0.44 | 0.56 | 479 |
| **Mean of six types** | 4.49 ± 0.56 | 32.79 ± 0.41 | 5.44 ± 0.07 | | |
