# Andon

Machine-sound anomaly detection. Andon learns what a healthy machine sounds like from normal recordings only, then raises a green / yellow / red andon signal when it starts to sound wrong. Stage 2 runs the same pipeline on a Raspberry Pi Pico 2 that cuts power to a real fan when it hears a fault.

> Work in progress. Results tables, the dashboard GIF and the demo video are added as each phase is finished. See [PLAN.md](PLAN.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest -q
uv run python -m andon.data.download fan ToyConveyor
```

The download fetches the DCASE 2020 Task 2 development data for two machine types (about 3.3 GB) into `data/`, checks it, and prints the number of clips per machine ID. Use `all` to fetch all six machine types (8.1 GB).

## Data

DCASE 2020 Task 2 development dataset, built from ToyADMOS and MIMII, licensed CC BY-NC-SA 4.0: https://zenodo.org/records/3678171

## Licence

Code: MIT. Data: see above.
