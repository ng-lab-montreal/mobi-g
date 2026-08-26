# Mobi-G Technical Validation

Technical validation and descriptive statistics code for the **Mobility in
Geriatrics (Mobi-G)** dataset — a wearable accelerometry dataset with labeled
postures and steps, collected from older adults across inpatient, emergency,
and outpatient settings at the Centre hospitalier de l'Université de Montréal
(CHUM).

This repository contains only the analysis code used to produce the
technical validation and descriptive statistics reported in the accompanying
data descriptor paper. The dataset itself is not included here — see
**Data availability** below.

## Contents

- `code/technical_validation.py` — sample rate/continuity, accelerometer
  calibration, saturation/outlier detection, axis statistics per posture and
  step, and prespecified per-participant axis checks.
- `code/descriptive_statistics.py` — recording duration and descriptive
  tables (participant characteristics, recording counts by activity and
  stratification variable, per-participant activity/step summary).

## Requirements

- Python ≥ 3.10
- Dependencies pinned in `requirements.txt`:

  ```
  pip install -r requirements.txt
  ```

## Data availability

These scripts expect the **labeled** dataset (`data_with_label/`) and
`participants.csv`, as described in the data descriptor paper's Data
Availability section. Download the dataset from doi:10.5683/SP4/BL2YTA and place it locally, or point the scripts at it
via environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `MOBI_G_DATA_ROOT` | Path to the data directory (must contain `data_with_label/`) | `<project_root>/data` |
| `MOBI_G_PARTICIPANTS_CSV` | Full path to `participants.csv` | `<project_root>/data/participants.csv` |

If you don't set these, place the dataset under `./data/` relative to this
repository's root:

```
mobi-g/
├── code/
│   ├── technical_validation.py
│   └── descriptive_statistics.py
└── data/
    ├── data_with_label/
    │   └── P01/, P02/, ...
    └── participants.csv
```

## Running

From the `code/` directory:

```bash
python technical_validation.py
python descriptive_statistics.py
```

Each script prints a progress log to the console and writes its outputs to
directories created next to the script itself, inside `code/`:

- `code/technical_validation/` and `code/figures/technical_validation/`
- `code/descriptive_statistics/`

## Citing

If you use this code, please cite the accompanying data descriptor paper —
see `CITATION.cff`.

## License

Released under CC BY-NC 4.0 — see `LICENSE`. Non-commercial use with
attribution.
