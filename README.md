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
  calibration (static norm within ±5% of 9.81 m/s²), saturation/outlier
  detection, axis statistics per posture and step (mean, SD, 1st and 99th
  percentiles of absolute acceleration), prespecified per-participant axis
  checks, and average step waveforms of the acceleration norm by protocol,
  gait speed and walking aid.
- `code/descriptive_statistics.py` — recording duration and descriptive
  tables (participant characteristics, recording counts by activity and
  stratification variable, per-participant activity/step summary).
- `code/gotov_comparison.py` — comparison with a published dataset: the axis
  statistics per posture reproduced on GOTOV (healthy adults over 60, ankle
  accelerometer) with the same definitions, after resampling to 52 Hz and
  mapping to the Mobi-G axes; the axis mapping is checked on the data.

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
| `GOTOV_DATA_ROOT` | Folder with `Activity_Measurements.zip` and `GOTOV_DataSummary.csv` from GOTOV | `<project_root>/data/gotov` |

The GOTOV comparison additionally needs two files from the GOTOV dataset
(4TU.ResearchData, doi:10.4121/12716081, v2; Paraschiakos et al. 2020,
doi:10.1007/s11257-020-09268-2): `Activity_Measurements.zip` (kept zipped)
and `GOTOV_DataSummary.csv`.

If you don't set these, place the data under `./data/` relative to this
repository's root:

```
mobi-g/
├── code/
│   ├── technical_validation.py
│   ├── descriptive_statistics.py
│   └── gotov_comparison.py
└── data/
    ├── data_with_label/
    │   └── P01/, P02/, ...
    ├── participants.csv
    └── gotov/
        ├── Activity_Measurements.zip
        └── GOTOV_DataSummary.csv
```

## Running

From the `code/` directory:

```bash
python technical_validation.py
python descriptive_statistics.py
python gotov_comparison.py   # after technical_validation.py; needs the GOTOV files
```

Each script prints a progress log to the console and writes its outputs to
directories created next to the script itself, inside `code/`:

- `code/technical_validation/` and `code/figures/technical_validation/`
- `code/descriptive_statistics/`
- `code/gotov_comparison/`

Outputs that appear in the data descriptor paper are named after it:

| File | Paper |
|---|---|
| `technical_validation/table2_posture_axis_stats.csv`, `table2_step_axis_stats.csv` | Table II |
| `gotov_comparison/table3_mobig_vs_gotov.csv` | Table III |
| `figures/technical_validation/fig3_step_waveforms.png` (data: `technical_validation/fig3_step_waveforms.csv`, `fig3_step_waveform_groups.csv`) | Figure 3 |

## Citing

If you use this code, please cite the accompanying data descriptor paper —
see `CITATION.cff`.

## License

Released under CC BY-SA 4.0 — see `LICENSE`. Share and adapt, even
commercially, as long as attribution is given and derivatives are shared
under the same license.
