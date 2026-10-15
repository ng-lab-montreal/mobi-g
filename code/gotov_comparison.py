"""
Comparison with a published dataset (Table III): the axis statistics per
posture (Table II) reproduced on GOTOV ankle data, interleaved with the
Mobi-G rows.

Same definitions as technical_validation.py::_axis_stats (copied verbatim
below): per recording, mean / SD / P1 / P99 of |axis|, then averaged across
recordings.

GOTOV: 4TU.ResearchData doi:10.4121/12716081 (v2); Paraschiakos et al.,
UMUAI 2020, doi:10.1007/s11257-020-09268-2. GENEActiv on the right ankle,
±8 g, 83.3 Hz, units g. Download Activity_Measurements.zip and
GOTOV_DataSummary.csv from the 4TU record into GOTOV_DATA_ROOT; CSVs are read
straight from the zip (no extraction). Run technical_validation.py first —
its Table II output (table2_posture_axis_stats.csv) is read for the Mobi-G rows.

Environment variables:
  GOTOV_DATA_ROOT   folder with the two GOTOV files
                    default: <project_root>/data/gotov

Outputs saved to  gotov_comparison/  (Table III: table3_mobig_vs_gotov.csv)

Recording unit = participant × contiguous label block. Per block:
  1. trim 3 s at each end (labels are imputed from researcher-timed start/end)
  2. saturation screening on raw samples: |value| >= 8 g (GENEActiv range)
     is clipped, >= 95% of 8 g is near-limit
  3. g → m/s² (×9.81), anti-aliased polyphase resampling 83.33 → 52 Hz
  4. fixed axis mapping to the Mobi-G frame: Mobi-G x (tibial) = GA y,
     y (AP) = GA x, z (ML) = GA z. No sign harmonization needed — all
     statistics are on |axis|. The mapping is checked on the data: for each
     participant, the axis with the largest mean |acceleration| should be
     x (tibial) in standing and z (ML) in side-lying; counts are printed.
"""

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

_HERE        = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = _HERE.parent
GOTOV_ROOT   = Path(os.environ.get("GOTOV_DATA_ROOT", PROJECT_ROOT / "data" / "gotov"))
OUT_DIR      = _HERE / "gotov_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MOBIG_T2     = _HERE / "technical_validation" / "table2_posture_axis_stats.csv"

G        = 9.81
FS_OUT   = 52.0                # Hz — Mobi-G sampling rate
UP, DOWN = 78, 125             # GENEActiv in GOTOV: 83.333 Hz × 78/125 = 52.0 Hz
TRIM_S   = 3.0
PAD_S    = 1.0                 # extra raw margin resampled then dropped (filter edge transients)
MIN_S    = 10.0                # skip blocks shorter than this after trimming
SAT_ABS  = 8.0                 # g — sensor range
SAT_NEAR = 0.95 * SAT_ABS

AXIS_MAP = {"x": "y", "y": "x", "z": "z"}   # Mobi-G axis ← GENEActiv column
AXES = ["x", "y", "z"]

# GOTOV label → Table II label (a block may count toward several rows).
#   sitting  = office chair only (closest to a bedside chair; sofa excluded)
#   lying    = reported as two separate rows (pooling would mix two opposite
#              gravity patterns): side-lying left/right, and couch with feet up
#              (shank horizontal, resembles supine)
#   walking  = slow pace only (closest to hospitalized gait; outdoors in GOTOV)
LABEL_MAP = {
    "standing":       ["standing"],
    "sittingChair":   ["sitting"],
    "lyingDownLeft":  ["lying_side"],
    "lyingDownRight": ["lying_side"],
    "sittingCouch":   ["lying_couch"],
    "walkingSlow":    ["walking"],
}
ROW_ORDER   = ["lying", "lying_side", "lying_couch", "sitting", "standing", "walking"]  # Table III row order
GOTOV_ROWS  = [r for r in ROW_ORDER if r != "lying"]


def _axis_stats(subset: pd.DataFrame, label: str) -> dict:
    # Verbatim from technical_validation.py — keep in sync.
    # Per-recording stats, then mean across recordings. All stats on |axis| —
    # orientation-independent, since the anklet can be worn in two orientations
    # that flip axis signs. P1/P99 instead of min/max — robust to isolated
    # spikes and clipped samples.
    row = {"label": label, "n_recordings": subset["recording"].nunique()}
    for ax in AXES:
        grp = subset[ax].abs().groupby(subset["recording"])
        row[f"{ax}_mean"] = grp.mean().mean()
        row[f"{ax}_std"]  = grp.std().mean()
        row[f"{ax}_p01"]  = grp.quantile(0.01).mean()
        row[f"{ax}_p99"]  = grp.quantile(0.99).mean()
    return row


z = zipfile.ZipFile(GOTOV_ROOT / "Activity_Measurements.zip")
files = sorted(n for n in z.namelist()
               if re.search(r"-geneactiv-ankle\.csv$", n, re.I) and not n.startswith("__MACOSX"))
print(f"{len(files)} ankle files")

parts, sat_rows, rec_rows = [], [], []
for f in files:
    pid = Path(f).name.split("-")[0]
    df = pd.read_csv(z.open(f), usecols=["time", "x", "y", "z", "labels"], dtype={"labels": str})
    lab = df["labels"].fillna("NA")
    bid = (lab != lab.shift()).cumsum()
    for k, ((_, gl), blk) in enumerate(df.groupby([bid, lab], sort=False)):
        if gl not in LABEL_MAP:
            continue
        t = (blk["time"].to_numpy() - blk["time"].iloc[0]) / 1000
        dur = t[-1]
        keep = (t >= TRIM_S - PAD_S) & (t <= dur - TRIM_S + PAD_S)
        if dur - 2 * TRIM_S < MIN_S:
            print(f"  skip {pid} {gl}: {dur:.0f} s")
            continue
        raw = blk.loc[keep, ["x", "y", "z"]]
        core = raw[(t[keep] >= TRIM_S) & (t[keep] <= dur - TRIM_S)]
        rec = f"{pid}_{gl}_{k}"
        for ax in AXES:
            v = core[AXIS_MAP[ax]].abs().to_numpy()
            sat_rows.append(dict(participant=pid, recording=rec, gotov_label=gl, axis=ax,
                                 n_rows=len(v), max_abs_g=round(float(v.max()), 3),
                                 n_clipped=int((v >= SAT_ABS).sum()),
                                 n_near_limit=int(((v >= SAT_NEAR) & (v < SAT_ABS)).sum())))
        rs = {ax: resample_poly(raw[AXIS_MAP[ax]].to_numpy() * G, UP, DOWN) for ax in AXES}
        n_pad = int(round(PAD_S * FS_OUT))
        out = pd.DataFrame({ax: rs[ax][n_pad:len(rs[ax]) - n_pad] for ax in AXES})
        out["recording"] = rec
        out["participant"] = pid
        for row_label in LABEL_MAP[gl]:
            parts.append(out.assign(label=row_label))
        rec_rows.append(dict(participant=pid, recording=rec, gotov_label=gl,
                             duration_s=round(len(out) / FS_OUT, 1),
                             norm_mean=float(np.sqrt((out[AXES] ** 2).sum(axis=1)).mean())))

flat = pd.concat(parts, ignore_index=True)
recs = pd.DataFrame(rec_rows)
sat  = pd.DataFrame(sat_rows)

# Axis-mapping check: axis with the largest mean |acceleration| per participant
axis_checks = []
for lbl, expected in [("standing", "x"), ("lying_side", "z")]:
    dominant = (flat[flat["label"] == lbl].groupby("participant")[AXES]
                .apply(lambda d: d.abs().mean()).idxmax(axis=1))
    axis_checks.append(f"{lbl}: gravity on {expected} for "
                       f"{int((dominant == expected).sum())}/{len(dominant)} participants")

# Table II on GOTOV
gotov = pd.DataFrame([_axis_stats(flat[flat["label"] == lbl], lbl) for lbl in GOTOV_ROWS])
gotov["n_participants"] = [flat.loc[flat["label"] == lbl, "participant"].nunique() for lbl in GOTOV_ROWS]
gotov.insert(0, "dataset", "GOTOV")
gotov.to_csv(OUT_DIR / "gotov_posture_axis_stats.csv", index=False, float_format="%.2f")

# Interleave with Mobi-G per label
mobig = pd.read_csv(MOBIG_T2)
mobig = mobig[mobig["label"].isin(ROW_ORDER)].assign(dataset="Mobi-G")
table3 = pd.concat([mobig, gotov], ignore_index=True)
table3["_o"] = table3["label"].map({l: i for i, l in enumerate(ROW_ORDER)})
table3["_d"] = (table3["dataset"] == "GOTOV").astype(int)
table3 = table3.sort_values(["_o", "_d"]).drop(columns=["_o", "_d"])
cols = ["label", "dataset", "n_recordings", "n_participants"] + \
       [f"{a}_{s}" for a in AXES for s in ["mean", "std", "p01", "p99"]]
table3 = table3[cols]
table3["n_participants"] = table3["n_participants"].astype("Int64")
table3.to_csv(OUT_DIR / "table3_mobig_vs_gotov.csv", index=False, float_format="%.2f")

# Context: per-block mean norm (calibration analogue), saturation, demographics
recs.to_csv(OUT_DIR / "gotov_recordings.csv", index=False, float_format="%.4f")
sat.to_csv(OUT_DIR / "gotov_saturation_check.csv", index=False)

summ = pd.read_csv(GOTOV_ROOT / "GOTOV_DataSummary.csv")
inc = summ[summ["ID"].isin(flat["participant"].unique())]
walk = summ[summ["ID"].isin(flat.loc[flat["label"] == "walking", "participant"].unique())]

pd.set_option("display.width", 220)
print("\nMobi-G vs GOTOV (m/s²):")
print(table3.to_string(index=False, float_format="%.2f"))
print("\nPer-recording mean norm (m/s²) by GOTOV label:")
print(recs.groupby("gotov_label")["norm_mean"].describe()[["count", "mean", "min", "max"]].round(2))
print("\nAxis-mapping check: " + "  |  ".join(axis_checks))
print("Saturation (included blocks): clipped rows =",
      int((sat["n_clipped"] > 0).sum()), "/", len(sat), "| near-limit rows =",
      int((sat["n_near_limit"] > 0).sum()))
for name, d in [("included", inc), ("walking subset", walk)]:
    print(f"GOTOV {name}: n={len(d)}  age {d.age.mean():.1f} ± {d.age.std():.1f} "
          f"[{d.age.min()}–{d.age.max()}]  sex {d.gender.map({'man': 'M', 'vrouw': 'F'}).value_counts().to_dict()}")
print(f"\n→ {OUT_DIR}")
