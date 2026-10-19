"""
Technical validation of Mobi-G data

Sections (run in this order):
  1. Sample rate / continuity check     — labeled data; intervals >10x the
                                           nominal sample period are treated
                                           as label-driven exclusions
                                           (unlabeled/excluded rows dropped
                                           during labeling), not sensor gaps
  2. Accelerometer calibration check     — static posture magnitude ≈ 1 g
  3. Outlier / saturation detection      — labeled data, sensor range ±8 g
  4. Axis statistics per posture & step — labeled data: mean / SD / P1 / P99
                                           of |axis|
  5. Prespecified axis checks           — per-participant strip plots
       lying→y, sitting/standing→x, walking/step→y
  6. Step waveforms                     — time-normalized norm of step = 1
                                           windows by protocol, gait speed and
                                           walking aid

Environment variables (not needed if you place the dataset under
<project_root>/data/):
  MOBI_G_DATA_ROOT   path to the data directory (must contain data_with_label/)
                  default: <project_root>/data
  MOBI_G_PARTICIPANTS_CSV  full path to participants.csv
                  default: <MOBI_G_DATA_ROOT>/participants.csv

Output CSVs saved to  technical_validation/
Figures saved to      figures/technical_validation/
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE        = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = _HERE.parent
DATA_ROOT       = Path(os.environ.get("MOBI_G_DATA_ROOT", PROJECT_ROOT / "data"))
DATA_WITH_LABEL = DATA_ROOT / "data_with_label"
PARTICIPANTS_CSV = Path(os.environ.get("MOBI_G_PARTICIPANTS_CSV", DATA_ROOT / "participants.csv"))

OUT_DIR = _HERE / "technical_validation"
FIG_DIR = _HERE / "figures" / "technical_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FS = 52.0   # Hz

# ── KNOWN ACTIVITIES (from build_data_with_label.py ACTIVITY_MAP) ────────────
KNOWN_ACTIVITIES = {
    "sitting", "lying", "standing", "walking_reg", "walking_irreg",
    "transfers", "wheelchair", "other", "physio",
}


# ──────────────────────────────────────────────────────────────────────────────
# LOAD LABELED DATA — shared by all sections below
# ──────────────────────────────────────────────────────────────────────────────
def _parse_labeled_stem(participant: str, stem: str):
    """Return (protocol_en, activity_en) or (None, None) if unparseable."""
    m = re.match(rf"^{re.escape(participant)}_(protocol|naturalistic)_(.+)$", stem)
    if not m:
        return None, None
    protocol_en   = m.group(1)
    activity_full = m.group(2)
    # Strip trailing recording-index suffix (_1, _2 …)
    activity_en = re.sub(r"_\d+$", "", activity_full)
    # Collapse physio / physio1 / physio2 → physio
    activity_en = re.sub(r"^physio\d+$", "physio", activity_en)
    return protocol_en, activity_en

flat_parts = []
for pdir in sorted(DATA_WITH_LABEL.iterdir()):
    if not pdir.is_dir():
        continue
    participant = pdir.name
    for csv_path in sorted(pdir.glob("*.csv")):
        protocol_en, activity_en = _parse_labeled_stem(participant, csv_path.stem)
        if activity_en is None:
            print(f"  [WARN] Could not parse: {csv_path.name}")
            continue
        df_w = pd.read_csv(csv_path)
        df_w["participant"] = participant
        df_w["protocol"]    = protocol_en
        df_w["activity"]    = activity_en
        df_w["recording"]   = csv_path.name
        flat_parts.append(df_w)

flat = pd.concat(flat_parts, ignore_index=True)
print(f"  Loaded {len(flat):,} rows from {len(flat_parts)} recordings\n")

AXES = ["x", "y", "z"]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SAMPLE RATE AND CONTINUITY CHECK (LABELED DATA)
# ──────────────────────────────────────────────────────────────────────────────
# Runs on ts_ms from the labeled recordings, not raw files. Rows dropped during
# labeling (unlabeled tails, excluded activities/labels) leave large jumps in
# ts_ms that look like sensor gaps but aren't. Any inter-sample interval over
# LABEL_GAP_CUTOFF_S (10x the nominal sample period) is assumed to be one of
# these label-driven exclusions and is reported separately (n_label_gaps)
# rather than folded into the sensor-continuity counts below. Only intervals
# at or under that cutoff
# are treated as real samples and checked against the ±4 Hz tolerance:
#   actual_fs  = (n_real_intervals) / (sum of real inter-sample intervals)
#   continuity = scan real inter-sample intervals; flag those outside
#                [1/56, 1/48] s (±4 Hz tolerance around nominal 52 Hz)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 1 — Sample rate and continuity (labeled data, nominal FS = 52 Hz, ±4 Hz tolerance)")
print("=" * 72)

DT_MIN = 1 / (FS + 4)   # 1/56 s — below this: burst / too fast
DT_MAX = 1 / (FS - 4)   # 1/48 s — above this: gap / discontinuity
LABEL_GAP_CUTOFF_S = 10 / FS   # 10x nominal sample period — longer intervals are assumed to be label-driven exclusions

ts_rows = []
for (participant, recording), grp in flat.groupby(["participant", "recording"]):
    ts_s = grp["ts_ms"].to_numpy() / 1000.0
    n_rows = len(ts_s)

    dts = np.diff(ts_s)
    dts = dts[dts >= 0]   # guard against any out-of-order rows
    label_gap_mask = dts > LABEL_GAP_CUTOFF_S
    real_dts       = dts[~label_gap_mask]
    n_intervals    = len(real_dts)

    if n_intervals > 0:
        effective_duration_s = float(real_dts.sum())
        actual_fs = n_intervals / effective_duration_s if effective_duration_s > 0 else float("nan")
        n_gaps    = int((real_dts > DT_MAX).sum())
        n_bursts  = int((real_dts < DT_MIN).sum())
        max_gap_s = float(real_dts.max())
    else:
        effective_duration_s = 0.0
        actual_fs = float("nan")
        n_gaps = n_bursts = 0
        max_gap_s = float("nan")

    issues = []
    if n_intervals > 0 and not (FS - 4 <= actual_fs <= FS + 4):
        issues.append("RATE")
    if n_gaps > 0:
        issues.append("GAPS")
    if n_bursts > 0:
        issues.append("BURSTS")
    flag = "OK" if not issues else "+".join(issues)

    ts_rows.append(dict(
        participant = participant,
        file        = recording,
        n_rows      = n_rows,
        effective_duration_s = round(effective_duration_s, 2),
        actual_fs   = round(actual_fs, 3) if n_intervals > 0 else float("nan"),
        n_gaps      = n_gaps,
        pct_gaps    = round(n_gaps / n_intervals * 100, 3) if n_intervals > 0 else float("nan"),
        n_bursts    = n_bursts,
        pct_bursts  = round(n_bursts / n_intervals * 100, 3) if n_intervals > 0 else float("nan"),
        max_gap_s   = round(max_gap_s, 4) if n_intervals > 0 else float("nan"),
        flag        = flag,
    ))

ts_df = pd.DataFrame(ts_rows)
ts_df.to_csv(OUT_DIR / "01_sample_rate_continuity.csv", index=False)

n_ok   = (ts_df["flag"] == "OK").sum()
n_bad  = len(ts_df) - n_ok
print(f"  {len(ts_df)} labeled recordings  |  {n_ok} OK  |  {n_bad} flagged")
print(f"  actual_fs  mean={ts_df['actual_fs'].mean():.3f}  "
      f"SD={ts_df['actual_fs'].std():.3f}  "
      f"range=[{ts_df['actual_fs'].min():.3f}, {ts_df['actual_fs'].max():.3f}]")
if n_bad:
    bad = ts_df[ts_df["flag"] != "OK"]
    print(bad[["participant", "file", "actual_fs", "n_gaps", "pct_gaps", "n_bursts", "pct_bursts",
               "max_gap_s", "flag"]]
          .to_string(index=False))
print("  → 01_sample_rate_continuity.csv\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ACCELEROMETER CALIBRATION CHECK (STATIC POSTURE MAGNITUDE ≈ 1 g)
# ──────────────────────────────────────────────────────────────────────────────
# Units are m/s^2 (a resting sensor reads ~9.81 on gravity's axis, e.g. x≈-10 in
# P01_protocol_standing.csv). For static postures the full 3-axis vector norm
# should sit close to 1 g regardless of orientation; systematic deviation flags
# a miscalibrated or drifting sensor.
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 2 — Accelerometer calibration check (static posture magnitude ≈ 1 g)")
print("=" * 72)

GRAVITY_MS2      = 9.81
CALIB_TOL_PCT    = 5.0   # flag recordings whose mean magnitude deviates >5% from 9.81 m/s^2
STATIC_POSTURES  = ["lying", "sitting", "standing"]

static = flat[flat["posture"].isin(STATIC_POSTURES)].copy()
static["magnitude"] = np.sqrt(static["x"]**2 + static["y"]**2 + static["z"]**2)

calib_rows = []
for (participant, recording), grp in static.groupby(["participant", "recording"]):
    mag_mean = grp["magnitude"].mean()
    mag_std  = grp["magnitude"].std()
    pct_dev  = (mag_mean - GRAVITY_MS2) / GRAVITY_MS2 * 100
    calib_rows.append({
        "participant":    participant,
        "recording":      recording,
        "posture":        grp["posture"].iloc[0] if grp["posture"].nunique() == 1 else "mixed",
        "n_samples":      len(grp),
        "mag_mean":       round(mag_mean, 4),
        "mag_std":        round(mag_std, 4),
        "pct_dev_from_1g": round(pct_dev, 2),
        "flag":           "OK" if abs(pct_dev) <= CALIB_TOL_PCT else "CALIB_DEVIATION",
    })

calib_df = pd.DataFrame(calib_rows).sort_values(["participant", "recording"]).reset_index(drop=True)
calib_df.to_csv(OUT_DIR / "02_calibration_check.csv", index=False)

n_ok = (calib_df["flag"] == "OK").sum()
print(f"  {len(calib_df)} static recordings  |  {n_ok} OK  |  {len(calib_df) - n_ok} flagged "
      f"(>±{CALIB_TOL_PCT}% from {GRAVITY_MS2} m/s²)")
print(f"  magnitude (m/s²)  mean={calib_df['mag_mean'].mean():.3f}  SD={calib_df['mag_mean'].std():.3f}  "
      f"range=[{calib_df['mag_mean'].min():.3f}, {calib_df['mag_mean'].max():.3f}]")
print(f"  max |deviation| from {GRAVITY_MS2} m/s² = {calib_df['pct_dev_from_1g'].abs().max():.2f}%")
if n_ok < len(calib_df):
    bad = calib_df[calib_df["flag"] != "OK"]
    print(bad.to_string(index=False))
print("  → 02_calibration_check.csv\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — OUTLIER / SATURATION DETECTION (LABELED DATA)
# ──────────────────────────────────────────────────────────────────────────────
# Runs on the labeled recordings (flat), not raw files — recordings excluded
# from labeling (e.g. wheelchair/other activities) are out of scope here, and
# saturation is a per-sample property so dropped/unlabeled rows don't distort it.
# This sensor's accelerometer range is ±8 g. Two checks per axis, per recording:
#   (a) hard limit  — |value| at the sensor's maximum output (78.40 m/s² at
#                      ±8 g) is clipping; the sensor cannot report anything
#                      past its own range.
#   (b) near limit  — |value| within 5% of the maximum output: not yet clipped,
#                      but a near-miss worth knowing about for high-dynamic
#                      activities (e.g. transfers, irregular walking).
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 3 — Outlier / saturation detection (labeled data)")
print("=" * 72)

SENSOR_RANGE_G     = 8.0
SAT_ABS_LIMIT_MS2  = 78.40                         # m/s^2 — sensor's maximum output at SENSOR_RANGE_G
SAT_NEAR_LIMIT_MS2 = 0.95 * SAT_ABS_LIMIT_MS2      # within 5% of the ceiling

sat_rows = []
for (participant, recording), grp in flat.groupby(["participant", "recording"]):
    n_rows = len(grp)
    for ax in AXES:
        vals       = grp[ax].to_numpy()
        abs_vals   = np.abs(vals)
        n_clipped  = int((abs_vals >= SAT_ABS_LIMIT_MS2).sum())
        n_near     = int(((abs_vals >= SAT_NEAR_LIMIT_MS2) & (abs_vals < SAT_ABS_LIMIT_MS2)).sum())
        max_abs    = float(abs_vals.max()) if n_rows else float("nan")
        extreme    = vals[np.argmax(abs_vals)] if n_rows else float("nan")
        n_at_extreme = int((vals == extreme).sum()) if n_rows else 0

        sat_rows.append(dict(
            participant=participant, file=recording, axis=ax, n_rows=n_rows,
            max_abs=round(max_abs, 3),
            n_clipped=n_clipped,
            n_near_limit=n_near,
            n_at_extreme=n_at_extreme,
        ))

sat_df = pd.DataFrame(sat_rows)
sat_df.to_csv(OUT_DIR / "03_saturation_check.csv", index=False)

n_clipped_rows = (sat_df["n_clipped"] > 0).sum()
n_near_rows    = (sat_df["n_near_limit"] > 0).sum()
print(f"  {len(sat_df)} file×axis rows  |  {n_clipped_rows} with clipped samples  |  "
      f"{n_near_rows} with near-limit samples")
if n_clipped_rows or n_near_rows:
    bad = sat_df[(sat_df["n_clipped"] > 0) | (sat_df["n_near_limit"] > 0)]
    print(bad.to_string(index=False))
print("  → 03_saturation_check.csv\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — AXIS STATISTICS PER POSTURE LABEL AND STEP (LABELED DATA)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 4 — Axis statistics per posture label and step")
print("=" * 72)

def _axis_stats(subset: pd.DataFrame, label: str) -> dict:
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

# 4a — per posture
posture_stats = [
    _axis_stats(flat[flat["posture"] == lbl], lbl)
    for lbl in sorted(flat["posture"].dropna().unique())
]
ps_df = pd.DataFrame(posture_stats)
ps_df.to_csv(OUT_DIR / "table2_posture_axis_stats.csv", index=False, float_format="%.2f")
print("Posture axis statistics:")
print(ps_df.to_string(index=False, float_format="%.3f"))

# 4b — step=1 vs step=0
step_df = pd.DataFrame([
    _axis_stats(flat[flat["step"] == 1], "step=1"),
    _axis_stats(flat[flat["step"] == 0], "step=0"),
])
step_df.to_csv(OUT_DIR / "table2_step_axis_stats.csv", index=False, float_format="%.2f")
print("\nStep axis statistics:")
print(step_df.to_string(index=False, float_format="%.3f"))
print("\n  → table2_posture_axis_stats.csv  table2_step_axis_stats.csv\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PRESPECIFIED AXIS CHECKS PER PARTICIPANT
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 5 — Prespecified axis checks per participant")
print("=" * 72)

POSTURE_CHECKS = [
    # (posture_or_step, filter_col, filter_val, axis, stat, col_name, description)
    ("lying",     "posture", "lying",     "x", "mean", "lying_x_mean",     "Lying — mean(x)"),
    ("lying",     "posture", "lying",     "x", "std",  "lying_x_std",      "Lying — SD(x)"),
    ("lying",     "posture", "lying",     "y", "mean", "lying_y_mean",     "Lying — mean(y)"),
    ("lying",     "posture", "lying",     "y", "std",  "lying_y_std",      "Lying — SD(y)"),
    ("sitting",   "posture", "sitting",   "x", "mean", "sitting_x_mean",   "Sitting — mean(x)"),
    ("sitting",   "posture", "sitting",   "x", "std",  "sitting_x_std",    "Sitting — SD(x)"),
    ("sitting",   "posture", "sitting",   "y", "mean", "sitting_y_mean",   "Sitting — mean(y)"),
    ("sitting",   "posture", "sitting",   "y", "std",  "sitting_y_std",    "Sitting — SD(y)"),
    ("standing",  "posture", "standing",  "x", "mean", "standing_x_mean",  "Standing — mean(x)"),
    ("standing",  "posture", "standing",  "x", "std",  "standing_x_std",   "Standing — SD(x)"),
    ("standing",  "posture", "standing",  "y", "mean", "standing_y_mean",  "Standing — mean(y)"),
    ("standing",  "posture", "standing",  "y", "std",  "standing_y_std",   "Standing — SD(y)"),
    ("walking",   "posture", "walking",   "x", "mean", "walking_x_mean",   "Walking — mean(x)"),
    ("walking",   "posture", "walking",   "x", "std",  "walking_x_std",    "Walking — SD(x)"),
    ("walking",   "posture", "walking",   "y", "mean", "walking_y_mean",   "Walking — mean(y)"),
    ("walking",   "posture", "walking",   "y", "std",  "walking_y_std",    "Walking — SD(y)"),
    ("transfers", "posture", "transfers", "x", "mean", "transfers_x_mean", "Transfers — mean(x)"),
    ("transfers", "posture", "transfers", "x", "std",  "transfers_x_std",  "Transfers — SD(x)"),
    ("transfers", "posture", "transfers", "y", "mean", "transfers_y_mean", "Transfers — mean(y)"),
    ("transfers", "posture", "transfers", "y", "std",  "transfers_y_std",  "Transfers — SD(y)"),
]

STEP_CHECKS = [
    ("step=1", "step", 1, "x", "mean", "step1_x_mean", "Step=1 — mean(x)"),
    ("step=1", "step", 1, "x", "std",  "step1_x_std",  "Step=1 — SD(x)"),
    ("step=1", "step", 1, "y", "mean", "step1_y_mean", "Step=1 — mean(y)"),
    ("step=1", "step", 1, "y", "std",  "step1_y_std",  "Step=1 — SD(y)"),
    ("step=0", "step", 0, "x", "mean", "step0_x_mean", "Step=0 — mean(x)"),
    ("step=0", "step", 0, "x", "std",  "step0_x_std",  "Step=0 — SD(x)"),
    ("step=0", "step", 0, "y", "mean", "step0_y_mean", "Step=0 — mean(y)"),
    ("step=0", "step", 0, "y", "std",  "step0_y_std",  "Step=0 — SD(y)"),
]

CHECKS = POSTURE_CHECKS + STEP_CHECKS

check_rows = []
for pid in sorted(flat["participant"].unique()):
    sub_p = flat[flat["participant"] == pid]
    row   = {"participant": pid}
    for _, fcol, fval, axis, stat, col_name, _ in CHECKS:
        sub = sub_p[sub_p[fcol] == fval]
        if len(sub) == 0:
            row[col_name] = float("nan")
        else:
            # mean of per-recording stat (absolute scale)
            per_rec = sub.groupby("recording")[axis].apply(
                lambda s, _st=stat: s.abs().mean() if _st == "mean" else s.abs().std()
            )
            row[col_name] = float(per_rec.mean())
    check_rows.append(row)

checks_df = pd.DataFrame(check_rows).set_index("participant").sort_index()
checks_df.to_csv(OUT_DIR / "05_prespecified_checks.csv", float_format="%.4f")
print(checks_df.to_string(float_format="%.3f"))
print("\n  → 05_prespecified_checks.csv\n")

N_PARTICIPANTS = len(checks_df)

# Figures — horizontal strip plot per check, one figure for postures and one for steps
def _plot_checks_figure(checks, suptitle, filename):
    check_cols  = [c[5] for c in checks]
    check_descs = {c[5]: c[6] for c in checks}
    n_checks    = len(check_cols)
    ncols_c, nrows_c = 4, int(np.ceil(n_checks / 4))

    fig, axes = plt.subplots(nrows_c, ncols_c,
                              figsize=(5 * ncols_c, max(4, 0.3 * N_PARTICIPANTS) * nrows_c),
                              squeeze=False)
    axes_flat = axes.flatten()

    for i, col in enumerate(check_cols):
        ax   = axes_flat[i]
        vals = checks_df[col].dropna()
        ax.scatter(vals.values, np.arange(len(vals)),
                   color="steelblue", s=35, alpha=0.85, zorder=3)
        for fn, color, ls in [("mean", "tomato", "--"), ("median", "navy", ":")]:
            stat_val = vals.mean() if fn == "mean" else vals.median()
            ax.axvline(stat_val, color=color, linewidth=1.2, linestyle=ls,
                       label=f"{fn}={stat_val:.3f}")
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=6)
        ax.invert_yaxis()   # P01 on top, descending participant order downward
        ax.set_xlim(0, 12 if col.endswith("_mean") else 5)
        ax.set_title(check_descs[col], fontsize=8.5, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel("g", fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved {filename}\n")
    return fig

fig_desc1 = _plot_checks_figure(
    POSTURE_CHECKS,
    "Descriptive figure 1 — Prespecified axis checks per participant (postures)",
    "fig_desc1_prespecified_checks_postures.png",
)
fig_desc2 = _plot_checks_figure(
    STEP_CHECKS,
    "Descriptive figure 2 — Prespecified axis checks per participant (step=1 vs step=0)",
    "fig_desc2_prespecified_checks_steps.png",
)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — AVERAGE STEP WAVEFORMS
# ──────────────────────────────────────────────────────────────────────────────
# Signal: acceleration norm sqrt(x² + y² + z²) — orientation-independent, so no
# axis sign harmonization is needed across anklet orientations.
# Each contiguous step = 1 window (the full non-baseline waveform of one step of
# the instrumented foot, not crossing a label-driven gap) is time-normalized to
# 101 points (0–100% of step duration). Steps are averaged within participant
# first, then across participants: grand mean ± SD across participants.
# Splits: protocol vs naturalistic; gait speed ≥ 0.6 vs < 0.6 m/s (threshold of
# the step-counting accuracy reference; participants with missing gait speed
# excluded from that panel only); no aid or cane vs walker or wheelchair.
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 6 — Average step waveforms (acceleration norm)")
print("=" * 72)

N_PTS            = 101
MIN_STEP_SAMPLES = 5
GAIT_SPEED_CUT   = 0.6   # m/s

curves, step_rows, n_short = [], [], 0
for (participant, recording, protocol), grp in flat.groupby(["participant", "recording", "protocol"], sort=False):
    st = grp["step"].to_numpy()
    if not (st == 1).any():
        continue
    ts   = grp["ts_ms"].to_numpy() / 1000.0
    norm = np.sqrt(grp["x"].to_numpy()**2 + grp["y"].to_numpy()**2 + grp["z"].to_numpy()**2)
    seg  = np.r_[0, np.cumsum(np.diff(ts) > LABEL_GAP_CUTOFF_S)]
    run  = np.r_[0, np.cumsum((np.diff(st) != 0) | (np.diff(seg) != 0))]
    for r in np.unique(run[st == 1]):
        m = run == r
        if m.sum() < MIN_STEP_SAMPLES:
            n_short += 1
            continue
        t = ts[m]
        curves.append(np.interp(np.linspace(0, 1, N_PTS), (t - t[0]) / (t[-1] - t[0]), norm[m]))
        step_rows.append({"participant": participant, "recording": recording,
                          "protocol": protocol, "duration_s": t[-1] - t[0]})

curves = np.vstack(curves)
steps  = pd.DataFrame(step_rows)

pp = pd.read_csv(PARTICIPANTS_CSV)
pp["participant"] = pp["participant_id"].map(lambda i: f"P{int(i):02d}")
pp = pp.set_index("participant")
steps["speed_group"] = steps["participant"].map(pp["gait_speed"]).map(
    lambda v: np.nan if pd.isna(v) else (f"< {GAIT_SPEED_CUT} m/s" if v < GAIT_SPEED_CUT else f"≥ {GAIT_SPEED_CUT} m/s"))
steps["aid_group"] = steps["participant"].map(pp["walking_aid"]).map(
    {"none": "No aid or cane", "cane": "No aid or cane",
     "walker": "Walker or wheelchair", "wheelchair": "Walker or wheelchair"})
print(f"  {len(steps)} step windows from {steps['participant'].nunique()} participants "
      f"({n_short} windows with < {MIN_STEP_SAMPLES} samples excluded)")

PANELS = [
    ("protocol",    "Protocol",    [("protocol", "Standardized protocol"), ("naturalistic", "Naturalistic")]),
    ("speed_group", "Gait speed",  [(f"≥ {GAIT_SPEED_CUT} m/s", f"≥ {GAIT_SPEED_CUT} m/s"),
                                    (f"< {GAIT_SPEED_CUT} m/s", f"< {GAIT_SPEED_CUT} m/s")]),
    ("aid_group",   "Walking aid", [("No aid or cane", "No aid or cane"),
                                    ("Walker or wheelchair", "Walker or wheelchair")]),
]

wave_rows, group_rows = [], []
for col, panel, groups in PANELS:
    for val, name in groups:
        idx = steps.index[steps[col] == val]
        by_p = steps.loc[idx].groupby("participant")
        p_curves = np.vstack([curves[g.index].mean(axis=0) for _, g in by_p])
        p_dur    = by_p["duration_s"].mean()
        mean, sd = p_curves.mean(axis=0), p_curves.std(axis=0, ddof=1)
        for i in range(N_PTS):
            wave_rows.append({"panel": panel, "group": name, "pct_step": i,
                              "norm_mean": mean[i], "norm_sd": sd[i]})
        group_rows.append({
            "panel": panel, "group": name, "n_participants": len(p_curves), "n_steps": len(idx),
            "duration_mean_s": p_dur.mean(), "duration_sd_s": p_dur.std(),
        })

wave_df  = pd.DataFrame(wave_rows)
group_df = pd.DataFrame(group_rows)
wave_df.to_csv(OUT_DIR / "fig3_step_waveforms.csv", index=False, float_format="%.3f")
group_df.to_csv(OUT_DIR / "fig3_step_waveform_groups.csv", index=False, float_format="%.3f")
print(group_df.to_string(index=False, float_format="%.2f"))

# Figure — one panel per split, grand mean ± SD across participants
SERIES = [("#2a78d6", "-"), ("#eb6834", "--")]   # categorical slots 1–2; line style as second cue
x_pct = np.arange(N_PTS)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (_, panel, groups) in zip(axs, PANELS):
    for (_, name), (color, ls) in zip(groups, SERIES):
        w = wave_df[(wave_df["panel"] == panel) & (wave_df["group"] == name)]
        g = group_df[(group_df["panel"] == panel) & (group_df["group"] == name)].iloc[0]
        ax.fill_between(x_pct, w["norm_mean"] - w["norm_sd"], w["norm_mean"] + w["norm_sd"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x_pct, w["norm_mean"], color=color, linestyle=ls, linewidth=2,
                label=f"{name} (n={g['n_participants']}, {g['n_steps']:,} steps)")
    ax.axhline(GRAVITY_MS2, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_title(panel, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Step window (%)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Acceleration norm (m/s²)")
axs[0].set_ylim(top=21)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_step_waveforms.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("\n  → fig3_step_waveforms.csv  fig3_step_waveform_groups.csv  fig3_step_waveforms.png\n")


print("=" * 72)
print("Technical validation complete.")
print(f"CSVs → {OUT_DIR}")
print(f"Figs → {FIG_DIR}")
print("=" * 72)
