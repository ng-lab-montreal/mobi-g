"""
Descriptive statistics for Mobi-G data.

Sections:
  1. Recording duration                 — minutes of posture and step count data
  2. Descriptive tables
       Table 1 — participant characteristics (mean±SD, n%)
       Table 2 — recording counts per activity × stratification variable
       Table 3 — per-participant activity presence + step counts by activity

Runs entirely on the labeled dataset (data/data_with_label/).

Environment variables (not needed if you place the dataset under
<project_root>/data/):
  MOBI_G_DATA_ROOT          path to the data directory (must contain data_with_label/)
                         default: <project_root>/data
  MOBI_G_PARTICIPANTS_CSV   full path to participants.csv
                         default: <project_root>/data/participants.csv

Output CSVs saved to  descriptive_statistics/
"""

import os
import re
from pathlib import Path

import pandas as pd

_HERE        = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = _HERE.parent
DATA_ROOT       = Path(os.environ.get("MOBI_G_DATA_ROOT", PROJECT_ROOT / "data"))
DATA_WITH_LABEL = DATA_ROOT / "data_with_label"
PARTICIPANTS_CSV = Path(os.environ.get(
    "MOBI_G_PARTICIPANTS_CSV",
    PROJECT_ROOT / "data" / "participants.csv",
))

OUT_DIR = _HERE.parent / "descriptive_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 52.0   # Hz

STEP_ACTIVITIES    = ["walking_reg", "walking_irreg", "physio"]
POSTURE_ACTIVITIES = {"sitting", "lying", "standing", "walking_reg", "walking_irreg", "transfers", "physio"}

# ── LOAD PARTICIPANTS ─────────────────────────────────────────────────────────
pdf = pd.read_csv(PARTICIPANTS_CSV).set_index("participant_id")
pdf.index = pdf.index.map(lambda i: f"P{i:02d}")
N_PARTICIPANTS = len(pdf)

# age_group is a string range (e.g. '75-79'); convert to numeric midpoint for comparisons
def _age_midpoint(s: str) -> float:
    try:
        lo, hi = str(s).split("-")
        return (int(lo) + int(hi)) / 2
    except Exception:
        return float("nan")

pdf["age_midpoint"] = pdf["age_group"].map(_age_midpoint)

medians = {
    "age_midpoint":  float(pdf["age_midpoint"].median()),
    "gait_speed":    float(pdf["gait_speed"].median()),
    "sppb":          float(pdf["sppb"].median()),
    "function_oars": float(pdf["function_oars"].median()),
    "cfs":           float(pdf["cfs"].median()),
}
print(f"N = {N_PARTICIPANTS} participants")
print(f"Medians: {medians}\n")

# ── BINARIZE HELPERS ──────────────────────────────────────────────────────────
BINARIZE_MAP = {
    "gait_pattern": {
        "normal":   {"normal_gait"},
        "abnormal": {"asymmetrical_gait", "symmetrical_abnormal_gait", "non_ambulatory"},
    },
    "walking_aid": {
        "none_cane":         {"none", "cane"},
        "walker_wheelchair": {"walker", "wheelchair"},
    },
    "transferring_ability": {
        "independent": {"without_assistance"},
        "assisted":    {"partial_assistance", "complete_assistance"},
    },
    "cognitive_impairment": {
        "none":     {"none"},
        "impaired": {"mild_ncd", "moderate_ncd", "severe_ncd"},
    },
}

def _binarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add *_bin columns in-place for all split variables. Returns df."""
    df = df.copy()
    for col, groups in BINARIZE_MAP.items():
        if col not in df.columns:
            continue
        df[f"{col}_bin"] = df[col].map(
            lambda v, _g=groups: next((lbl for lbl, vals in _g.items() if v in vals), None)
        )
    for col, med in medians.items():
        if col not in df.columns:
            continue
        df[f"{col}_bin"] = df[col].apply(
            lambda v, _m=med: (f"<{_m:.1f}" if v < _m else f"≥{_m:.1f}") if pd.notna(v) else None
        )
    return df

pdf = _binarize_df(pdf)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD LABELED DATA — shared by sections below
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


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — RECORDING DURATION (MINUTES OF POSTURE AND STEP COUNT DATA)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 1 — Recording duration (minutes of posture and step count data)")
print("=" * 72)

dur_rows = []
for pid in sorted(flat["participant"].unique()):
    sub_p    = flat[flat["participant"] == pid]
    posture_min   = len(sub_p[sub_p["activity"].isin(POSTURE_ACTIVITIES)]) / FS / 60
    stepcount_min = len(sub_p[sub_p["activity"].isin(STEP_ACTIVITIES)]) / FS / 60
    step_min      = sub_p["step"].eq(1).sum() / FS / 60
    dur_rows.append({
        "participant":    pid,
        "posture_min":    round(posture_min, 2),
        "stepcount_min":  round(stepcount_min, 2),
        "step_min":       round(step_min, 2),
    })

dur_df = pd.DataFrame(dur_rows).set_index("participant").sort_index()
dur_df.to_csv(OUT_DIR / "01_recording_duration_minutes.csv")
print(dur_df.to_string())

for col, label in [("posture_min", "Posture"), ("stepcount_min", "Step count"), ("step_min", "Step=1 (actual stepping)")]:
    s = dur_df[col]
    print(f"\n{label}: total={s.sum():.1f} min  |  "
          f"mean={s.mean():.1f}  SD={s.std():.1f}  "
          f"range=[{s.min():.1f}, {s.max():.1f}]")
print("\n  → 01_recording_duration_minutes.csv\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DESCRIPTIVE TABLES
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("SECTION 2 — Descriptive tables")
print("=" * 72)

# ── TABLE 1: participant characteristics ─────────────────────────────────────
CONTINUOUS_VARS  = ["age_midpoint", "gait_speed", "sppb", "function_oars", "cfs"]
CATEGORICAL_VARS = ["age_group", "sex", "gait_pattern", "walking_aid",
                    "transferring_ability", "cognitive_impairment", "orthopedic_history"]

t1_rows = []
for col in CONTINUOUS_VARS:
    s = pdf[col].dropna()
    t1_rows.append({
        "variable":  col.replace("_", " "),
        "category":  "",
        "n":         len(s),
        "mean_sd":   f"{s.mean():.1f} ± {s.std():.1f}",
        "median_iqr": f"{s.median():.1f}  [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]",
        "range":     f"{s.min():.1f}–{s.max():.1f}",
    })
for col in CATEGORICAL_VARS:
    vc = pdf[col].value_counts(dropna=False)
    for val, cnt in vc.items():
        t1_rows.append({
            "variable": col.replace("_", " "),
            "category": str(val),
            "n":        cnt,
            "mean_sd":  f"{cnt} ({cnt/N_PARTICIPANTS*100:.1f}%)",
            "median_iqr": "", "range": "",
        })

t1_df = pd.DataFrame(t1_rows)
t1_df.to_csv(OUT_DIR / "table1_characteristics.csv", index=False)
print("TABLE 1 — Participant characteristics")
print(t1_df.to_string(index=False))
print("\n  → table1_characteristics.csv\n")

# ── Build recording inventory ─────────────────────────────────────────────────
rec_rows = []
for pdir in sorted(DATA_WITH_LABEL.iterdir()):
    if not pdir.is_dir():
        continue
    participant = pdir.name
    for csv_path in sorted(pdir.glob("*.csv")):
        protocol_en, activity_en = _parse_labeled_stem(participant, csv_path.stem)
        if activity_en is None:
            continue
        rec_rows.append({"participant": participant,
                         "protocol":    protocol_en,
                         "activity":    activity_en})

rec_df = pd.DataFrame(rec_rows)
rec_df = rec_df.join(pdf, on="participant", how="left")
rec_df = _binarize_df(rec_df)
activities = sorted(rec_df["activity"].unique())

# ── TABLE 2: recordings per activity × stratification ────────────────────────
def _rec_counts(subset: pd.DataFrame) -> dict:
    vc = subset["activity"].value_counts()
    return {act: int(vc.get(act, 0)) for act in activities}

STRAT_VARS = [
    ("protocol",             "protocol"),
    ("age",                  "age_midpoint_bin"),
    ("sex",                  "sex"),
    ("gait_pattern",         "gait_pattern_bin"),
    ("gait_speed",           "gait_speed_bin"),
    ("sppb",                 "sppb_bin"),
    ("walking_aid",          "walking_aid_bin"),
    ("transferring_ability", "transferring_ability_bin"),
    ("cognitive_impairment", "cognitive_impairment_bin"),
    ("orthopedic_history",   "orthopedic_history"),
    ("function_oars",        "function_oars_bin"),
    ("cfs",                  "cfs_bin"),
]

t2_rows = []
overall = {"stratum": "Overall", "group": "—",
           "n_participants": rec_df["participant"].nunique(),
           "n_recordings":   len(rec_df)}
overall.update(_rec_counts(rec_df))
t2_rows.append(overall)

for sv_label, sv_col in STRAT_VARS:
    if sv_col not in rec_df.columns:
        continue
    for grp in sorted(rec_df[sv_col].dropna().unique()):
        sub = rec_df[rec_df[sv_col] == grp]
        row = {"stratum": sv_label, "group": grp,
               "n_participants": sub["participant"].nunique(),
               "n_recordings":   len(sub)}
        row.update(_rec_counts(sub))
        t2_rows.append(row)

t2_df = pd.DataFrame(t2_rows, columns=["stratum", "group", "n_participants", "n_recordings"] + activities)
t2_df.to_csv(OUT_DIR / "table2_recordings_by_activity.csv", index=False)
print("TABLE 2 — Recording counts per activity × stratification")
print(t2_df.to_string(index=False))
print("\n  → table2_recordings_by_activity.csv\n")

# ── TABLE 3: per-participant activity presence + step counts ──────────────────
t3_rows = []
for pid in sorted(flat["participant"].unique()):
    sub_p = flat[flat["participant"] == pid]
    row   = {"participant": pid}
    for act in activities:
        sub_a = sub_p[sub_p["activity"] == act]
        row[f"{act}_n_recs"] = int(sub_a["recording"].nunique())
    for act in STEP_ACTIVITIES:
        sub_a = sub_p[sub_p["activity"] == act]
        s = sub_a["step"].reset_index(drop=True)
        row[f"{act}_steps"] = int(((s == 1) & (s.shift(1, fill_value=0) == 0)).sum())
    t3_rows.append(row)

t3_df = pd.DataFrame(t3_rows).set_index("participant").sort_index()
t3_df.to_csv(OUT_DIR / "table3_per_participant.csv")
print("TABLE 3 — Per-participant recording counts and step totals by activity")
print(t3_df.to_string())
print("\n  → table3_per_participant.csv\n")

print("=" * 72)
print("Descriptive statistics complete.")
print(f"CSVs → {OUT_DIR}")
print("=" * 72)
