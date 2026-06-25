"""
analyze_results.py  -  QUANT 2026  |  Paper figures
===============================================================
Generates the four figures used in the paper:
  P3  F1 macro vs n_qubits  (panel per SBERT model)   -> Fig. 2
  S3  Performance-stability tradeoff: mean(F1) vs std(F1) -> Fig. 3
  T2  Confronto assoluto tra stadi su scala log         -> Fig. 4
  T1  Best epoch medio per tecnica (panel per SBERT)    -> Fig. 5
plus the CSV stats: summary_timing_stability.csv and f1_stats.csv
(per-qubit mean +/- std behind Table 1).

Organizzazione: Amplitude | Angle_X  (panel)
               PCA vs W2K            (linee, saturazione colore)
               MiniLM / mpnet        (linestyle)

Uso:
    python analyze_results.py [--results_dir ./results] [--out_dir ./analysis]
"""

import argparse, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", default="./results")
parser.add_argument("--out_dir",     default="./analysis")
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
OUT_DIR     = Path(args.out_dir)
PLOTS_DIR   = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- PALETTE -----------------------------------------------------------------
import matplotlib.colors as mc

ENC_BASE   = {"amplitude": "#7C3AED", "angle_X": "#EA580C"}
COMPR_SAT  = {"pca": 0.45, "word2ket": 1.0, "w2k-angle": 1.0}
COMPR_DISP = {"pca": "PCA", "word2ket": "W2K", "w2k-angle": "W2K"}
SBERT_ST   = {"MiniLM": dict(linestyle="--", marker="o", ms=5),
              "mpnet":  dict(linestyle="-",  marker="s", ms=5)}
ENC_TITLE  = {"amplitude": "Amplitude Encoding", "angle_X": "Angle_X Encoding"}

def cfg_color(enc, compr):
    r, g, b = mc.to_rgb(ENC_BASE[enc])
    s = COMPR_SAT.get(compr, 0.7)
    return (r + (1-r)*(1-s), g + (1-g)*(1-s), b + (1-b)*(1-s))

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})

# --- LOAD --------------------------------------------------------------------
def infer_compression(row):
    p = str(row.get("preproc", ""))
    cm = str(row.get("compression_method", ""))
    if cm and cm not in ("nan", "None", ""):
        return cm
    if "word2ket" in p: return "word2ket"
    if "w2k-angle" in p: return "w2k-angle"
    return "pca"

csv_files = sorted(RESULTS_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"Nessun CSV in {RESULTS_DIR}")

dfs = []
for fp in csv_files:
    df = pd.read_csv(fp)
    for c in ("compression_method", "w2k_rank"):
        if c not in df.columns: df[c] = None
    df["compression_method"] = df.apply(infer_compression, axis=1)
    df["sbert_short"]  = df["sbert_model"].apply(lambda x: "MiniLM" if "MiniLM" in x else "mpnet")
    df["config_key"]   = df["encoding"] + "+" + df["compression_method"]
    df["is_lossless"]  = df["preproc"].str.contains("zero_pad")
    df["source_file"]  = fp.name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data["gen_gap"]    = data["val_f1_best"] - data["test_f1_macro"]
data["conv_ratio"] = data["best_epoch"]  / data["stopped_at"]

ENCODINGS = ["amplitude", "angle_X"]
COMPRS    = {"amplitude": ["pca", "word2ket"], "angle_X": ["pca", "w2k-angle"]}
MODELS    = sorted(data["sbert_short"].unique())
Q_RANGE   = sorted(data["n_qubits"].unique())


# --- SAVE HELPER -------------------------------------------------------------
def dual_save(draw_fn, basename, h_figsize=(13, 5), v_figsize=(7, 10),
              sharey=True, sharex=False):
    """Salva il plot nel layout orizzontale (_h, 1x2) usato nel paper."""
    fig, axes = plt.subplots(1, 2, figsize=h_figsize,
                             sharey=sharey, sharex=sharex)
    draw_fn(fig, axes)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{basename}_h.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {basename}_h.png")


# ============================================================================
# S3 - Performance-stability tradeoff: mean(F1) vs std(F1)
#      2 panel: MiniLM | mpnet  -  ogni linea = una config, punti = n_qubits
# ============================================================================
print("=" * 70)
print("S3  Performance-stability tradeoff: mean(F1) vs std(F1)")
print("=" * 70)

CFGS_ORDERED = [
    ("amplitude", "pca"),
    ("amplitude", "word2ket"),
    ("angle_X",   "pca"),
    ("angle_X",   "w2k-angle"),
]
CFG_LABEL = {
    ("amplitude", "pca"):      "Amplitude + PCA",
    ("amplitude", "word2ket"): "Amplitude + W2K",
    ("angle_X",   "pca"):      "Angle_X + PCA",
    ("angle_X",   "w2k-angle"):"Angle_X + W2K",
}

def _draw_s3(fig, axes):
    for ax, sbert in zip(axes, MODELS):
        for enc, compr in CFGS_ORDERED:
            sub = data[(data["encoding"] == enc) &
                       (data["compression_method"] == compr) &
                       (data["sbert_short"] == sbert)]
            if sub.empty: continue
            grp  = sub.groupby("n_qubits")["test_f1_macro"]
            mu   = grp.mean().sort_index()
            sd   = grp.std().fillna(0).sort_index()
            c    = cfg_color(enc, compr)
            lbl  = CFG_LABEL[(enc, compr)]
            ax.plot(mu.values, sd.values, color=c, lw=1.5, alpha=0.5)
            ax.scatter(mu.values, sd.values, color=c, s=50 + mu.index * 9,
                       zorder=4, label=lbl)
            for q in [mu.index[0], mu.index[-1]]:
                ax.annotate(str(q), (mu[q], sd[q]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=7.5, color=c)
        ax.set_xlabel("mean(F1 macro)"); ax.set_ylabel("std(F1 macro) tra seed")
        ax.annotate("ottimale: destra+basso ->", xy=(0.98, 0.04),
                    xycoords="axes fraction", ha="right",
                    fontsize=8, color="gray", style="italic")
    # legenda singola condivisa sopra la figura
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Config", fontsize=15, title_fontsize=18,
               ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.01), bbox_transform=fig.transFigure)

dual_save(_draw_s3, "s3_perf_stability_scatter",
          h_figsize=(13, 5.5), v_figsize=(7, 11), sharey=True, sharex=False)


# ============================================================================
# T1 - Best epoch medio per tecnica  (bar chart)
#      2 panel: MiniLM | mpnet
#      4 barre per panel: una per combinazione (Encoding + Reduction)
#      Altezza barra = mean(best_epoch); error bar = std(best_epoch)
# ============================================================================
print("=" * 70)
print("T1  Best epoch medio per tecnica, raggruppato per SBERT model")
print("=" * 70)

be_rows = []
for sbert in MODELS:
    for enc in ENCODINGS:
        for compr in COMPRS[enc]:
            sub = data[(data["encoding"] == enc) &
                       (data["compression_method"] == compr) &
                       (data["sbert_short"] == sbert)]
            if sub.empty: continue
            be_rows.append({
                "sbert": sbert, "enc": enc, "compr": compr,
                "label": f"{ENC_TITLE[enc].split()[0]}\n+ {COMPR_DISP[compr]}",
                "best_epoch_mean": sub["best_epoch"].mean(),
                "best_epoch_std":  sub["best_epoch"].std(),
                "n":               len(sub),
            })
be_df = pd.DataFrame(be_rows)

def _draw_t1(fig, axes):
    for ax, sbert in zip(axes, MODELS):
        sub_df = be_df[be_df["sbert"] == sbert].reset_index(drop=True)
        x = np.arange(len(sub_df))
        colors = [cfg_color(r["enc"], r["compr"]) for _, r in sub_df.iterrows()]
        ax.bar(x, sub_df["best_epoch_mean"],
               color=colors, alpha=0.85,
               yerr=sub_df["best_epoch_std"], capsize=4,
               error_kw={"lw": 1.2})
        for xi, mu, sd in zip(x, sub_df["best_epoch_mean"], sub_df["best_epoch_std"]):
            ax.text(xi, mu + sd + 0.5, f"{mu:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color="black")
        ax.set_xticks(x); ax.set_xticklabels(sub_df["label"], fontsize=8.5)
        ax.set_ylabel("best_epoch medio" if sbert == MODELS[0] else "")

dual_save(_draw_t1, "t1_best_epoch_by_technique")

# ============================================================================
# T2 - Confronto assoluto tra stadi su scala logaritmica
#      Ordine di grandezza per stadio e per config, media su n_qubits e seed
# ============================================================================
print("=" * 70)
print("T2  Confronto assoluto tra stadi: scala log")
print("=" * 70)

CFGS_ORDERED = [
    ("amplitude", "pca"),
    ("amplitude", "word2ket"),
    ("angle_X",   "pca"),
    ("angle_X",   "w2k-angle"),
]
CFG_LABEL = {
    ("amplitude", "pca"):       "Ampl+PCA",
    ("amplitude", "word2ket"):  "Ampl+W2K",
    ("angle_X",   "pca"):       "AnglX+PCA",
    ("angle_X",   "w2k-angle"): "AnglX+W2K",
}
STAGE_COLS   = ["t_preproc_total_s", "t_train_epoch_mean_s", "t_test_inference_s"]
STAGE_NAMES  = ["Preprocessing", "Train/epoch", "Test inference"]
STAGE_COLORS = ["#6D28D9", "#0369A1", "#B45309"]

log_rows = []
for enc, compr in CFGS_ORDERED:
    for sbert in MODELS:
        sub = data[(data["encoding"] == enc) &
                   (data["compression_method"] == compr) &
                   (data["sbert_short"] == sbert)]
        if sub.empty: continue
        row = {"label": f"{CFG_LABEL[(enc,compr)]}\n[{sbert}]",
               "enc": enc, "compr": compr, "sbert": sbert}
        for sc in STAGE_COLS:
            row[sc] = sub[sc].mean()
        log_rows.append(row)
log_df = pd.DataFrame(log_rows)

x = np.arange(len(log_df)); w = 0.22
fig, ax = plt.subplots(figsize=(13, 5))
for i, (sc, sname, scol) in enumerate(zip(STAGE_COLS, STAGE_NAMES, STAGE_COLORS)):
    ax.bar(x + (i - 1) * w, log_df[sc], w, label=sname, color=scol, alpha=0.82)

ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(log_df["label"], fontsize=8.5)
ax.set_ylabel("Tempo medio (s, scala log)")
ax.legend(fontsize=15, title="Stadio", title_fontsize=18,
          ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v:.3f}s" if v < 1 else f"{v:.1f}s"
))
ax.grid(axis="y", which="both", alpha=0.3)
for i, (sc, scol) in enumerate(zip(STAGE_COLS, STAGE_COLORS)):
    for j, val in enumerate(log_df[sc]):
        ax.text(x[j] + (i-1)*w, val * 1.4,
                f"{val:.3f}" if val < 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=6.5, color=scol, rotation=90)

fig.tight_layout()
fig.savefig(PLOTS_DIR / "t2_log_stage_comparison.png", bbox_inches="tight")
plt.close(fig)
print("  [plot] t2_log_stage_comparison.png")


# ============================================================================
# P3 - F1 macro vs n_qubits  (layout per SBERT model)
#      Analogo ad A1 ma diviso per SBERT model invece che per encoding.
#      4 linee per panel: una per ogni combinazione (encoding + compression).
# ============================================================================
print("=" * 70)
print("P3  test_f1_macro vs n_qubits per SBERT model")
print("=" * 70)

def _draw_p3(fig, axes):
    for ax, sbert in zip(axes, MODELS):
        for enc in ENCODINGS:
            for compr in COMPRS[enc]:
                sub = data[(data["encoding"] == enc) &
                           (data["compression_method"] == compr) &
                           (data["sbert_short"] == sbert)]
                if sub.empty: continue
                grp = sub.groupby("n_qubits")["test_f1_macro"]
                mu  = grp.mean(); sd = grp.std().fillna(0)
                c   = cfg_color(enc, compr)
                lbl = f"{ENC_TITLE[enc].split()[0]} + {COMPR_DISP[compr]}"
                ax.plot(mu.index, mu.values, color=c, lw=2, label=lbl,
                        marker="o", ms=5)
                ax.fill_between(mu.index, mu - sd, mu + sd, alpha=0.12, color=c)
        ax.set_xlabel("n_qubits"); ax.set_ylabel("test_f1_macro")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    # legenda singola condivisa sopra la figura
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Encoding + Reduction", fontsize=15, title_fontsize=18,
               ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.01), bbox_transform=fig.transFigure)

dual_save(_draw_p3, "p3_f1_vs_qubits_by_sbert")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print()
print("=" * 70)
print("SUMMARY - aggregato su tutti i qubit e seed")
print("  (epoch_mean da data, resto da data completo)")
print("=" * 70)

s_perf = (data.groupby(["encoding", "compression_method"])
              .agg(n=("test_f1_macro","count"),
                   f1_mean=("test_f1_macro","mean"),
                   f1_std=("test_f1_macro","std"),
                   f1_cov=("test_f1_macro", lambda x: x.std()/x.mean()),
                   gen_gap_mean=("gen_gap","mean"),
                   stopped_mean=("stopped_at","mean"),
                   conv_ratio_mean=("conv_ratio","mean"))
              .round(4))
s_time = (data.groupby(["encoding","compression_method"])
              ["t_train_epoch_mean_s"].mean().round(3)
              .rename("epoch_s_mean"))
summary = s_perf.join(s_time)
print(summary.to_string())
summary.to_csv(OUT_DIR / "summary_timing_stability.csv")
print(f"\n  [ok] summary_timing_stability.csv")

# ============================================================================
# PER-QUBIT F1 STATS  (data behind Table 1; was compute_table.py)
#   mean +/- std of test macro F1 per (backbone, encoding, compression, n_qubits)
# ============================================================================
f1_stats = (data.groupby(["sbert_short", "encoding", "compression_method", "n_qubits"])
                ["test_f1_macro"]
                .agg(mean="mean", std="std", count="count")
                .reset_index())
f1_stats["std"] = f1_stats["std"].fillna(0.0)
f1_stats = f1_stats.sort_values(
    ["sbert_short", "encoding", "compression_method", "n_qubits"])
f1_stats.to_csv(OUT_DIR / "f1_stats.csv", index=False)
print(f"  [ok] f1_stats.csv")

# ============================================================================
print("\n" + "=" * 70)
print(f"Done -> {OUT_DIR.resolve()}")
for f in sorted(OUT_DIR.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUT_DIR)}")
print("=" * 70)