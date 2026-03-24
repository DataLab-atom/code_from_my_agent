"""
Fig 5 — Publication quality: Topology controls K₃ efficacy
Target: Nature/NeurIPS single-column width (3.5 inch)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import os

# Nature-style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "axes.grid": False,
    "pdf.fonttype": 42,  # TrueType for editability
    "ps.fonttype": 42,
})

FIG_DIR = "paper/figures"


def plot_fig5():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load sparse network data
    with open("sparse_network_K3.json") as f:
        sparse = json.load(f)

    # Load all-to-all K3 asymmetry for comparison
    with open("K3_asymmetry.json") as f:
        asym = json.load(f)

    # Load finite-size data
    with open("finite_size_very_large_N.json") as f:
        fs = json.load(f)

    fig = plt.figure(figsize=(7.0, 5.5))  # double-column
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.08, right=0.97, top=0.93, bottom=0.08)

    # ═══ Panel a: Sparse vs Dense — the key result ═══
    ax_a = fig.add_subplot(gs[0, 0])

    from collections import defaultdict
    by_p = defaultdict(list)
    for r in sparse["results"]:
        by_p[r["p"]].append(r)

    palette = {"0.1": "#D32F2F", "0.3": "#F57C00", "1.0": "#1976D2"}
    markers = {"0.1": "v", "0.3": "s", "1.0": "o"}

    for p in [0.1, 0.3, 1.0]:
        entries = by_p[p]
        K3 = [e["K3"] for e in entries]
        r_mean = [e["r_mean"] for e in entries]
        r_std = [e.get("r_std", 0) for e in entries]
        c = palette[str(p)]
        m = markers[str(p)]
        n_tri = entries[0]["n_triangles"]
        label = f"$p={p}$  ({n_tri:,} $\\triangle$)"
        ax_a.errorbar(K3, r_mean, yerr=r_std, fmt=f"{m}-", color=c,
                      ms=4, lw=1.2, capsize=1.5, capthick=0.5, label=label)

    ax_a.set_xlabel("$K_3$ (three-body coupling)")
    ax_a.set_ylabel("Order parameter $r$")
    ax_a.legend(frameon=True, fancybox=False, edgecolor="0.7",
                borderpad=0.3, handlelength=1.5, loc="upper left")
    ax_a.set_ylim(-0.02, 1.0)
    ax_a.text(-0.18, 1.05, "a", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ═══ Panel b: r vs K₃ at all-to-all (fine scan) ═══
    ax_b = fig.add_subplot(gs[0, 1])

    K3_a = np.array(asym["K3_list"])
    r1 = np.array(asym["r1_steady"])
    r2 = np.array(asym["r2_steady"])

    ax_b.plot(K3_a, r1, "-", color="#1976D2", lw=1.2, label="$r_1$")
    ax_b.plot(K3_a, r2, "--", color="#F57C00", lw=1.0, label="$r_2$")
    ax_b.fill_between(K3_a[K3_a >= 0], 0, r1[K3_a >= 0],
                       alpha=0.08, color="#1976D2")
    ax_b.fill_between(K3_a[K3_a <= 0], 0, r1[K3_a <= 0],
                       alpha=0.08, color="#D32F2F")
    ax_b.axvline(0, color="0.5", ls=":", lw=0.4)
    ax_b.set_xlabel("$K_3$")
    ax_b.set_ylabel("Order parameter")
    ax_b.legend(frameon=True, fancybox=False, edgecolor="0.7",
                borderpad=0.3, handlelength=1.2, loc="lower right")
    ax_b.text(-0.18, 1.05, "b", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")
    ax_b.set_title("All-to-all ($N=200$, $\\sigma=1$, $K_2=2$)", fontsize=7)

    # ═══ Panel c: Finite-size scaling N→20000 ═══
    ax_c = fig.add_subplot(gs[1, 0])

    results = fs["results"]
    N_list = [r["N"] for r in results]
    delta = [r["K3=1.0"] - r["K3=0.0"] for r in results]

    # Gaussian exact prediction
    delta_inf = 0.909 - 0.716

    ax_c.plot(N_list, delta, "ko-", ms=4, lw=1.2,
              label="Numerical $\\Delta r$")
    ax_c.axhline(delta_inf, color="#4CAF50", ls="--", lw=0.8,
                 label=f"Gaussian exact ($N\\to\\infty$): {delta_inf:.3f}")

    # OA prediction
    sigma = fs["sigma"]
    Delta = sigma * np.sqrt(2 * np.pi) / np.pi
    K2 = fs["K2"]

    def oa_r(K2v, K3v):
        a, b, c = K3v, -(K2v + K3v), K2v - 2 * Delta
        if abs(a) < 1e-12:
            u = -c / b if abs(b) > 1e-12 else 0
            return np.sqrt(u) if 0 < u < 1 else 0
        disc = b * b - 4 * a * c
        if disc < 0:
            return 0
        sols = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        rs = [np.sqrt(u) for u in sols if 0 < u < 1]
        return max(rs) if rs else 0

    delta_oa = oa_r(K2, 1.0) - oa_r(K2, 0.0)
    ax_c.axhline(delta_oa, color="#D32F2F", ls="--", lw=0.8,
                 label=f"Lorentzian OA: {delta_oa:+.3f}")
    ax_c.axhline(0, color="0.7", ls=":", lw=0.3)

    ax_c.set_xscale("log")
    ax_c.set_xlabel("$N$ (system size)")
    ax_c.set_ylabel("$\\Delta r$ ($K_3$=+1 vs 0)")
    ax_c.legend(frameon=True, fancybox=False, edgecolor="0.7",
                borderpad=0.3, handlelength=1.2, fontsize=5.5, loc="center right")
    ax_c.text(-0.18, 1.05, "c", transform=ax_c.transAxes,
              fontsize=10, fontweight="bold")

    # ═══ Panel d: Locking count bar chart ═══
    ax_d = fig.add_subplot(gs[1, 1])

    with open("locking_order.json") as f:
        lock = json.load(f)

    t_neg = np.array(lock["t_lock_K3neg"])
    t_zero = np.array(lock["t_lock_K3zero"])
    t_pos = np.array(lock["t_lock_K3pos"])
    N_osc = len(lock["omega_i"])

    counts = [(t_neg >= 0).sum(), (t_zero >= 0).sum(), (t_pos >= 0).sum()]
    labels = ["$K_3<0$", "$K_3=0$", "$K_3>0$"]
    colors = ["#D32F2F", "#757575", "#1976D2"]

    bars = ax_d.bar(labels, counts, color=colors, width=0.6, edgecolor="0.3",
                     linewidth=0.3, alpha=0.85)
    for bar, c in zip(bars, counts):
        ax_d.text(bar.get_x() + bar.get_width() / 2, c + 3,
                  f"{c}/{N_osc}", ha="center", fontsize=6, fontweight="bold")

    ax_d.set_ylabel("Locked oscillators")
    ax_d.set_ylim(0, 210)
    ax_d.axhline(N_osc, color="0.7", ls=":", lw=0.3)
    ax_d.text(-0.18, 1.05, "d", transform=ax_d.transAxes,
              fontsize=10, fontweight="bold")
    ax_d.set_title("Winner-takes-all mechanism", fontsize=7)

    # Suptitle
    fig.text(0.5, 0.97,
             "Higher-order coupling efficacy depends on network topology",
             ha="center", fontsize=9, fontweight="bold")

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{FIG_DIR}/fig5.{fmt}")
    print(f"Saved: {FIG_DIR}/fig5.pdf")


if __name__ == "__main__":
    plot_fig5()
