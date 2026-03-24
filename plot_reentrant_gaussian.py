"""
Gaussian 精确 re-entrant 边界 — 论文核心 figure
数据: MathAgent 直接提供的 re-entrant 边界点
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 1.0,
    "lines.linewidth": 2,
})

FIG_DIR = "figures"


def plot_reentrant():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Gaussian re-entrant boundary (from MathAgent)
    reentrant_pts = [
        (1.63, 5.77), (1.76, 5.57), (1.89, 5.47), (2.03, 5.37),
        (2.16, 5.16), (2.29, 5.06), (2.42, 4.96), (2.55, 4.76),
        (2.68, 4.66), (2.82, 4.56), (2.95, 4.35), (3.08, 4.25),
        (3.21, 4.15), (3.34, 3.95), (3.47, 3.85), (3.61, 3.75),
        (3.74, 3.54), (3.87, 3.44), (4.00, 3.34),
    ]
    re_K2 = np.array([p[0] for p in reentrant_pts])
    re_K3 = np.array([p[1] for p in reentrant_pts])

    sigma = 1.0
    Delta = sigma * np.sqrt(2 * np.pi) / np.pi
    Kc = 2 * Delta  # 1.596

    fig, ax = plt.subplots(figsize=(8, 7))

    # Fill regions
    K2_range = np.linspace(0, 5, 200)
    K3_range = np.linspace(0, 7, 200)

    # Incoherent region (below Kc)
    ax.fill_between(K3_range, 0, Kc, alpha=0.08, color="gray")
    ax.text(3.5, 0.7, "Incoherent\n($K_2 < K_c$)", fontsize=11,
            ha="center", color="0.4", style="italic")

    # Sync region (above Kc, below explosive and reentrant)
    ax.fill_between([0, 2], Kc, 5, alpha=0.08, color="#2196F3")
    ax.text(0.5, 3.0, "Stable\nsynchrony", fontsize=11,
            ha="center", color="#1565C0", fontweight="bold")

    # Re-entrant region annotation
    ax.annotate("Re-entrant:\n$K_3$ first helps,\nthen destroys sync",
                xy=(3.0, 4.2), xytext=(1.0, 5.5),
                fontsize=10, color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#E53935", alpha=0.9))

    # OA onset: K2 = Kc (horizontal)
    ax.axhline(Kc, color="#2196F3", ls="--", lw=2.5, alpha=0.8,
               label=f"OA onset: $K_2^c = 2\\Delta = {Kc:.3f}$")

    # OA explosive: K3 = K2 (diagonal)
    k_diag = np.linspace(0, 5, 100)
    ax.plot(k_diag, k_diag, color="#FF9800", ls="--", lw=2,
            label="Explosive boundary: $K_3 = K_2$")

    # Gaussian re-entrant boundary
    ax.plot(re_K3, re_K2, "o-", color="#E53935", ms=5, lw=2.5,
            label="Re-entrant boundary (Gaussian exact)", zorder=10)

    # Time-delay reachable region (K3 >= 0 only, K3 < K2)
    K2_td = np.linspace(Kc, 5, 100)
    ax.fill_between(np.minimum(K2_td, K2_td), K2_td, 5,
                    where=K2_td > 0, alpha=0.04, color="#4CAF50")
    ax.text(0.3, 4.5, "Time-delay\nreachable", fontsize=9,
            color="#2E7D32", style="italic", alpha=0.8)

    ax.set_xlabel("$K_3$ (three-body coupling strength)")
    ax.set_ylabel("$K_2$ (pairwise coupling strength)")
    ax.set_title("Phase diagram with re-entrant synchronization boundary\n"
                 f"($\\sigma = {sigma}$, Gaussian frequency distribution)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.1)

    path = os.path.join(FIG_DIR, "fig_reentrant_gaussian.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_reentrant()
