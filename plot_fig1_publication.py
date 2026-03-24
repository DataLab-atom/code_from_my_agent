"""
Fig 1 — Publication quality: K₂×K₃ phase diagram with literature reconciliation
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def plot_fig1():
    os.makedirs("paper/figures", exist_ok=True)

    with open("scan_K2_K3.json") as f:
        d = json.load(f)

    K2 = np.array(d["K2_list"])
    K3 = np.array(d["K3_list"])
    r = np.array(d["r"])
    basin = np.array(d["basin_prob"])
    tc = np.array(d["tc"])
    sigma = d["sigma"]
    Kc = 2 * sigma * np.sqrt(2 * np.pi) / np.pi

    K3g, K2g = np.meshgrid(K3, K2)

    # Load analytic overlay
    try:
        with open("analytical_phase_diagram.json") as f:
            ad = json.load(f)
        ar = np.array(ad["sigma_results"][f"sigma={sigma}"]["r"])
        aK2 = np.array(ad["K2_range"])
        aK3 = np.array(ad["K3_range"])
    except:
        ar = None

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.08, right=0.95, top=0.93, bottom=0.07)

    # ═══ Panel a: Order parameter r ═══
    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(K3g, K2g, r, cmap="viridis", vmin=0, vmax=1,
                       shading="auto", rasterized=True)
    cs = ax.contour(K3g, K2g, r, levels=[0.3, 0.5, 0.7],
                    colors=["w"], linewidths=[0.4, 0.8, 0.4],
                    linestyles=["--", "-", "--"])
    ax.clabel(cs, fmt="%.1f", fontsize=5, colors="w")

    # OA analytic r=0.5 overlay
    if ar is not None:
        aK3g, aK2g = np.meshgrid(aK3, aK2)
        ax.contour(aK3g, aK2g, ar, levels=[0.5],
                   colors=["cyan"], linewidths=[1.0], linestyles=["--"])

    # Onset + explosive
    ax.axhline(Kc, color="w", ls=":", lw=0.6, alpha=0.8)
    k_diag = np.linspace(max(K2.min(), K3.min()), min(K2.max(), K3.max()), 50)
    ax.plot(k_diag, k_diag, "w--", lw=0.6, alpha=0.6)

    # Literature boxes
    for (x, y, w, h, label, color) in [
        (-0.3, Kc * 0.7, 0.6, Kc * 0.4, "Muolo\n2025", "#FFEB3B"),
        (1.0, Kc * 1.3, 0.8, 1.5, "Zhang\n2024", "#FF9800"),
        (0.2, Kc * 1.1, 0.6, Kc * 0.5, "Wang\n2025", "#76FF03"),
    ]:
        rect = Rectangle((x, y), w, h, fill=False, edgecolor=color,
                          lw=0.8, ls="--", alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.5, label, color=color, fontsize=5,
                ha="center", va="center", fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("$r$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("$K_3$")
    ax.set_ylabel("$K_2$")
    ax.text(-0.20, 1.05, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    # ═══ Panel b: Basin probability ═══
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(K3g, K2g, basin, cmap="RdBu", vmin=0, vmax=1,
                          shading="auto", rasterized=True)
    cs2 = ax2.contour(K3g, K2g, basin, levels=[0.5],
                      colors=["k"], linewidths=[0.8])
    ax2.axhline(Kc, color="k", ls=":", lw=0.5, alpha=0.5)
    ax2.plot(k_diag, k_diag, "k--", lw=0.5, alpha=0.5)

    # K3>0 creates basin annotation
    ax2.annotate("$K_3>0$ creates\nsync basin", xy=(1.5, 1.5),
                 fontsize=5.5, ha="center", color="#1565C0", fontweight="bold")
    ax2.annotate("", xy=(1.2, 1.2), xytext=(0.3, 0.5),
                 arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8))

    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cb2.set_label("Basin prob.", fontsize=7)
    cb2.ax.tick_params(labelsize=6)
    ax2.set_xlabel("$K_3$")
    ax2.set_ylabel("$K_2$")
    ax2.text(-0.20, 1.05, "b", transform=ax2.transAxes, fontsize=10, fontweight="bold")

    # ═══ Panel c: Convergence time ═══
    ax3 = fig.add_subplot(gs[1, 0])
    tc_clip = np.clip(tc, 0, np.percentile(tc, 90))
    im3 = ax3.pcolormesh(K3g, K2g, tc_clip, cmap="magma_r",
                          shading="auto", rasterized=True)
    ax3.axhline(Kc, color="cyan", ls=":", lw=0.5)
    ax3.plot(k_diag, k_diag, "cyan", ls="--", lw=0.5, alpha=0.5)

    cb3 = fig.colorbar(im3, ax=ax3, shrink=0.85, pad=0.02)
    cb3.set_label("Conv. time", fontsize=7)
    cb3.ax.tick_params(labelsize=6)
    ax3.set_xlabel("$K_3$")
    ax3.set_ylabel("$K_2$")
    ax3.text(-0.20, 1.05, "c", transform=ax3.transAxes, fontsize=10, fontweight="bold")

    # ═══ Panel d: Three metrics summary at K₂=2.1 ═══
    ax4 = fig.add_subplot(gs[1, 1])
    K2_idx = np.argmin(np.abs(K2 - 2.1))

    ax4.plot(K3, r[K2_idx, :], "o-", color="#1976D2", ms=3, lw=1.0,
             label="$r$")
    ax4.plot(K3, basin[K2_idx, :], "s-", color="#4CAF50", ms=3, lw=1.0,
             label="Basin")

    ax4_tc = ax4.twinx()
    tc_norm = tc[K2_idx, :] / max(tc[K2_idx, :].max(), 1)
    ax4_tc.plot(K3, tc_norm, "^--", color="#FF9800", ms=2.5, lw=0.8,
                label="Conv. time (norm.)")
    ax4_tc.set_ylabel("Norm. conv. time", fontsize=7, color="#FF9800")
    ax4_tc.tick_params(axis="y", labelcolor="#FF9800", labelsize=6)

    ax4.set_xlabel("$K_3$")
    ax4.set_ylabel("$r$ / Basin prob.", fontsize=7)
    ax4.axvline(0, color="0.7", ls=":", lw=0.3)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_tc.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, frameon=True,
               fancybox=False, edgecolor="0.7", fontsize=5.5, loc="lower right")
    ax4.text(-0.20, 1.05, "d", transform=ax4.transAxes, fontsize=10, fontweight="bold")
    ax4.set_title(f"$K_2 = {K2[K2_idx]:.1f}$ slice", fontsize=7)

    fig.text(0.5, 0.97,
             f"$K_2 \\times K_3$ phase diagram ($\\sigma={sigma}$, $N={d['N']}$)",
             ha="center", fontsize=9, fontweight="bold")

    for fmt in ["pdf", "png"]:
        fig.savefig(f"paper/figures/fig1.{fmt}")
    print("Saved: paper/figures/fig1.pdf")


if __name__ == "__main__":
    plot_fig1()
