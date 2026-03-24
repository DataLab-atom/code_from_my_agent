"""
滞后回线 — 揭示爆炸式同步 (K₃>K₂ 区域)
数据: hysteresis.json (sim/parameter-scan)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "figure.dpi": 200,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
})

FIG_DIR = "figures"


def plot_hysteresis(data_path="hysteresis.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    K2 = np.array(data["K2_list"])
    K3_list = data["K3_list"]
    r_fwd = [np.array(r) for r in data["r_forward"]]
    r_bwd = [np.array(r) for r in data["r_backward"]]
    sigma = data["sigma"]
    N = data["N"]

    nK3 = len(K3_list)
    ncols = min(4, nK3)
    nrows = (nK3 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.3, "wspace": 0.15})
    axes = axes.flatten()

    for i, K3 in enumerate(K3_list):
        ax = axes[i]
        ax.plot(K2, r_fwd[i], "o-", color="#2196F3", ms=3, lw=1.2,
                label="Forward ($K_2$ ↑)")
        ax.plot(K2, r_bwd[i], "s-", color="#E53935", ms=3, lw=1.2,
                label="Backward ($K_2$ ↓)")

        # hysteresis area
        gap = np.abs(r_fwd[i] - r_bwd[i])
        hysteresis_area = np.trapz(gap, K2)
        ax.fill_between(K2, r_fwd[i], r_bwd[i], alpha=0.15, color="purple")

        ax.set_title(f"$K_3={K3:+.1f}$  (H={hysteresis_area:.2f})", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.15)
        ax.text(-0.12 if i % ncols == 0 else -0.05, 1.06, chr(65 + i),
                transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("$K_2$")
        if i % ncols == 0:
            ax.set_ylabel("$r$")
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.9)

    for j in range(nK3, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Hysteresis loops: forward vs backward $K_2$ sweeps ($\\sigma={sigma}$, $N={N}$)",
                 fontsize=12, y=1.02)

    path = os.path.join(FIG_DIR, "hysteresis.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_hysteresis()
