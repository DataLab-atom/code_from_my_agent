"""
K₃ 正负不对称性深度分析
数据: K3_asymmetry.json (sim/parameter-scan, Task C.1)
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


def plot_K3_asymmetry(data_path="K3_asymmetry.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    K3 = np.array(data["K3_list"])
    r1 = np.array(data["r1_steady"])
    r2 = np.array(data["r2_steady"])
    basin = np.array(data["basin_prob"])
    ts_all = data["r1_timeseries"]
    K2 = data["K2"]
    sigma = data["sigma"]
    rec_every = data["record_every_steps"]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: r₁ and r₂ vs K₃
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(K3, r1, "o-", color="#2196F3", ms=3, lw=1.2, label="$r_1$")
    ax_a.plot(K3, r2, "s-", color="#FF9800", ms=3, lw=1.2, label="$r_2$")
    ax_a.axvline(0, color="gray", ls="--", lw=0.6)
    ax_a.fill_betweenx([0, 1], K3.min(), 0, alpha=0.05, color="red")
    ax_a.fill_betweenx([0, 1], 0, K3.max(), alpha=0.05, color="blue")
    ax_a.set_xlabel("$K_3$")
    ax_a.set_ylabel("Order parameter")
    ax_a.set_title("$r_1$ and $r_2$ vs $K_3$")
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.15)
    ax_a.text(-0.15, 1.06, "A", transform=ax_a.transAxes,
              fontsize=13, fontweight="bold", va="top")

    # Panel B: basin probability vs K₃
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(K3, basin, "D-", color="#4CAF50", ms=3, lw=1.2)
    ax_b.axvline(0, color="gray", ls="--", lw=0.6)
    ax_b.set_xlabel("$K_3$")
    ax_b.set_ylabel("Basin probability")
    ax_b.set_title("Synchronization accessibility")
    ax_b.grid(True, alpha=0.15)
    ax_b.text(-0.15, 1.06, "B", transform=ax_b.transAxes,
              fontsize=13, fontweight="bold", va="top")

    # Panel C: r₂/r₁² ratio (OA predicts r₂ = r₁²)
    ax_c = fig.add_subplot(gs[0, 2])
    r1_sq = r1 ** 2
    ratio = np.where(r1_sq > 0.01, r2 / r1_sq, np.nan)
    ax_c.plot(K3, ratio, "^-", color="#9C27B0", ms=3, lw=1.2)
    ax_c.axhline(1, color="red", ls="--", lw=1, label="OA prediction: $r_2 = r_1^2$")
    ax_c.axvline(0, color="gray", ls="--", lw=0.6)
    ax_c.set_xlabel("$K_3$")
    ax_c.set_ylabel("$r_2 / r_1^2$")
    ax_c.set_title("OA validity check")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.15)
    ax_c.text(-0.15, 1.06, "C", transform=ax_c.transAxes,
              fontsize=13, fontweight="bold", va="top")

    # Panel D: r₁(t) time series for 3 representative K₃ values
    ax_d = fig.add_subplot(gs[1, :])
    # K₃ = -2, 0, +2 (or nearest)
    targets = [-2.0, 0.0, 2.0]
    colors_ts = ["#E53935", "#757575", "#1E88E5"]
    for target, clr in zip(targets, colors_ts):
        idx = np.argmin(np.abs(K3 - target))
        ts = np.array(ts_all[idx])
        t_axis = np.arange(len(ts)) * rec_every * 0.01  # dt=0.01
        ax_d.plot(t_axis, ts, color=clr, lw=0.8, alpha=0.9,
                  label=f"$K_3={K3[idx]:+.1f}$")

    ax_d.set_xlabel("Time $t$")
    ax_d.set_ylabel("$r_1(t)$")
    ax_d.set_title("Order parameter time series")
    ax_d.legend(fontsize=9, framealpha=0.9)
    ax_d.grid(True, alpha=0.15)
    ax_d.text(-0.05, 1.06, "D", transform=ax_d.transAxes,
              fontsize=13, fontweight="bold", va="top")

    fig.suptitle(f"$K_3$ asymmetry analysis ($K_2={K2}$, $\\sigma={sigma}$, $N={data['N']}$)",
                 fontsize=14, y=1.01)

    path = os.path.join(FIG_DIR, "K3_asymmetry.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_K3_asymmetry()
