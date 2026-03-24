"""
临界慢化现象 — 相变附近的弛豫时间发散
数据: critical_slowing.json (sim/parameter-scan)
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
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
})

FIG_DIR = "figures"


def plot_critical_slowing(data_path="critical_slowing.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    K2 = np.array(data["K2_list"])
    K3_list = data["K3_list"]
    tau = [np.array(t) for t in data["tau_relax"]]
    r_ss = [np.array(r) for r in data["r_steady"]]
    Kc = data["Kc_est"]

    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(K3_list)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.3})

    # Panel A: τ vs K₂
    ax = axes[0]
    for i, (K3, t, c) in enumerate(zip(K3_list, tau, colors)):
        ax.plot(K2, t, "o-", color=c, ms=4, lw=1.5, label=f"$K_3={K3:+.1f}$")
    ax.axvline(Kc, color="gray", ls="--", lw=1, label=f"$K_c \\approx {Kc}$")
    ax.set_xlabel("$K_2$")
    ax.set_ylabel("Relaxation time $\\tau$")
    ax.set_title("Critical slowing down near $K_c$")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_yscale("log")
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Panel B: r vs K₂
    ax2 = axes[1]
    for i, (K3, r, c) in enumerate(zip(K3_list, r_ss, colors)):
        ax2.plot(K2, r, "o-", color=c, ms=4, lw=1.5, label=f"$K_3={K3:+.1f}$")
    ax2.axvline(Kc, color="gray", ls="--", lw=1)
    ax2.axhline(0.5, color="gray", ls=":", lw=0.5)
    ax2.set_xlabel("$K_2$")
    ax2.set_ylabel("Steady-state $r$")
    ax2.set_title("Order parameter near transition")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.15)
    ax2.text(-0.12, 1.06, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")

    fig.suptitle(f"Critical slowing down ($N={data['N']}$, $\\sigma={data['sigma']}$)",
                 fontsize=13, y=1.02)

    path = os.path.join(FIG_DIR, "critical_slowing.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_critical_slowing()
