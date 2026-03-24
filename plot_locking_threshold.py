"""
锁定阈值分析 — 揭示高阶耦合的非经典锁定机制
数据: locking_order.json (sim/parameter-scan)
KuramotoThinker Task Y.1
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


def plot_locking_threshold(data_path="locking_order.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    omega = np.array(data["omega_i"])
    omega_abs = np.abs(omega)
    t_neg = np.array(data["t_lock_K3neg"])
    t_zero = np.array(data["t_lock_K3zero"])
    t_pos = np.array(data["t_lock_K3pos"])
    K2 = data["K2"]

    configs = [
        (t_neg, "$K_3 < 0$", "#E53935"),
        (t_zero, "$K_3 = 0$", "#757575"),
        (t_pos, "$K_3 > 0$", "#1E88E5"),
    ]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Row 1: Locking probability P(locked | omega) sigmoid curves
    ax_sig = fig.add_subplot(gs[0, :2])

    omega_bins = np.linspace(0, omega_abs.max(), 20)
    bin_centers = (omega_bins[:-1] + omega_bins[1:]) / 2

    for t_lock, label, color in configs:
        locked = t_lock >= 0
        probs = []
        for j in range(len(omega_bins) - 1):
            mask = (omega_abs >= omega_bins[j]) & (omega_abs < omega_bins[j + 1])
            if mask.sum() > 0:
                probs.append(locked[mask].mean())
            else:
                probs.append(np.nan)
        ax_sig.plot(bin_centers, probs, "o-", color=color, ms=5, lw=1.5, label=label)

    ax_sig.axhline(0.5, color="gray", ls=":", lw=0.5)
    ax_sig.set_xlabel("$|\\omega_i|$")
    ax_sig.set_ylabel("$P(\\mathrm{locked} \\mid \\omega)$")
    ax_sig.set_title("Locking probability vs. natural frequency")
    ax_sig.legend(fontsize=9)
    ax_sig.grid(True, alpha=0.15)
    ax_sig.text(-0.08, 1.06, "A", transform=ax_sig.transAxes,
                fontsize=14, fontweight="bold", va="top")

    # Row 1 right: omega_c vs K3 (schematic from 3 points)
    ax_oc = fig.add_subplot(gs[0, 2])

    omega_c_vals = []
    K3_labels = [-1, 0, 1]
    for t_lock in [t_neg, t_zero, t_pos]:
        locked = t_lock >= 0
        if locked.any() and (~locked).any():
            # omega_c = max |omega| where P(locked) > 0.5
            omega_sorted = np.sort(omega_abs)
            locked_sorted = locked[np.argsort(omega_abs)]
            # rolling mean
            window = 20
            for idx in range(len(omega_sorted) - window):
                if locked_sorted[idx:idx + window].mean() < 0.5:
                    omega_c_vals.append(omega_sorted[idx])
                    break
            else:
                omega_c_vals.append(omega_sorted[-1])
        else:
            omega_c_vals.append(0)

    ax_oc.bar(K3_labels, omega_c_vals, width=0.6,
              color=["#E53935", "#757575", "#1E88E5"], alpha=0.8)
    ax_oc.set_xlabel("$K_3$")
    ax_oc.set_ylabel("Critical frequency $\\omega_c$")
    ax_oc.set_title("$\\omega_c$ shifts with $K_3$")
    ax_oc.grid(True, alpha=0.15, axis="y")
    ax_oc.text(-0.15, 1.06, "B", transform=ax_oc.transAxes,
               fontsize=14, fontweight="bold", va="top")

    # Row 2: locked vs unlocked frequency distributions
    for pi, (t_lock, label, color) in enumerate(configs):
        ax = fig.add_subplot(gs[1, pi])
        locked = t_lock >= 0
        unlocked = ~locked

        bins = np.linspace(0, omega_abs.max(), 15)
        if locked.any():
            ax.hist(omega_abs[locked], bins=bins, alpha=0.6, color=color,
                    label=f"Locked ({locked.sum()})", density=True)
        if unlocked.any():
            ax.hist(omega_abs[unlocked], bins=bins, alpha=0.4, color="red",
                    edgecolor="red", linewidth=1.2, histtype="step",
                    label=f"Unlocked ({unlocked.sum()})", density=True)

        ax.set_xlabel("$|\\omega_i|$")
        if pi == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.text(-0.12 if pi == 0 else -0.05, 1.06, chr(67 + pi),
                transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.suptitle(
        f"Locking threshold analysis ($K_2={K2}$, $N={len(omega)}$)\n"
        "Non-classical: low-$|\\omega|$ oscillators also fail to lock at negative $K_3$",
        fontsize=12, y=1.04)

    path = os.path.join(FIG_DIR, "locking_threshold.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")

    # Print omega_c values
    print(f"\nomega_c values: K3<0: {omega_c_vals[0]:.3f}, K3=0: {omega_c_vals[1]:.3f}, K3>0: {omega_c_vals[2]:.3f}")
    return fig


if __name__ == "__main__":
    plot_locking_threshold()
