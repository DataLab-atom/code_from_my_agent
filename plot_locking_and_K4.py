"""
锁定顺序分析 + K₄ 四体耦合效应
数据来源: GPU-Claude-Opus (sim/parameter-scan 分支)
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


def plot_locking_order(data_path="locking_order.json"):
    """
    振荡器锁定时间 vs 自然频率 |ωᵢ|
    比较 K₃<0, K₃=0, K₃>0 三种情况
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    omega = np.array(data["omega_i"])
    t_neg = np.array(data["t_lock_K3neg"])
    t_zero = np.array(data["t_lock_K3zero"])
    t_pos = np.array(data["t_lock_K3pos"])
    K2 = data["K2"]
    sigma = data["sigma"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                              sharey=True, gridspec_kw={"wspace": 0.08})

    configs = [
        (t_neg, "$K_3 < 0$", "#E53935"),
        (t_zero, "$K_3 = 0$", "#757575"),
        (t_pos, "$K_3 > 0$", "#1E88E5"),
    ]

    for ax, (t_lock, label, color) in zip(axes, configs):
        # -1 表示未锁定, -0.01 表示初始即锁定
        locked = t_lock >= 0
        unlocked = t_lock < -0.5

        ax.scatter(np.abs(omega[locked]), t_lock[locked],
                   c=color, s=15, alpha=0.6, edgecolors="none")
        ax.scatter(np.abs(omega[unlocked]),
                   np.full(unlocked.sum(), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 6),
                   c=color, s=15, alpha=0.3, marker="x")

        n_locked = locked.sum()
        n_total = len(omega)

        ax.set_xlabel("$|\\omega_i|$ (natural frequency)")
        ax.set_title(f"{label} ({n_locked}/{n_total} locked)")
        ax.grid(True, alpha=0.15)

        # 趋势线
        if locked.sum() > 5:
            valid = (t_lock > 0) & locked
            if valid.sum() > 5:
                z = np.polyfit(np.abs(omega[valid]), t_lock[valid], 1)
                x_fit = np.linspace(0, np.abs(omega).max(), 50)
                ax.plot(x_fit, np.polyval(z, x_fit), "--", color=color, lw=1, alpha=0.7)

    axes[0].set_ylabel("Locking time $t_{\\mathrm{lock}}$")

    fig.suptitle(f"Oscillator locking order ($K_2={K2}$, $\\sigma={sigma}$, $N={len(omega)}$)",
                 fontsize=13, y=1.02)

    for i, ax in enumerate(axes):
        ax.text(-0.12 if i == 0 else -0.05, 1.06, chr(65 + i),
                transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    path = os.path.join(FIG_DIR, "locking_order.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


def plot_K4_effect(data_path="K4_test.json"):
    """
    K₄ 四体耦合对 r 的影响
    GPU 发现 "K₄ is NOT negligible"
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    K4 = np.array(data["K4_list"])
    r1 = np.array(data["r1"])
    K2 = data["K2"]
    K3 = data["K3"]
    sigma = data["sigma"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(K4, r1, "o-", color="#4CAF50", ms=5, lw=1.8)
    ax.axvline(0, color="gray", ls="--", lw=0.6)
    ax.axhline(r1[np.argmin(np.abs(K4))], color="gray", ls=":", lw=0.5,
               label=f"$r(K_4=0)={r1[np.argmin(np.abs(K4))]:.3f}$")

    # 标注变化幅度
    r_range = r1.max() - r1.min()
    r_at_zero = r1[np.argmin(np.abs(K4))]
    pct = r_range / r_at_zero * 100

    ax.annotate(f"$\\Delta r / r_0 = {pct:.1f}\\%$",
                xy=(K4[-1], r1[-1]), xytext=(K4[-1] * 0.6, r1.min() + 0.002),
                arrowprops=dict(arrowstyle="->", color="0.4"),
                fontsize=10, color="0.3")

    ax.set_xlabel("$K_4$ (four-body coupling)")
    ax.set_ylabel("Order parameter $r$")
    ax.set_title(f"Four-body coupling effect ($K_2={K2}$, $K_3={K3}$, $\\sigma={sigma}$)")
    ax.legend()
    ax.grid(True, alpha=0.15)

    path = os.path.join(FIG_DIR, "K4_effect.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_locking_order()
    plot_K4_effect()
