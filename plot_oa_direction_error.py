"""
关键发现：Lorentzian OA 给出错误的 K₃ 效应方向
Gaussian exact 与数值一致，Lorentzian OA 方向相反
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 13,
    "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 1.0,
    "lines.linewidth": 2,
})

FIG_DIR = "figures"


def oa_r_star(K2, K3, Delta):
    a, b, c = K3, -(K2 + K3), K2 - 2 * Delta
    if abs(a) < 1e-12:
        u = -c / b if abs(b) > 1e-12 else 0
        return np.sqrt(u) if 0 < u < 1 else 0
    disc = b**2 - 4 * a * c
    if disc < 0: return 0
    sols = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    rs = [np.sqrt(u) for u in sols if 0 < u < 1]
    return max(rs) if rs else 0


def plot_direction_error():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load numerical
    with open("scan_K2_K3.json") as f:
        d_num = json.load(f)
    K2_num = np.array(d_num["K2_list"])
    K3_num = np.array(d_num["K3_list"])
    r_num = np.array(d_num["r"])

    # Load Gaussian exact
    with open("gaussian_exact_results.json") as f:
        d_gauss = json.load(f)
    K2_gauss = np.array(d_gauss["K2_range"])
    K3_gauss = np.array(d_gauss["K3_range"])
    r_gauss = np.array(d_gauss["r_gaussian_2d"])

    sigma = 1.0
    Delta = sigma * np.sqrt(2 * np.pi) / np.pi

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": 0.3})

    # Panel A: r* vs K₃ at fixed K₂ ≈ 2.1
    ax = axes[0]
    K2_target = 2.1

    # Numerical
    ki_num = np.argmin(np.abs(K2_num - K2_target))
    ax.plot(K3_num, r_num[ki_num, :], "ko-", ms=5, lw=2,
            label=f"Numerical ($N=200$)", zorder=10)

    # Gaussian exact
    ki_gauss = np.argmin(np.abs(K2_gauss - K2_target))
    K3_gauss_plot = K3_gauss[K3_gauss <= K3_num.max() + 0.5]
    r_gauss_plot = r_gauss[ki_gauss, :len(K3_gauss_plot)]
    ax.plot(K3_gauss_plot, r_gauss_plot, "s--", color="#4CAF50", ms=5, lw=2,
            label="Gaussian exact ($N \\to \\infty$)")

    # Lorentzian OA
    K3_fine = np.linspace(K3_num.min(), K3_num.max(), 50)
    r_oa = [oa_r_star(K2_num[ki_num], k3, Delta) for k3 in K3_fine]
    ax.plot(K3_fine, r_oa, "--", color="#E53935", lw=2.5,
            label="Lorentzian OA")

    ax.axvline(0, color="gray", ls=":", lw=0.5)
    ax.axhline(0.5, color="gray", ls=":", lw=0.5)

    # Arrows showing wrong direction
    ax.annotate("", xy=(1.5, 0.38), xytext=(1.5, 0.50),
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=2.5))
    ax.text(1.6, 0.44, "OA says\n$K_3$ hurts", color="#E53935", fontsize=9)

    ax.annotate("", xy=(1.5, 0.94), xytext=(1.5, 0.80),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    ax.text(0.6, 0.88, "Reality:\n$K_3$ helps!", color="black", fontsize=9,
            fontweight="bold")

    ax.set_xlabel("$K_3$")
    ax.set_ylabel("$r^*$")
    ax.set_title(f"$r^*$ vs $K_3$ at $K_2 \\approx {K2_target}$")
    ax.legend(fontsize=9, framealpha=0.95, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15)
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top")

    # Panel B: Direction match map
    ax2 = axes[1]
    K3g, K2g = np.meshgrid(K3_num, K2_num)
    K3_zero_idx = np.argmin(np.abs(K3_num))

    direction_match = np.zeros_like(r_num)
    for i in range(len(K2_num)):
        r_oa_0 = oa_r_star(K2_num[i], 0, Delta)
        r_num_0 = r_num[i, K3_zero_idx]
        for j in range(len(K3_num)):
            r_oa_j = oa_r_star(K2_num[i], K3_num[j], Delta)
            oa_dir = np.sign(r_oa_j - r_oa_0)
            num_dir = np.sign(r_num[i, j] - r_num_0)
            if abs(K3_num[j]) < 0.2:
                direction_match[i, j] = 0  # neutral
            elif oa_dir == num_dir:
                direction_match[i, j] = 1  # match
            else:
                direction_match[i, j] = -1  # mismatch

    im = ax2.pcolormesh(K3g, K2g, direction_match, cmap="RdYlGn",
                         vmin=-1, vmax=1, shading="auto", rasterized=True)
    ax2.axhline(2 * Delta, color="white", ls=":", lw=1)
    k_diag = np.linspace(max(K2_num.min(), K3_num.min()),
                          min(K2_num.max(), K3_num.max()), 50)
    ax2.plot(k_diag, k_diag, "w--", lw=1, alpha=0.7)

    cb = fig.colorbar(im, ax=ax2, ticks=[-1, 0, 1], shrink=0.9)
    cb.set_ticklabels(["Opposite", "Neutral", "Agree"])

    # Count
    agree = (direction_match == 1).sum()
    oppose = (direction_match == -1).sum()
    total = agree + oppose
    ax2.text(0.05, 0.95,
             f"Direction agreement: {agree}/{total} = {agree/total*100:.0f}%",
             transform=ax2.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax2.set_xlabel("$K_3$")
    ax2.set_ylabel("$K_2$")
    ax2.set_title("OA vs Numerical: direction agreement map")
    ax2.text(-0.12, 1.06, "B", transform=ax2.transAxes,
             fontsize=15, fontweight="bold", va="top")

    fig.suptitle(
        "Lorentzian OA predicts wrong $K_3$ effect direction for Gaussian distribution",
        fontsize=13, y=1.02)

    path = os.path.join(FIG_DIR, "oa_direction_error.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_direction_error()
