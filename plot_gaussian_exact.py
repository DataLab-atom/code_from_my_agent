"""
Gaussian 精确自洽 vs Lorentzian OA 对比图
数据: gaussian_exact_results.json (math/ott-antonsen)
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


def plot_gaussian_exact(data_path="gaussian_exact_results.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    K2 = np.array(data["K2_range"])
    K3 = np.array(data["K3_range"])
    r_gauss = np.array(data["r_gaussian_2d"])
    sigma = data["sigma"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"wspace": 0.3})

    # Panel A: Gaussian exact phase diagram
    ax = axes[0]
    K3g, K2g = np.meshgrid(K3, K2)
    im = ax.pcolormesh(K3g, K2g, r_gauss, cmap="RdYlGn", vmin=0, vmax=1,
                       shading="auto", rasterized=True)
    ax.contour(K3g, K2g, r_gauss, levels=[0.1, 0.3, 0.5],
               colors=["0.3"], linewidths=[0.5, 0.8, 1.2],
               linestyles=[":", "--", "-"])
    # explosive boundary
    k2_line = np.linspace(max(K2.min(), K3.min()), min(K2.max(), K3.max()), 50)
    ax.plot(k2_line, k2_line, "w--", lw=1, alpha=0.7)
    fig.colorbar(im, ax=ax, label="$r^*$", shrink=0.9)
    ax.set_xlabel("$K_3$")
    ax.set_ylabel("$K_2$")
    ax.set_title(f"Gaussian exact ($\\sigma={sigma}$)")
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Panel B & C: Gaussian vs OA comparison for different K₃
    comparisons = [
        (data["comparison_K3_0"], "$K_3=0$"),
        (data["comparison_K3_1"], "$K_3=1$"),
    ]
    colors_pair = [("#2196F3", "#FF9800"), ("#4CAF50", "#E53935")]

    for pi, (comp, label) in enumerate(comparisons):
        ax2 = axes[pi + 1]
        K2_c = [e["K2"] for e in comp]
        r_g = [e["r_gaussian"] for e in comp]
        r_o = [e["r_OA"] for e in comp]
        diff = [e["diff"] for e in comp]

        c1, c2 = colors_pair[pi]
        ax2.plot(K2_c, r_g, "o-", color=c1, ms=4, lw=1.5, label="Gaussian exact")
        ax2.plot(K2_c, r_o, "s--", color=c2, ms=4, lw=1.5, label="Lorentzian OA")

        # error shading
        ax2.fill_between(K2_c, r_g, r_o, alpha=0.15, color="gray")

        ax2.set_xlabel("$K_2$")
        ax2.set_ylabel("$r^*$")
        ax2.set_title(f"Gaussian vs OA ({label})")
        ax2.legend(fontsize=8, framealpha=0.9)
        ax2.grid(True, alpha=0.15)
        ax2.text(-0.12, 1.06, chr(66 + pi), transform=ax2.transAxes,
                 fontsize=13, fontweight="bold", va="top")

    fig.suptitle("Gaussian exact self-consistent equation vs. Ott-Antonsen (Lorentzian)",
                 fontsize=12, y=1.02)

    path = os.path.join(FIG_DIR, "gaussian_exact.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_gaussian_exact()
