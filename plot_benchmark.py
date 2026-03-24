"""
经典 Kuramoto Kc 基准验证图
利用 GPU-Claude-Opus 的 benchmark_classical.json 数据
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
})

FIG_DIR = "figures"


def plot_classical_benchmark(data_path="benchmark_classical.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    K2 = np.array(data["K2_list"])
    r = np.array(data["r_list"])
    sigma = data.get("sigma", 1.0)
    N = data.get("N", 200)

    # 理论值: Kc = 2σ√(2π)/π ≈ 1.596σ
    Kc = 2.0 * sigma * np.sqrt(2 * np.pi) / np.pi

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.3})

    # Panel A: r vs K₂
    ax = axes[0]
    ax.plot(K2, r, "o-", color="#2196F3", ms=5, lw=1.5, label="Simulation")
    ax.axvline(Kc, color="red", ls="--", lw=1.5, label=f"$K_c = {Kc:.3f}$")
    ax.set_xlabel("$K_2$")
    ax.set_ylabel("Order parameter $r$")
    ax.set_title(f"Classical Kuramoto ($\\sigma={sigma}$, $N={N}$, $K_3=0$)")
    ax.legend()
    ax.grid(True, alpha=0.15)
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Panel B: r vs √(K₂ - Kc) — 检验 r ~ √(K-Kc) 标度律
    ax2 = axes[1]
    mask = K2 > Kc * 1.05  # 临界点以上
    if mask.any():
        x = np.sqrt(K2[mask] - Kc)
        y = r[mask]
        ax2.plot(x, y, "o", color="#4CAF50", ms=5)

        # 线性拟合
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(0, x.max(), 100)
        ax2.plot(x_fit, np.polyval(coeffs, x_fit), "r--", lw=1.5,
                 label=f"Linear fit: $r \\approx {coeffs[0]:.2f}\\sqrt{{K_2-K_c}}$")

    ax2.set_xlabel("$\\sqrt{K_2 - K_c}$")
    ax2.set_ylabel("Order parameter $r$")
    ax2.set_title("Scaling law verification: $r \\sim \\sqrt{K_2 - K_c}$")
    ax2.legend()
    ax2.grid(True, alpha=0.15)
    ax2.text(-0.12, 1.06, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")

    path = os.path.join(FIG_DIR, "benchmark_classical.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_classical_benchmark()
