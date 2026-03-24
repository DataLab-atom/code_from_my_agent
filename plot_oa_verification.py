"""
OA 精度验证图 — 解析 Kc vs 数值 Kc
比较两个 Δ 近似:
  (旧) Δ = σ√(π/2)       → Kc = 2σ√(π/2) ≈ 2.507σ  (过高)
  (新) Δ = σ√(2π)/π       → Kc = 2σ√(2π)/π ≈ 1.596σ  (MathAgent 修正)
  数值                    → Kc ≈ 1.72σ (GPU 结果)
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


def plot_oa_verification(data_path="verify_oa_results.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    entries = data["Kc_vs_sigma"]
    sigmas = [e["sigma"] for e in entries]
    Kc_oa_old = [e["Kc_OA"] for e in entries]  # Δ=σ√(π/2)
    Kc_num = [e["Kc_num"] for e in entries]

    sigmas = np.array(sigmas)
    # 新 OA: Δ = σ√(2π)/π
    Kc_oa_new = 2 * sigmas * np.sqrt(2 * np.pi) / np.pi
    # 经典公式: Kc = 2√(2π)σ (这是 kuramoto_problem.md 里写的)
    Kc_classic = 2 * np.sqrt(2 * np.pi) * sigmas

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.3})

    # Panel A: Kc vs σ
    ax = axes[0]
    s_fine = np.linspace(0.1, 2.2, 100)
    ax.plot(s_fine, 2 * s_fine * np.sqrt(np.pi / 2), "--",
            color="#FF9800", lw=1.5, label="OA old: $2\\sigma\\sqrt{\\pi/2}$")
    ax.plot(s_fine, 2 * s_fine * np.sqrt(2 * np.pi) / np.pi, "--",
            color="#2196F3", lw=1.5, label="OA new: $2\\sigma\\sqrt{2\\pi}/\\pi$")
    ax.plot(s_fine, 2 * np.sqrt(2 * np.pi) * s_fine, ":",
            color="gray", lw=1, label="$2\\sqrt{2\\pi}\\sigma$ (problem.md)")
    ax.plot(sigmas, Kc_num, "ko-", ms=7, lw=2, label="Numerical $K_c$", zorder=10)

    ax.set_xlabel("$\\sigma$")
    ax.set_ylabel("$K_c$")
    ax.set_title("Critical coupling: analytic vs. numerical")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Panel B: 相对误差
    ax2 = axes[1]
    err_old = (np.array(Kc_oa_old) - np.array(Kc_num)) / np.array(Kc_num) * 100
    err_new = (Kc_oa_new - np.array(Kc_num)) / np.array(Kc_num) * 100

    ax2.bar(np.arange(len(sigmas)) - 0.15, err_old, 0.3,
            color="#FF9800", alpha=0.8, label="OA old")
    ax2.bar(np.arange(len(sigmas)) + 0.15, err_new, 0.3,
            color="#2196F3", alpha=0.8, label="OA new")
    ax2.set_xticks(range(len(sigmas)))
    ax2.set_xticklabels([f"{s:.1f}" for s in sigmas])
    ax2.set_xlabel("$\\sigma$")
    ax2.set_ylabel("Relative error (%)")
    ax2.set_title("$K_c$ prediction error")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15, axis="y")
    ax2.text(-0.12, 1.06, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")

    path = os.path.join(FIG_DIR, "oa_verification.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_oa_verification()
