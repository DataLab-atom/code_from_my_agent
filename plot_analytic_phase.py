"""
解析相图可视化 — 基于 MathAgent 的 Ott-Antonsen 推导
数据来源: analytical_phase_diagram.json (math/ott-antonsen 分支)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
})

FIG_DIR = "figures"


def load_analytic_data(path="analytical_phase_diagram.json"):
    with open(path) as f:
        data = json.load(f)
    return data


def plot_analytic_phase_diagrams(data, save=True):
    """
    对每个 σ 画解析 r*(K₂,K₃) 热力图
    叠加 onset 线、explosive 线、多稳态区域、时延可达曲线
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    K2 = np.array(data["K2_range"])
    K3 = np.array(data["K3_range"])
    sigma_results = data["sigma_results"]
    onset_K2c = data["onset_K2c"]
    nfp_sigma1 = np.array(data.get("n_fixed_points_sigma1", []))

    sigma_keys = sorted(sigma_results.keys(), key=lambda s: float(s.split("=")[1]))
    nS = len(sigma_keys)

    ncols = min(3, nS)
    nrows = (nS + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    if nS == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    K3g, K2g = np.meshgrid(K3, K2)

    for i, sk in enumerate(sigma_keys):
        ax = axes[i]
        sr = sigma_results[sk]
        sigma = float(sk.split("=")[1])
        r_star = np.array(sr["r"])

        im = ax.pcolormesh(K3g, K2g, r_star, cmap="RdYlGn", vmin=0, vmax=1,
                           shading="auto", rasterized=True)

        # 相边界等值线
        ax.contour(K3g, K2g, r_star, levels=[0.1, 0.3, 0.5, 0.7],
                   colors=["0.3"], linewidths=[0.5, 0.6, 1.0, 0.6],
                   linestyles=[":", "--", "-", "--"])

        # onset 线
        Kc = onset_K2c.get(sk, sr.get("Kc", 2 * sigma * np.sqrt(2 * np.pi) / np.pi))
        if Kc <= K2.max():
            ax.axhline(Kc, color="white", ls=":", lw=1, alpha=0.8)
            ax.text(K3.max() * 0.85, Kc + 0.15, f"$K_c$={Kc:.2f}",
                    color="white", fontsize=7, ha="right")

        # explosive 线 K₃ = K₂
        k2_line = np.linspace(max(K2.min(), K3.min()), min(K2.max(), K3.max()), 50)
        ax.plot(k2_line, k2_line, "w--", lw=1, alpha=0.7)

        # 多稳态区域 (仅 σ=1)
        if sk == "sigma=1.0" and nfp_sigma1.size > 0:
            ax.contour(K3g, K2g, nfp_sigma1, levels=[1.5],
                       colors=["yellow"], linewidths=[1.2], linestyles=["-"])

        # 时延可达曲线
        for dk, dv in data.get("delay_reachable_curves", {}).items():
            K2_bare = float(dk.split("=")[1])
            if isinstance(dv, dict) and "K2_eff" in dv and "K3_eff" in dv:
                ax.plot(dv["K3_eff"], dv["K2_eff"], "c-", lw=0.8, alpha=0.5)

        ax.set_xlabel("$K_3$")
        ax.set_ylabel("$K_2$")
        ax.set_title(f"$\\sigma={sigma:.1f}$ (analytic)")
        ax.text(-0.12, 1.06, chr(65 + i), transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    # colorbar
    fig.colorbar(im, ax=axes[:nS].tolist(), label="$r^*$ (Ott-Antonsen)",
                 shrink=0.8, pad=0.02)

    for j in range(nS, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Analytic phase diagrams (Ott-Antonsen reduction)",
                 fontsize=14, y=1.02)

    if save:
        path = os.path.join(FIG_DIR, "analytic_phase_diagrams.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    data = load_analytic_data()
    plot_analytic_phase_diagrams(data)
