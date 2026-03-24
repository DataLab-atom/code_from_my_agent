"""
核心数值相图 — K₂×K₃ 三指标热力图 + 解析叠加
数据: scan_K2_K3.json (sim/parameter-scan) + analytical_phase_diagram.json (math/ott-antonsen)
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


def oa_delta(sigma):
    return sigma * np.sqrt(2 * np.pi) / np.pi


def plot_numerical_phase(scan_path="scan_K2_K3.json",
                          analytic_path="analytical_phase_diagram.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(scan_path) as f:
        d = json.load(f)

    K2 = np.array(d["K2_list"])
    K3 = np.array(d["K3_list"])
    r = np.array(d["r"])
    tc = np.array(d["tc"])
    basin = np.array(d["basin_prob"])
    sigma = d["sigma"]
    N = d["N"]
    Kc = 2 * oa_delta(sigma)

    K3g, K2g = np.meshgrid(K3, K2)

    # Load analytic for overlay
    analytic_r = None
    analytic_K2 = None
    analytic_K3 = None
    try:
        with open(analytic_path) as f:
            ad = json.load(f)
        sk = f"sigma={sigma}"
        if sk in ad["sigma_results"]:
            analytic_r = np.array(ad["sigma_results"][sk]["r"])
            analytic_K2 = np.array(ad["K2_range"])
            analytic_K3 = np.array(ad["K3_range"])
    except:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"wspace": 0.25})

    # --- Panel A: Order parameter r ---
    ax = axes[0]
    im = ax.pcolormesh(K3g, K2g, r, cmap="RdYlGn", vmin=0, vmax=1,
                       shading="auto", rasterized=True)
    cs = ax.contour(K3g, K2g, r, levels=[0.3, 0.5, 0.7],
                    colors=["0.2"], linewidths=[0.6, 1.2, 0.6],
                    linestyles=["--", "-", "--"])
    ax.clabel(cs, fmt="%.1f", fontsize=7)

    # Analytic r=0.5 contour overlay
    if analytic_r is not None:
        aK3g, aK2g = np.meshgrid(analytic_K3, analytic_K2)
        ax.contour(aK3g, aK2g, analytic_r, levels=[0.5],
                   colors=["cyan"], linewidths=[1.5], linestyles=["--"])

    # OA onset + explosive
    ax.axhline(Kc, color="white", ls=":", lw=1, alpha=0.8)
    ax.text(K3.max() * 0.8, Kc + 0.15, f"$K_c$={Kc:.2f}", color="white", fontsize=8, ha="right")
    k2_line = np.linspace(max(K2.min(), K3.min()), min(K2.max(), K3.max()), 50)
    ax.plot(k2_line, k2_line, "w--", lw=1, alpha=0.7)

    fig.colorbar(im, ax=ax, label="$r$", shrink=0.9)
    ax.set_xlabel("$K_3$")
    ax.set_ylabel("$K_2$")
    ax.set_title("Order parameter $r$")
    ax.text(-0.15, 1.06, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # --- Panel B: Basin probability ---
    ax2 = axes[1]
    im2 = ax2.pcolormesh(K3g, K2g, basin, cmap="YlGnBu", vmin=0, vmax=1,
                          shading="auto", rasterized=True)
    cs2 = ax2.contour(K3g, K2g, basin, levels=[0.3, 0.5, 0.7],
                      colors=["0.3"], linewidths=[0.6, 1.2, 0.6],
                      linestyles=["--", "-", "--"])
    ax2.clabel(cs2, fmt="%.1f", fontsize=7)
    ax2.axhline(Kc, color="white", ls=":", lw=1, alpha=0.6)
    ax2.plot(k2_line, k2_line, "w--", lw=1, alpha=0.5)
    fig.colorbar(im2, ax=ax2, label="Basin prob.", shrink=0.9)
    ax2.set_xlabel("$K_3$")
    ax2.set_ylabel("$K_2$")
    ax2.set_title("Basin probability")
    ax2.text(-0.15, 1.06, "B", transform=ax2.transAxes,
             fontsize=14, fontweight="bold", va="top")

    # --- Panel C: Convergence time ---
    ax3 = axes[2]
    tc_plot = np.where(tc > 0, tc, np.nan)
    vmax_tc = np.nanpercentile(tc_plot, 90)
    im3 = ax3.pcolormesh(K3g, K2g, tc_plot, cmap="YlOrRd_r", vmin=0, vmax=vmax_tc,
                          shading="auto", rasterized=True)
    ax3.axhline(Kc, color="white", ls=":", lw=1, alpha=0.6)
    ax3.plot(k2_line, k2_line, "w--", lw=1, alpha=0.5)
    fig.colorbar(im3, ax=ax3, label="Conv. time", shrink=0.9)
    ax3.set_xlabel("$K_3$")
    ax3.set_ylabel("$K_2$")
    ax3.set_title("Convergence time")
    ax3.text(-0.15, 1.06, "C", transform=ax3.transAxes,
             fontsize=14, fontweight="bold", va="top")

    fig.suptitle(
        f"Numerical $K_2 \\times K_3$ phase diagram ($\\sigma={sigma}$, $N={N}$)\n"
        f"Cyan dashed: OA analytic $r=0.5$; White dotted: $K_c={Kc:.2f}$; White dashed: $K_3=K_2$",
        fontsize=11, y=1.06)

    path = os.path.join(FIG_DIR, "numerical_phase_K2K3.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


def plot_freq_distribution(data_path="freq_distribution.json"):
    """频率分布对比: Gaussian vs Lorentzian vs Uniform"""
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"gaussian": "#2196F3", "lorentzian": "#E53935", "uniform": "#4CAF50"}
    markers = {"gaussian": "o", "lorentzian": "s", "uniform": "D"}

    for dist_name, result in data.items():
        if not isinstance(result, dict) or "K2_list" not in result:
            continue
        K2 = result["K2_list"]
        r = result["r"]
        c = colors.get(dist_name, "gray")
        m = markers.get(dist_name, "o")
        ax.plot(K2, r, f"{m}-", color=c, ms=4, lw=1.5,
                label=dist_name.capitalize())

    ax.set_xlabel("$K_2$")
    ax.set_ylabel("Order parameter $r$")
    ax.set_title("Frequency distribution effect on synchronization ($K_3=0$)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    path = os.path.join(FIG_DIR, "freq_distribution.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_numerical_phase()
    try:
        plot_freq_distribution()
    except Exception as e:
        print(f"freq_distribution skipped: {e}")
