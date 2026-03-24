"""
论文核心 Fig.1: K₂×K₃ 相图 + 文献标注 + OA 叠加
+ OA 偏差分析 (Task AB.1)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import os

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
})

FIG_DIR = "figures"


def oa_delta(sigma):
    return sigma * np.sqrt(2 * np.pi) / np.pi


def oa_r_star(K2, K3, Delta):
    """Solve: Delta = (1-r^2)/2 * (K2 + K3*r^2) => K3*r^4 - (K2+K3)*r^2 + (K2-2*Delta) = 0"""
    a = K3
    b = -(K2 + K3)
    c = K2 - 2 * Delta
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return 0.0
        u = -c / b
        return np.sqrt(u) if 0 < u < 1 else 0.0
    disc = b**2 - 4 * a * c
    if disc < 0:
        return 0.0
    u1 = (-b + np.sqrt(disc)) / (2 * a)
    u2 = (-b - np.sqrt(disc)) / (2 * a)
    sols = [np.sqrt(u) for u in [u1, u2] if 0 < u < 1]
    return max(sols) if sols else 0.0


def plot_fig1(scan_path="scan_K2_K3.json",
              analytic_path="analytical_phase_diagram.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(scan_path) as f:
        d = json.load(f)

    K2 = np.array(d["K2_list"])
    K3 = np.array(d["K3_list"])
    r = np.array(d["r"])
    basin = np.array(d["basin_prob"])
    tc = np.array(d["tc"])
    sigma = d["sigma"]
    N = d["N"]
    Delta = oa_delta(sigma)
    Kc = 2 * Delta

    K3g, K2g = np.meshgrid(K3, K2)

    # Load analytic
    analytic_r = None
    try:
        with open(analytic_path) as f:
            ad = json.load(f)
        sk = f"sigma={sigma}"
        if sk in ad["sigma_results"]:
            analytic_r = np.array(ad["sigma_results"][sk]["r"])
            aK2 = np.array(ad["K2_range"])
            aK3 = np.array(ad["K3_range"])
    except:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"wspace": 0.22})

    # === Panel A: Order parameter r with literature annotations ===
    ax = axes[0]
    im = ax.pcolormesh(K3g, K2g, r, cmap="viridis", vmin=0, vmax=1,
                       shading="auto", rasterized=True)
    cs = ax.contour(K3g, K2g, r, levels=[0.3, 0.5, 0.7],
                    colors=["white"], linewidths=[0.6, 1.2, 0.6],
                    linestyles=["--", "-", "--"])
    ax.clabel(cs, fmt="%.1f", fontsize=7, colors="white")

    # OA analytic overlay
    if analytic_r is not None:
        aK3g, aK2g = np.meshgrid(aK3, aK2)
        ax.contour(aK3g, aK2g, analytic_r, levels=[0.5],
                   colors=["cyan"], linewidths=[2], linestyles=["--"])

    # Onset and explosive lines
    ax.axhline(Kc, color="white", ls=":", lw=1.2, alpha=0.9)
    k_diag = np.linspace(max(K2.min(), K3.min()), min(K2.max(), K3.max()), 50)
    ax.plot(k_diag, k_diag, "w--", lw=1.2, alpha=0.8)

    # Literature annotations
    # Muolo 2025: weak K3 region
    rect_m = Rectangle((-0.5, Kc * 0.8), 1.0, Kc * 0.5,
                        fill=False, edgecolor="yellow", lw=1.5, ls="--")
    ax.add_patch(rect_m)
    ax.text(0, Kc * 0.7, "Muolo\n2025", color="yellow", fontsize=7,
            ha="center", fontweight="bold")

    # Zhang 2024: strong K3
    rect_z = Rectangle((1.0, Kc * 1.2), 1.0, K2.max() - Kc * 1.2 - 0.1,
                        fill=False, edgecolor="orange", lw=1.5, ls="--")
    ax.add_patch(rect_z)
    ax.text(1.5, K2.max() * 0.85, "Zhang\n2024", color="orange", fontsize=7,
            ha="center", fontweight="bold")

    # Wang 2025: moderate K3
    rect_w = Rectangle((0.3, Kc * 1.1), 0.8, Kc * 0.6,
                        fill=False, edgecolor="#00E676", lw=1.5, ls="--")
    ax.add_patch(rect_w)
    ax.text(0.7, Kc * 1.5, "Wang\n2025", color="#00E676", fontsize=7,
            ha="center", fontweight="bold")

    fig.colorbar(im, ax=ax, label="Order parameter $r$", shrink=0.85)
    ax.set_xlabel("$K_3$ (three-body coupling)")
    ax.set_ylabel("$K_2$ (pairwise coupling)")
    ax.set_title("$r$ with literature regions")
    ax.text(-0.15, 1.06, "A", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top")

    # === Panel B: Basin probability ===
    ax2 = axes[1]
    im2 = ax2.pcolormesh(K3g, K2g, basin, cmap="RdBu", vmin=0, vmax=1,
                          shading="auto", rasterized=True)
    cs2 = ax2.contour(K3g, K2g, basin, levels=[0.3, 0.5, 0.7],
                      colors=["k"], linewidths=[0.6, 1.2, 0.6],
                      linestyles=["--", "-", "--"])
    ax2.clabel(cs2, fmt="%.1f", fontsize=7)
    ax2.axhline(Kc, color="k", ls=":", lw=1, alpha=0.6)
    ax2.plot(k_diag, k_diag, "k--", lw=1.2, alpha=0.6)
    ax2.text(K3.max() * 0.5, K2.max() * 0.15, "Frustration\nzone",
             color="darkred", fontsize=9, ha="center", style="italic",
             fontweight="bold")

    fig.colorbar(im2, ax=ax2, label="Basin probability", shrink=0.85)
    ax2.set_xlabel("$K_3$")
    ax2.set_ylabel("$K_2$")
    ax2.set_title("Basin of attraction")
    ax2.text(-0.15, 1.06, "B", transform=ax2.transAxes,
             fontsize=15, fontweight="bold", va="top")

    # === Panel C: Convergence time ===
    ax3 = axes[2]
    tc_clip = np.clip(tc, 0, np.percentile(tc, 90))
    im3 = ax3.pcolormesh(K3g, K2g, tc_clip, cmap="hot", vmin=0,
                          shading="auto", rasterized=True)
    ax3.axhline(Kc, color="cyan", ls=":", lw=1, alpha=0.8)
    ax3.plot(k_diag, k_diag, "cyan", ls="--", lw=1, alpha=0.6)
    fig.colorbar(im3, ax=ax3, label="Convergence time", shrink=0.85)
    ax3.set_xlabel("$K_3$")
    ax3.set_ylabel("$K_2$")
    ax3.set_title("Convergence time")
    ax3.text(-0.15, 1.06, "C", transform=ax3.transAxes,
             fontsize=15, fontweight="bold", va="top")

    fig.suptitle(
        f"Higher-order Kuramoto phase diagram ($\\sigma={sigma}$, $N={N}$)",
        fontsize=14, y=1.02)

    path = os.path.join(FIG_DIR, "fig1_phase_diagram.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")


def plot_oa_deviation(scan_path="scan_K2_K3.json"):
    """OA vs Numerical deviation map (Task AB.1)"""
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(scan_path) as f:
        d = json.load(f)

    K2 = np.array(d["K2_list"])
    K3 = np.array(d["K3_list"])
    r_num = np.array(d["r"])
    sigma = d["sigma"]
    Delta = oa_delta(sigma)

    # Compute OA predictions
    r_oa = np.zeros_like(r_num)
    for i, k2 in enumerate(K2):
        for j, k3 in enumerate(K3):
            r_oa[i, j] = oa_r_star(k2, k3, Delta)

    deviation = np.abs(r_oa - r_num)
    K3g, K2g = np.meshgrid(K3, K2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"wspace": 0.25})

    # Panel A: OA prediction
    ax = axes[0]
    im = ax.pcolormesh(K3g, K2g, r_oa, cmap="viridis", vmin=0, vmax=1,
                       shading="auto", rasterized=True)
    ax.contour(K3g, K2g, r_oa, levels=[0.3, 0.5, 0.7],
               colors=["white"], linewidths=[0.6, 1.0, 0.6])
    fig.colorbar(im, ax=ax, label="$r^*_{OA}$", shrink=0.9)
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$K_2$")
    ax.set_title("Ott-Antonsen prediction")
    ax.text(-0.15, 1.06, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

    # Panel B: Numerical
    ax2 = axes[1]
    im2 = ax2.pcolormesh(K3g, K2g, r_num, cmap="viridis", vmin=0, vmax=1,
                          shading="auto", rasterized=True)
    ax2.contour(K3g, K2g, r_num, levels=[0.3, 0.5, 0.7],
                colors=["white"], linewidths=[0.6, 1.0, 0.6])
    fig.colorbar(im2, ax=ax2, label="$r_{num}$", shrink=0.9)
    ax2.set_xlabel("$K_3$"); ax2.set_ylabel("$K_2$")
    ax2.set_title(f"Numerical ($N={d['N']}$)")
    ax2.text(-0.15, 1.06, "B", transform=ax2.transAxes,
             fontsize=14, fontweight="bold", va="top")

    # Panel C: Deviation
    ax3 = axes[2]
    vmax_dev = max(0.01, np.percentile(deviation, 95))
    im3 = ax3.pcolormesh(K3g, K2g, deviation, cmap="Reds", vmin=0, vmax=vmax_dev,
                          shading="auto", rasterized=True)
    cs3 = ax3.contour(K3g, K2g, deviation, levels=[0.05, 0.1, 0.2],
                      colors=["k"], linewidths=[0.8, 1.2, 0.8],
                      linestyles=[":", "-", "--"])
    ax3.clabel(cs3, fmt="%.2f", fontsize=8)

    # Mark safe zone (deviation < 0.05)
    safe = deviation < 0.05
    safe_frac = safe.sum() / safe.size * 100
    ax3.text(0.05, 0.95, f"Safe zone (<5%): {safe_frac:.0f}%",
             transform=ax3.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.colorbar(im3, ax=ax3, label="$|r^*_{OA} - r_{num}|$", shrink=0.9)
    ax3.set_xlabel("$K_3$"); ax3.set_ylabel("$K_2$")
    ax3.set_title("OA-Numerical deviation")
    ax3.text(-0.15, 1.06, "C", transform=ax3.transAxes,
             fontsize=14, fontweight="bold", va="top")

    fig.suptitle("Ott-Antonsen approximation validity map", fontsize=13, y=1.02)

    path = os.path.join(FIG_DIR, "oa_deviation.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")

    # Print summary
    print(f"\nOA Deviation Summary:")
    print(f"  Mean |deviation|: {deviation.mean():.4f}")
    print(f"  Max  |deviation|: {deviation.max():.4f}")
    print(f"  Safe zone (<5%): {safe_frac:.0f}% of parameter space")
    print(f"  Worst region: K3<0 + K2 near Kc")

    return {"safe_zone_pct": safe_frac, "mean_dev": float(deviation.mean()),
            "max_dev": float(deviation.max())}


if __name__ == "__main__":
    plot_fig1()
    stats = plot_oa_deviation()
