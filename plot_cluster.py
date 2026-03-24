"""
集群结构分析图 — K₃ 对振荡器聚类的影响
数据: cluster_structure.json (sim/parameter-scan)
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


def plot_cluster_structure(data_path="cluster_structure.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    results = data["results"]
    K3_vals = [r["K3"] for r in results]
    K2_vals = [r["K2"] for r in results]
    n_clusters = [r["n_clusters"] for r in results]
    r1_vals = [r["r1"] for r in results]
    r2_vals = [r["r2"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"wspace": 0.3})

    # Panel A: n_clusters vs K₃
    ax = axes[0]
    ax.plot(K3_vals, n_clusters, "o-", color="#9C27B0", ms=5)
    ax.set_xlabel("$K_3$")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster count vs $K_3$")
    ax.grid(True, alpha=0.15)
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Panel B: r₁ and r₂ vs K₃
    ax2 = axes[1]
    ax2.plot(K3_vals, r1_vals, "o-", color="#2196F3", ms=5, label="$r_1$")
    ax2.plot(K3_vals, r2_vals, "s-", color="#FF9800", ms=5, label="$r_2$")
    ax2.set_xlabel("$K_3$")
    ax2.set_ylabel("Order parameter")
    ax2.set_title("$r_1$ and $r_2$ vs $K_3$")
    ax2.legend()
    ax2.grid(True, alpha=0.15)
    ax2.text(-0.12, 1.06, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")

    # Panel C: r₂/r₁ ratio — sensitivity indicator
    ax3 = axes[2]
    r1_arr = np.array(r1_vals)
    r2_arr = np.array(r2_vals)
    ratio = np.where(r1_arr > 0.05, r2_arr / r1_arr, np.nan)
    ax3.plot(K3_vals, ratio, "D-", color="#4CAF50", ms=5)
    ax3.axhline(1, color="gray", ls="--", lw=0.6)
    ax3.set_xlabel("$K_3$")
    ax3.set_ylabel("$r_2 / r_1$")
    ax3.set_title("Second-order sensitivity")
    ax3.grid(True, alpha=0.15)
    ax3.text(-0.12, 1.06, "C", transform=ax3.transAxes,
             fontsize=13, fontweight="bold", va="top")

    fig.suptitle(f"Cluster structure analysis ($N={data['N']}$, $\\sigma={data['sigma']}$)",
                 fontsize=13, y=1.02)

    path = os.path.join(FIG_DIR, "cluster_structure.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_cluster_structure()
