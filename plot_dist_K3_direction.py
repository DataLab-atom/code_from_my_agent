"""
频率分布 × K₃ 方向效应对比
核心发现：所有分布在 N=200 下 K₃>0 都帮助同步，但 OA(N→∞) 说抑制
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "figure.dpi": 200,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.0, "lines.linewidth": 2,
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


def plot():
    os.makedirs(FIG_DIR, exist_ok=True)

    with open("freq_distribution.json") as f:
        d = json.load(f)

    results = d["results"]
    sigma = d["sigma"]
    Delta = sigma * np.sqrt(2 * np.pi) / np.pi

    # Organize by (dist_type, K3)
    data = {}
    for r in results:
        key = r["dist_type"]
        if key not in data:
            data[key] = {}
        data[key][r["K3"]] = r

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.25})

    dist_colors = {"gaussian": "#2196F3", "lorentzian": "#E53935", "uniform": "#4CAF50"}
    K3_values = [-1.0, 0.0, 1.0]
    K3_styles = {-1.0: ":", 0.0: "-", 1.0: "--"}

    # Panel A: r vs K2 for each distribution at K3=0 and K3=+1
    ax = axes[0]
    for dist, color in dist_colors.items():
        if dist not in data:
            continue
        for K3 in [0.0, 1.0]:
            if K3 not in data[dist]:
                continue
            r = data[dist][K3]
            style = "-" if K3 == 0 else "--"
            label = f"{dist.capitalize()}, $K_3={K3:+.0f}$"
            ax.plot(r["K2_list"], r["r1"], style, color=color, lw=1.8, label=label)

    # OA prediction
    K2_fine = np.linspace(0, 6, 100)
    for K3, ls in [(0, "-"), (1, "--")]:
        r_oa = [oa_r_star(k2, K3, Delta) for k2 in K2_fine]
        ax.plot(K2_fine, r_oa, ls, color="gray", lw=1, alpha=0.5,
                label=f"OA, $K_3={K3}$" if K3 == 0 else None)

    ax.set_xlabel("$K_2$")
    ax.set_ylabel("$r$")
    ax.set_title("$r$ vs $K_2$ ($K_3=0$ solid, $K_3=+1$ dashed)")
    ax.legend(fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.text(-0.12, 1.06, "A", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top")

    # Panel B: K3 effect (delta r from K3=0) at K2=2.0
    ax2 = axes[1]
    K2_target = 2.0
    bar_x = np.arange(len(dist_colors))
    width = 0.35

    delta_r_pos = []
    delta_r_neg = []
    dist_names = []
    for dist in ["gaussian", "lorentzian", "uniform"]:
        if dist not in data:
            continue
        dist_names.append(dist.capitalize())
        r_zero = None
        r_pos = None
        r_neg = None
        for K3 in [0.0, 1.0, -1.0]:
            if K3 in data[dist]:
                r = data[dist][K3]
                K2_idx = min(range(len(r["K2_list"])),
                             key=lambda i: abs(r["K2_list"][i] - K2_target))
                if K3 == 0:
                    r_zero = r["r1"][K2_idx]
                elif K3 == 1:
                    r_pos = r["r1"][K2_idx]
                elif K3 == -1:
                    r_neg = r["r1"][K2_idx]
        delta_r_pos.append(r_pos - r_zero if r_pos and r_zero else 0)
        delta_r_neg.append(r_neg - r_zero if r_neg and r_zero else 0)

    # OA prediction
    r_oa_0 = oa_r_star(K2_target, 0, Delta)
    r_oa_p = oa_r_star(K2_target, 1, Delta)
    r_oa_n = oa_r_star(K2_target, -1, Delta)

    ax2.bar(bar_x - width / 2, delta_r_pos, width, color="#4CAF50", alpha=0.8,
            label="$K_3 = +1$")
    ax2.bar(bar_x + width / 2, delta_r_neg, width, color="#F44336", alpha=0.8,
            label="$K_3 = -1$")
    ax2.axhline(r_oa_p - r_oa_0, color="gray", ls="--", lw=1,
                label=f"OA: $K_3=+1$ ({r_oa_p - r_oa_0:+.3f})")
    ax2.axhline(0, color="k", lw=0.5)

    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(dist_names)
    ax2.set_ylabel("$\\Delta r = r(K_3) - r(K_3=0)$")
    ax2.set_title(f"$K_3$ effect at $K_2 = {K2_target}$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15, axis="y")
    ax2.text(-0.12, 1.06, "B", transform=ax2.transAxes,
             fontsize=15, fontweight="bold", va="top")

    # Panel C: Kc shift by K3
    ax3 = axes[2]
    for dist, color in dist_colors.items():
        if dist not in data:
            continue
        K3_vals = sorted(data[dist].keys())
        Kc_vals = [data[dist][k3]["Kc_measured"] for k3 in K3_vals]
        ax3.plot(K3_vals, Kc_vals, "o-", color=color, ms=6, lw=1.8,
                 label=dist.capitalize())

    ax3.axhline(2 * Delta, color="gray", ls="--", lw=1,
                label=f"OA: $K_c = {2*Delta:.3f}$")
    ax3.set_xlabel("$K_3$")
    ax3.set_ylabel("$K_c$ (measured)")
    ax3.set_title("$K_c$ shift by $K_3$ and distribution")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.15)
    ax3.text(-0.12, 1.06, "C", transform=ax3.transAxes,
             fontsize=15, fontweight="bold", va="top")

    fig.suptitle(
        "$K_3 > 0$ helps synchronization for ALL distributions at $N=200$\n"
        "(OA predicts the opposite — finite-size effect flips the direction)",
        fontsize=13, y=1.05)

    path = os.path.join(FIG_DIR, "dist_K3_direction.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot()
