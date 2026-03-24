"""
耦合预算约束下的最优分配 — Pareto 前沿
数据: budget_optimization.json (math/ott-antonsen)
K₂ + K₃ = C (固定预算), 求最大 r*
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


def plot_budget_optimization(data_path="budget_optimization.json"):
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    sigmas = data["sigmas"]
    budgets = data["budgets"]
    results = data["results"]

    fig, axes = plt.subplots(1, len(sigmas), figsize=(5.5 * len(sigmas), 5),
                              sharey=True, gridspec_kw={"wspace": 0.08})
    if len(sigmas) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(budgets)))

    for si, (ax, sigma) in enumerate(zip(axes, sigmas)):
        sk = f"sigma={sigma}"
        sr = results[sk]

        opt_C = []
        opt_ratio = []
        opt_r = []

        for bi, (C, clr) in enumerate(zip(budgets, colors)):
            ck = f"C={C}"
            if ck not in sr:
                continue
            cr = sr[ck]
            if "below_threshold" in cr:
                continue

            K3_opt = cr["K3_optimal"]
            K2_opt = cr["K2_optimal"]
            r_opt = cr["r_optimal"]
            ratio = cr["ratio_K3_C"]

            opt_C.append(C)
            opt_ratio.append(ratio)
            opt_r.append(r_opt)

            # mark on plot
            ax.bar(C, r_opt, width=0.6, color=clr, alpha=0.8,
                   label=f"$C={C:.0f}$: $K_3/C={ratio:.2f}$")

        ax.set_xlabel("Budget $C = K_2 + K_3$")
        ax.set_title(f"$\\sigma = {sigma}$")
        if si == 0:
            ax.set_ylabel("Optimal $r^*$")
        ax.legend(fontsize=7, framealpha=0.9, loc="upper left")
        ax.grid(True, alpha=0.15)
        ax.text(-0.12 if si == 0 else -0.05, 1.06, chr(65 + si),
                transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    fig.suptitle("Optimal coupling allocation under budget constraint $K_2 + K_3 = C$",
                 fontsize=12, y=1.02)

    path = os.path.join(FIG_DIR, "budget_optimization.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_budget_optimization()
