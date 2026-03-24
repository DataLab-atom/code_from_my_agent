"""
All 8 paper figures — unified publication quality (Nature/NeurIPS)
Consistent: Times New Roman 7pt, 600dpi, 0.5pt axes, Type42 PDF
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from collections import defaultdict
import os

# ─── Global Nature style ───
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "lines.linewidth": 1.0, "lines.markersize": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "paper/figures"
C_BLUE = "#1976D2"
C_RED = "#D32F2F"
C_ORANGE = "#F57C00"
C_GREEN = "#4CAF50"
C_PURPLE = "#7B1FA2"
C_GRAY = "#757575"

def _label(ax, s, x=-0.18, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold")

def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{OUT}/{name}.{fmt}")
    plt.close(fig)
    print(f"  {name}")


# ════════════════════════════════════════════════════════════
# Fig 2: Hysteresis loops
# ════════════════════════════════════════════════════════════
def fig2():
    d = json.load(open("hysteresis.json"))
    K2 = np.array(d["K2_list"])
    K3s = d["K3_list"]
    rf = [np.array(r) for r in d["r_forward"]]
    rb = [np.array(r) for r in d["r_backward"]]

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.0),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.3, "wspace": 0.12})
    axes = axes.flatten()

    for i, K3 in enumerate(K3s):
        ax = axes[i]
        ax.plot(K2, rf[i], "-", color=C_BLUE, lw=0.8, label="Forward")
        ax.plot(K2, rb[i], "-", color=C_RED, lw=0.8, label="Backward")
        H = np.trapz(np.abs(rf[i] - rb[i]), K2)
        ax.fill_between(K2, rf[i], rb[i], alpha=0.12, color=C_PURPLE)
        ax.set_title(f"$K_3={K3:+.1f}$  $H={H:.2f}$", fontsize=6)
        ax.set_ylim(-0.05, 1.05)
        if i >= 4: ax.set_xlabel("$K_2$")
        if i % 4 == 0: ax.set_ylabel("$r$")
        if i == 0: ax.legend(fontsize=5, frameon=False, loc="upper left")
        _label(ax, chr(97 + i), x=-0.12)

    # Summary bar in last panel
    ax_s = axes[7]
    ax_s.clear()
    areas = [np.trapz(np.abs(rf[i] - rb[i]), K2) for i in range(len(K3s))]
    colors = [C_RED if K3 < 0 else C_GRAY if K3 == 0 else C_BLUE for K3 in K3s]
    ax_s.bar(range(len(K3s)), areas, color=colors, width=0.7, edgecolor="0.3", lw=0.3)
    ax_s.set_xticks(range(len(K3s)))
    ax_s.set_xticklabels([f"{K3:+.1f}" for K3 in K3s], fontsize=5)
    ax_s.set_xlabel("$K_3$")
    ax_s.set_ylabel("$H$ (area)")
    ax_s.set_title("Hysteresis area", fontsize=6)
    _label(ax_s, "h", x=-0.12)

    fig.suptitle("Hysteresis loops: forward vs backward $K_2$ sweeps", fontsize=8, y=0.98)
    _save(fig, "fig2")


# ════════════════════════════════════════════════════════════
# Fig 3: K₃ asymmetry (4 panels)
# ════════════════════════════════════════════════════════════
def fig3():
    d = json.load(open("K3_asymmetry.json"))
    K3 = np.array(d["K3_list"])
    r1 = np.array(d["r1_steady"])
    r2 = np.array(d["r2_steady"])
    basin = np.array(d["basin_prob"])
    ts = d["r1_timeseries"]
    rec = d["record_every_steps"]

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(K3, r1, "-", color=C_BLUE, lw=1.0, label="$r_1$")
    ax_a.plot(K3, r2, "--", color=C_ORANGE, lw=0.8, label="$r_2$")
    ax_a.axvline(0, color="0.7", ls=":", lw=0.3)
    ax_a.set_xlabel("$K_3$"); ax_a.set_ylabel("Order parameter")
    ax_a.legend(frameon=False); _label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(K3, basin, "-", color=C_GREEN, lw=1.0)
    ax_b.axvline(0, color="0.7", ls=":", lw=0.3)
    ax_b.set_xlabel("$K_3$"); ax_b.set_ylabel("Basin probability")
    _label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    r1sq = r1**2
    ratio = np.where(r1sq > 0.01, r2 / r1sq, np.nan)
    ax_c.plot(K3, ratio, "-", color=C_PURPLE, lw=1.0)
    ax_c.axhline(1, color=C_RED, ls="--", lw=0.6, label="OA: $r_2=r_1^2$")
    ax_c.axvline(0, color="0.7", ls=":", lw=0.3)
    ax_c.set_xlabel("$K_3$"); ax_c.set_ylabel("$r_2/r_1^2$")
    ax_c.legend(frameon=False, fontsize=5); _label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    for target, clr, ls in [(-2, C_RED, "-"), (0, C_GRAY, "-"), (2, C_BLUE, "-")]:
        idx = np.argmin(np.abs(K3 - target))
        t_arr = np.array(ts[idx])
        t_axis = np.arange(len(t_arr)) * rec * 0.01
        ax_d.plot(t_axis, t_arr, color=clr, lw=0.6, alpha=0.9,
                  label=f"$K_3={K3[idx]:+.1f}$")
    ax_d.set_xlabel("Time"); ax_d.set_ylabel("$r_1(t)$")
    ax_d.legend(frameon=False, fontsize=5); _label(ax_d, "d")

    fig.suptitle(f"$K_3$ asymmetry ($K_2={d['K2']}$, $\\sigma={d['sigma']}$, $N={d['N']}$)",
                 fontsize=8, y=0.98)
    _save(fig, "fig3")


# ════════════════════════════════════════════════════════════
# Fig 4: Critical slowing down
# ════════════════════════════════════════════════════════════
def fig4():
    d = json.load(open("critical_slowing.json"))
    K2 = np.array(d["K2_list"])
    K3s = d["K3_list"]
    tau = [np.array(t) for t in d["tau_relax"]]
    r_ss = [np.array(r) for r in d["r_steady"]]
    Kc = d["Kc_est"]

    colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    for i, (K3, t, c) in enumerate(zip(K3s, tau, colors)):
        ax.plot(K2, t, "o-", color=c, ms=2.5, lw=0.8, label=f"$K_3={K3:+.1f}$")
    ax.axvline(Kc, color="0.5", ls="--", lw=0.5)
    ax.set_xlabel("$K_2$"); ax.set_ylabel("$\\tau_{\\mathrm{relax}}$")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=5); _label(ax, "a")

    ax2 = axes[1]
    for i, (K3, r, c) in enumerate(zip(K3s, r_ss, colors)):
        ax2.plot(K2, r, "o-", color=c, ms=2.5, lw=0.8, label=f"$K_3={K3:+.1f}$")
    ax2.axvline(Kc, color="0.5", ls="--", lw=0.5)
    ax2.axhline(0.5, color="0.7", ls=":", lw=0.3)
    ax2.set_xlabel("$K_2$"); ax2.set_ylabel("Steady-state $r$")
    ax2.legend(frameon=False, fontsize=5); _label(ax2, "b")

    fig.suptitle(f"Critical slowing down ($N={d['N']}$, $\\sigma={d['sigma']}$)",
                 fontsize=8, y=0.98)
    _save(fig, "fig4")


# ════════════════════════════════════════════════════════════
# Fig 6: Locking threshold
# ════════════════════════════════════════════════════════════
def fig6():
    d = json.load(open("locking_order.json"))
    omega = np.array(d["omega_i"])
    omega_abs = np.abs(omega)
    configs = [
        (np.array(d["t_lock_K3neg"]), "$K_3<0$", C_RED),
        (np.array(d["t_lock_K3zero"]), "$K_3=0$", C_GRAY),
        (np.array(d["t_lock_K3pos"]), "$K_3>0$", C_BLUE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.3})

    # Panel a: sigmoid curves
    ax = axes[0]
    bins = np.linspace(0, omega_abs.max(), 20)
    centers = (bins[:-1] + bins[1:]) / 2
    for t_lock, label, color in configs:
        locked = t_lock >= 0
        probs = []
        for j in range(len(bins) - 1):
            mask = (omega_abs >= bins[j]) & (omega_abs < bins[j + 1])
            probs.append(locked[mask].mean() if mask.sum() > 0 else np.nan)
        ax.plot(centers, probs, "o-", color=color, ms=2.5, lw=0.8, label=label)
    ax.axhline(0.5, color="0.7", ls=":", lw=0.3)
    ax.set_xlabel("$|\\omega_i|$"); ax.set_ylabel("$P(\\mathrm{locked} | \\omega)$")
    ax.legend(frameon=False, fontsize=5); _label(ax, "a")

    # Panel b: omega_c bar
    ax2 = axes[1]
    from scipy.optimize import curve_fit
    def sigmoid(x, wc, delta):
        return 1.0 / (1.0 + np.exp((x - wc) / max(delta, 0.01)))

    wc_vals = []
    for t_lock, label, color in configs:
        locked = (t_lock >= 0).astype(float)
        try:
            popt, _ = curve_fit(sigmoid, omega_abs, locked, p0=[1.5, 0.3], maxfev=5000)
            wc_vals.append(popt[0])
        except:
            wc_vals.append(0)

    bars = ax2.bar(["$K_3<0$", "$K_3=0$", "$K_3>0$"], wc_vals,
                    color=[C_RED, C_GRAY, C_BLUE], width=0.5,
                    edgecolor="0.3", lw=0.3)
    for bar, wc in zip(bars, wc_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, wc + 0.05,
                 f"{wc:.2f}", ha="center", fontsize=6)
    ax2.set_ylabel("$\\omega_c$ (locking threshold)")
    _label(ax2, "b")

    fig.suptitle(f"Locking threshold analysis ($K_2={d['K2']}$, $N={len(omega)}$)",
                 fontsize=8, y=0.98)
    _save(fig, "fig6")


# ════════════════════════════════════════════════════════════
# Fig 7: OA direction error
# ════════════════════════════════════════════════════════════
def fig7():
    d_num = json.load(open("scan_K2_K3.json"))
    d_gauss = json.load(open("gaussian_exact_results.json"))

    K2_n = np.array(d_num["K2_list"])
    K3_n = np.array(d_num["K3_list"])
    r_n = np.array(d_num["r"])
    K2_g = np.array(d_gauss["K2_range"])
    K3_g = np.array(d_gauss["K3_range"])
    r_g = np.array(d_gauss["r_gaussian_2d"])

    sigma = d_num["sigma"]
    Delta = sigma * np.sqrt(2 * np.pi) / np.pi

    def oa_r(K2, K3):
        a, b, c = K3, -(K2 + K3), K2 - 2 * Delta
        if abs(a) < 1e-12:
            u = -c / b if abs(b) > 1e-12 else 0
            return np.sqrt(u) if 0 < u < 1 else 0
        disc = b * b - 4 * a * c
        if disc < 0: return 0
        sols = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        rs = [np.sqrt(u) for u in sols if 0 < u < 1]
        return max(rs) if rs else 0

    K2_target = 2.1
    ki_n = np.argmin(np.abs(K2_n - K2_target))
    ki_g = np.argmin(np.abs(K2_g - K2_target))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    ax.plot(K3_n, r_n[ki_n, :], "ko-", ms=3, lw=1.2, label=f"Numerical ($N={d_num['N']}$)")
    K3_gp = K3_g[K3_g <= K3_n.max() + 0.5]
    ax.plot(K3_gp, r_g[ki_g, :len(K3_gp)], "s--", color=C_GREEN, ms=3, lw=1.0,
            label="Gaussian exact ($N\\to\\infty$)")
    K3_fine = np.linspace(K3_n.min(), K3_n.max(), 50)
    r_oa = [oa_r(K2_n[ki_n], k3) for k3 in K3_fine]
    ax.plot(K3_fine, r_oa, "--", color=C_RED, lw=1.2, label="Lorentzian OA")

    ax.annotate("OA: wrong\ndirection!", xy=(1.5, 0.38), fontsize=6,
                color=C_RED, fontweight="bold", ha="center")
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$r^*$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, edgecolor="0.7", fontsize=5.5, loc="lower left")
    _label(ax, "a")

    # Panel b: direction match map
    ax2 = axes[1]
    K3g, K2g = np.meshgrid(K3_n, K2_n)
    K3_zero = np.argmin(np.abs(K3_n))
    dm = np.zeros_like(r_n)
    for i in range(len(K2_n)):
        r_oa_0 = oa_r(K2_n[i], 0)
        r_n_0 = r_n[i, K3_zero]
        for j in range(len(K3_n)):
            if abs(K3_n[j]) < 0.2: dm[i, j] = 0
            else:
                oa_dir = np.sign(oa_r(K2_n[i], K3_n[j]) - r_oa_0)
                n_dir = np.sign(r_n[i, j] - r_n_0)
                dm[i, j] = 1 if oa_dir == n_dir else -1

    im = ax2.pcolormesh(K3g, K2g, dm, cmap="RdYlGn", vmin=-1, vmax=1,
                         shading="auto", rasterized=True)
    agree = (dm == 1).sum()
    total = (dm != 0).sum()
    ax2.text(0.05, 0.95, f"Agreement: {agree}/{total} ({agree/total*100:.0f}%)",
             transform=ax2.transAxes, fontsize=6, va="top",
             bbox=dict(boxstyle="round", facecolor="w", alpha=0.9, edgecolor="0.7"))
    cb = fig.colorbar(im, ax=ax2, ticks=[-1, 0, 1], shrink=0.85)
    cb.set_ticklabels(["Opposite", "Neutral", "Agree"])
    cb.ax.tick_params(labelsize=5)
    ax2.set_xlabel("$K_3$"); ax2.set_ylabel("$K_2$")
    _label(ax2, "b")

    fig.suptitle("Lorentzian OA predicts wrong $K_3$ effect direction",
                 fontsize=8, y=0.98)
    _save(fig, "fig7")


# ════════════════════════════════════════════════════════════
# Fig 8: Finite-size N=20000
# ════════════════════════════════════════════════════════════
def fig8():
    d = json.load(open("finite_size_very_large_N.json"))
    results = d["results"]
    N_list = [r["N"] for r in results]
    K3_keys = ["K3=-1.0", "K3=0.0", "K3=1.0", "K3=2.0"]
    K3_vals = [-1, 0, 1, 2]
    colors = [C_RED, C_GRAY, C_BLUE, C_GREEN]

    sigma = d["sigma"]
    Delta = sigma * np.sqrt(2 * np.pi) / np.pi
    K2 = d["K2"]

    def oa_r(K2v, K3v):
        a, b, c = K3v, -(K2v + K3v), K2v - 2 * Delta
        if abs(a) < 1e-12:
            u = -c / b if abs(b) > 1e-12 else 0
            return np.sqrt(u) if 0 < u < 1 else 0
        disc = b * b - 4 * a * c
        if disc < 0: return 0
        sols = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        rs = [np.sqrt(u) for u in sols if 0 < u < 1]
        return max(rs) if rs else 0

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    for ki, (k3k, k3v, c) in enumerate(zip(K3_keys, K3_vals, colors)):
        r_vals = [r[k3k] for r in results]
        ax.plot(N_list, r_vals, "o-", color=c, ms=3, lw=1.0, label=f"$K_3={k3v:+d}$")
    ax.set_xscale("log"); ax.set_xlabel("$N$"); ax.set_ylabel("$r$")
    ax.legend(frameon=False, fontsize=5); _label(ax, "a")

    ax2 = axes[1]
    delta = [r["K3=1.0"] - r["K3=0.0"] for r in results]
    ax2.plot(N_list, delta, "ko-", ms=4, lw=1.2, label="Numerical")
    delta_inf = 0.909 - 0.716
    delta_oa = oa_r(K2, 1.0) - oa_r(K2, 0.0)
    ax2.axhline(delta_inf, color=C_GREEN, ls="--", lw=0.8,
                label=f"Gaussian exact: {delta_inf:.3f}")
    ax2.axhline(delta_oa, color=C_RED, ls="--", lw=0.8,
                label=f"Lorentzian OA: {delta_oa:+.3f}")
    ax2.axhline(0, color="0.7", ls=":", lw=0.3)
    ax2.set_xscale("log"); ax2.set_xlabel("$N$")
    ax2.set_ylabel("$\\Delta r$ ($K_3$=+1 vs 0)")
    ax2.legend(frameon=True, edgecolor="0.7", fontsize=5.5); _label(ax2, "b")

    fig.suptitle(f"Finite-size scaling to $N=20\\,000$ ($K_2={K2}$, $\\sigma={sigma}$)",
                 fontsize=8, y=0.98)
    _save(fig, "fig8")


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...")
    fig2(); fig3(); fig4(); fig6(); fig7(); fig8()
    print("Done. All in paper/figures/")
