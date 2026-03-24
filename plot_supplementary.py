"""
All supplementary figures — unified publication quality
Same style as main figures: Times New Roman 7pt, 600dpi, Type42
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator

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

OUT = "paper/supplementary"
CB, CR, CO, CG, CP, CGR = "#1976D2", "#D32F2F", "#F57C00", "#4CAF50", "#7B1FA2", "#757575"

def _label(ax, s): ax.text(-0.15, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold")
def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for fmt in ["pdf", "png"]: fig.savefig(f"{OUT}/{name}.{fmt}")
    plt.close(fig); print(f"  {name}")

def oa_delta(sigma): return sigma * np.sqrt(2*np.pi) / np.pi
def oa_r(K2, K3, Delta):
    a,b,c = K3, -(K2+K3), K2-2*Delta
    if abs(a)<1e-12:
        u = -c/b if abs(b)>1e-12 else 0
        return np.sqrt(u) if 0<u<1 else 0
    disc = b*b-4*a*c
    if disc<0: return 0
    sols = [(-b+np.sqrt(disc))/(2*a), (-b-np.sqrt(disc))/(2*a)]
    rs = [np.sqrt(u) for u in sols if 0<u<1]
    return max(rs) if rs else 0


# ── S1: Benchmark classical ──
def s_benchmark():
    d = json.load(open("benchmark_classical.json"))
    K2, r = np.array(d["K2_list"]), np.array(d["r_list"])
    sigma = d.get("sigma", 1.0)
    Kc = 2*sigma*np.sqrt(2*np.pi)/np.pi
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    ax = axes[0]
    ax.plot(K2, r, "o-", color=CB, ms=3, lw=1); ax.axvline(Kc, color=CR, ls="--", lw=0.8)
    ax.set_xlabel("$K_2$"); ax.set_ylabel("$r$"); _label(ax, "a")
    ax2 = axes[1]
    mask = K2 > Kc*1.05
    if mask.any():
        x = np.sqrt(K2[mask]-Kc); ax2.plot(x, r[mask], "o", color=CG, ms=3)
        c = np.polyfit(x, r[mask], 1); ax2.plot(np.linspace(0,x.max(),50), np.polyval(c,np.linspace(0,x.max(),50)), "--", color=CR, lw=0.8)
    ax2.set_xlabel("$\\sqrt{K_2-K_c}$"); ax2.set_ylabel("$r$"); _label(ax2, "b")
    fig.suptitle(f"Classical Kuramoto benchmark ($K_c={Kc:.3f}$)", fontsize=8, y=0.98)
    _save(fig, "benchmark_classical")

# ── S2: OA verification ──
def s_oa_verification():
    d = json.load(open("verify_oa_results.json"))
    entries = d["Kc_vs_sigma"]
    sigmas = [e["sigma"] for e in entries]
    Kc_num = [e["Kc_num"] for e in entries]
    Kc_oa_new = [2*s*np.sqrt(2*np.pi)/np.pi for s in sigmas]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(sigmas, Kc_num, "ko-", ms=4, lw=1.2, label="Numerical")
    ax.plot(sigmas, Kc_oa_new, "s--", color=CB, ms=3, lw=0.8, label="$2\\sigma\\sqrt{2\\pi}/\\pi$")
    ax.set_xlabel("$\\sigma$"); ax.set_ylabel("$K_c$"); ax.legend(frameon=False)
    fig.suptitle("$K_c$ verification", fontsize=8, y=0.98)
    _save(fig, "oa_verification")

# ── S3: K4 effect ──
def s_K4():
    d = json.load(open("K4_test.json"))
    K4, r1 = np.array(d["K4_list"]), np.array(d["r1"])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(K4, r1, "o-", color=CG, ms=3, lw=1); ax.axvline(0, color="0.7", ls=":", lw=0.3)
    ax.set_xlabel("$K_4$"); ax.set_ylabel("$r$")
    fig.suptitle(f"Four-body coupling effect ($K_2={d['K2']}$, $K_3={d['K3']}$)", fontsize=8, y=0.98)
    _save(fig, "K4_effect")

# ── S4: Cluster structure ──
def s_cluster():
    d = json.load(open("cluster_structure.json"))
    results = d["results"]
    K3v = [r["K3"] for r in results]
    nc = [r["n_clusters"] for r in results]
    r1v = [r["r1"] for r in results]
    r2v = [r["r2"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), gridspec_kw={"wspace": 0.35})
    axes[0].plot(K3v, nc, "o-", color=CP, ms=3, lw=1); axes[0].set_ylabel("Clusters"); _label(axes[0],"a")
    axes[1].plot(K3v, r1v, "o-", color=CB, ms=3, label="$r_1$"); axes[1].plot(K3v, r2v, "s-", color=CO, ms=3, label="$r_2$")
    axes[1].legend(frameon=False); axes[1].set_ylabel("Order param."); _label(axes[1],"b")
    ratio = np.where(np.array(r1v)**2>0.01, np.array(r2v)/np.array(r1v)**2, np.nan)
    axes[2].plot(K3v, ratio, "o-", color=CG, ms=3); axes[2].axhline(1, color=CR, ls="--", lw=0.5)
    axes[2].set_ylabel("$r_2/r_1^2$"); _label(axes[2],"c")
    for ax in axes: ax.set_xlabel("$K_3$")
    fig.suptitle("Cluster structure analysis", fontsize=8, y=0.98)
    _save(fig, "cluster_structure")

# ── S5: Gaussian exact ──
def s_gaussian():
    d = json.load(open("gaussian_exact_results.json"))
    K2, K3 = np.array(d["K2_range"]), np.array(d["K3_range"])
    r_g = np.array(d["r_gaussian_2d"])
    K3g, K2g = np.meshgrid(K3, K2)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.pcolormesh(K3g, K2g, r_g, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.contour(K3g, K2g, r_g, levels=[0.3,0.5,0.7], colors=["w"], linewidths=[0.4,0.8,0.4])
    fig.colorbar(im, ax=ax, shrink=0.85, label="$r^*$")
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$K_2$")
    fig.suptitle("Gaussian exact self-consistent phase diagram", fontsize=8, y=0.98)
    _save(fig, "gaussian_exact")

# ── S6: Budget optimization ──
def s_budget():
    d = json.load(open("budget_optimization.json"))
    sigmas, budgets = d["sigmas"], d["budgets"]
    results = d["results"]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(7, 3), sharey=True, gridspec_kw={"wspace": 0.08})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(budgets)))
    for si, (ax, sigma) in enumerate(zip(axes, sigmas)):
        for bi, (C, clr) in enumerate(zip(budgets, colors)):
            cr = results[f"sigma={sigma}"].get(f"C={C}", {})
            if "r_optimal" in cr and cr["r_optimal"] > 0:
                ax.bar(C, cr["r_optimal"], width=0.6, color=clr, alpha=0.8)
        ax.set_xlabel("Budget $C$"); ax.set_title(f"$\\sigma={sigma}$", fontsize=7)
        if si == 0: ax.set_ylabel("Optimal $r^*$")
    fig.suptitle("Budget-constrained optimal coupling", fontsize=8, y=0.98)
    _save(fig, "budget_optimization")

# ── S7: Freq distribution ──
def s_freq():
    d = json.load(open("freq_distribution.json"))
    results = d["results"]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    seen = set()
    for r in results:
        key = (r["dist_type"], r["K3"])
        if r["K3"] != 0 or key in seen: continue
        seen.add(key)
        c = {"gaussian": CB, "lorentzian": CR, "uniform": CG}.get(r["dist_type"], CGR)
        ax.plot(r["K2_list"], r["r1"], "o-", color=c, ms=2, lw=0.8, label=r["dist_type"].capitalize())
    ax.set_xlabel("$K_2$"); ax.set_ylabel("$r$"); ax.legend(frameon=False)
    fig.suptitle("Frequency distribution comparison ($K_3=0$)", fontsize=8, y=0.98)
    _save(fig, "freq_distribution")

# ── S8: Locking order scatter ──
def s_locking_order():
    d = json.load(open("locking_order.json"))
    omega = np.abs(np.array(d["omega_i"]))
    configs = [(np.array(d["t_lock_K3neg"]), "$K_3<0$", CR),
               (np.array(d["t_lock_K3zero"]), "$K_3=0$", CGR),
               (np.array(d["t_lock_K3pos"]), "$K_3>0$", CB)]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (tl, lab, c) in zip(axes, configs):
        locked = tl >= 0
        ax.scatter(omega[locked], tl[locked], c=c, s=5, alpha=0.5, edgecolors="none")
        ax.set_xlabel("$|\\omega_i|$"); ax.set_title(f"{lab} ({locked.sum()}/200)", fontsize=7)
    axes[0].set_ylabel("$t_{\\mathrm{lock}}$")
    fig.suptitle("Locking order scatter", fontsize=8, y=0.98)
    _save(fig, "locking_order")

# ── S9: OA deviation map ──
def s_oa_deviation():
    d = json.load(open("scan_K2_K3.json"))
    K2, K3, r_n = np.array(d["K2_list"]), np.array(d["K3_list"]), np.array(d["r"])
    Delta = oa_delta(d["sigma"])
    r_oa = np.array([[oa_r(k2, k3, Delta) for k3 in K3] for k2 in K2])
    dev = np.abs(r_oa - r_n)
    K3g, K2g = np.meshgrid(K3, K2)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.pcolormesh(K3g, K2g, dev, cmap="Reds", vmin=0, shading="auto", rasterized=True)
    ax.contour(K3g, K2g, dev, levels=[0.05,0.1,0.2], colors=["k"], linewidths=[0.4,0.6,0.4])
    fig.colorbar(im, ax=ax, shrink=0.85, label="$|r_{OA}-r_{num}|$")
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$K_2$")
    safe = (dev<0.05).sum()/dev.size*100
    ax.text(0.05, 0.95, f"Safe (<5%): {safe:.0f}%", transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round", facecolor="w", alpha=0.9))
    fig.suptitle("OA prediction deviation map", fontsize=8, y=0.98)
    _save(fig, "oa_deviation")

# ── S10: Z2/Z1 ratio ──
def s_Z2():
    d = json.load(open("Z2_Z1_ratio.json"))
    results = d["results"]
    K2v = sorted(set(r["K2"] for r in results))
    K3v = sorted(set(r["K3"] for r in results))
    mat = np.full((len(K2v), len(K3v)), np.nan)
    for r in results: mat[K2v.index(r["K2"]), K3v.index(r["K3"])] = r["Z2_over_Z1sq_abs"]
    K3g, K2g = np.meshgrid(K3v, K2v)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.pcolormesh(K3g, K2g, mat, cmap="RdYlGn", vmin=0, vmax=1.2, shading="auto", rasterized=True)
    ax.contour(K3g, K2g, mat, levels=[0.9,0.95,1.0], colors=["k"], linewidths=[0.4,0.6,0.8])
    fig.colorbar(im, ax=ax, shrink=0.85, label="$|Z_2/Z_1^2|$")
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$K_2$")
    fig.suptitle("OA manifold validity ($N=500$)", fontsize=8, y=0.98)
    _save(fig, "Z2_Z1_ratio")

# ── S11: Corrected vs numerical ──
def s_corrected():
    corr = json.load(open("corrected_phase_diagram.json"))
    num = json.load(open("scan_K2_K3.json"))
    K2c, K3c, r_c = np.array(corr["K2_range"]), np.array(corr["K3_range"]), np.array(corr["r_corrected"])
    K2n, K3n, r_n = np.array(num["K2_list"]), np.array(num["K3_list"]), np.array(num["r"])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    K3cg, K2cg = np.meshgrid(K3c, K2c)
    axes[0].pcolormesh(K3cg, K2cg, r_c, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
    axes[0].set_title("Corrected Gaussian exact"); _label(axes[0], "a")
    K3ng, K2ng = np.meshgrid(K3n, K2n)
    axes[1].pcolormesh(K3ng, K2ng, r_n, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
    axes[1].set_title(f"Numerical ($N={num['N']}$)"); _label(axes[1], "b")
    for ax in axes: ax.set_xlabel("$K_3$"); ax.set_ylabel("$K_2$")
    fig.suptitle("Corrected Gaussian exact vs Numerical", fontsize=8, y=0.98)
    _save(fig, "corrected_vs_numerical")

# ── S12: Hysteresis fine ──
def s_hyst_fine():
    d = json.load(open("hysteresis_fine.json"))
    K3s, H = np.array(d["K3_list"]), np.array(d["hysteresis_area"])
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(K3s, H, "o-", color=CP, ms=3, lw=1)
    ax.set_xlabel("$K_3$"); ax.set_ylabel("Hysteresis area $H$")
    fig.suptitle("Fine hysteresis scan (30 $K_3$ values)", fontsize=8, y=0.98)
    _save(fig, "hysteresis_fine")

# ── S13: Finite size K3 (N=20-500) ──
def s_finite_size():
    d = json.load(open("finite_size_K3.json"))
    N_list, K3s = d["N_list"], np.array(d["K3_list"])
    r_mean = np.array(d["r_mean"])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_list)))
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for ni, (N, c) in enumerate(zip(N_list, colors)):
        ax.plot(K3s, r_mean[ni], "o-", color=c, ms=3, lw=0.8, label=f"$N={N}$")
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$r$"); ax.legend(frameon=False, fontsize=5)
    fig.suptitle("Finite-size scaling ($N=20$-$500$)", fontsize=8, y=0.98)
    _save(fig, "finite_size_K3")

# ── S14: sigma-K3 interaction ──
def s_sigma_K3():
    d = json.load(open("scan_sigma_K2_K3.json"))
    sigma_list = d["sigma_list"]
    K2, K3, r = np.array(d["K2_list"]), np.array(d["K3_list"]), np.array(d["r"])
    K3_zero = np.argmin(np.abs(K3))
    delta_r = np.zeros((len(sigma_list), len(K3)))
    for si, sigma in enumerate(sigma_list):
        Kc = 2*sigma*np.sqrt(2*np.pi)/np.pi
        m = K2 > Kc*0.9
        if m.any():
            r0 = r[si, m, K3_zero].mean()
            for j in range(len(K3)): delta_r[si,j] = r[si,m,j].mean() - r0
    K3g, Sg = np.meshgrid(K3, sigma_list)
    vmax = max(abs(delta_r.min()), abs(delta_r.max()), 0.01)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.pcolormesh(K3g, Sg, delta_r, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax.contour(K3g, Sg, delta_r, levels=[0], colors=["k"], linewidths=[1])
    fig.colorbar(im, ax=ax, shrink=0.85, label="$\\Delta r$")
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$\\sigma$")
    fig.suptitle("$\\sigma \\times K_3$ interaction", fontsize=8, y=0.98)
    _save(fig, "sigma_K3_interaction")

# ── S15: Dist K3 direction ──
def s_dist_direction():
    d = json.load(open("freq_distribution.json"))
    results = d["results"]
    data = {}
    for r in results:
        key = r["dist_type"]
        if key not in data: data[key] = {}
        data[key][r["K3"]] = r
    fig, ax = plt.subplots(figsize=(4.5, 3))
    colors_d = {"gaussian": CB, "lorentzian": CR, "uniform": CG}
    for dist in ["gaussian", "lorentzian", "uniform"]:
        if dist not in data: continue
        K3_vals = sorted(data[dist].keys())
        Kc_vals = [data[dist][k3].get("Kc_measured", 0) for k3 in K3_vals]
        ax.plot(K3_vals, Kc_vals, "o-", color=colors_d[dist], ms=3, lw=1, label=dist.capitalize())
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$K_c$"); ax.legend(frameon=False)
    fig.suptitle("$K_c$ shift by distribution and $K_3$", fontsize=8, y=0.98)
    _save(fig, "dist_K3_direction")

# ── S16: K3 help/hurt map ──
def s_help_hurt():
    d = json.load(open("scan_sigma_K2_K3.json"))
    sigma_list = d["sigma_list"]
    K2, K3, r = np.array(d["K2_list"]), np.array(d["K3_list"]), np.array(d["r"])
    K3_zero = np.argmin(np.abs(K3))
    nS = len(sigma_list)
    fig, axes = plt.subplots(2, 3, figsize=(7, 5), gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    axes = axes.flatten()
    for si, sigma in enumerate(sigma_list):
        ax = axes[si]
        Kc = 2*sigma*np.sqrt(2*np.pi)/np.pi
        K3g, K2g = np.meshgrid(K3[K3>=0], K2)
        delta = np.zeros_like(K3g)
        for ki in range(len(K2)):
            r0 = r[si, ki, K3_zero]
            for j, k3 in enumerate(K3[K3>=0]):
                jf = np.argmin(np.abs(K3 - k3))
                delta[ki,j] = r[si,ki,jf] - r0
        vmax = max(abs(delta.min()), abs(delta.max()), 0.01)
        ax.pcolormesh(K3g, K2g/Kc, delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.contour(K3g, K2g/Kc, delta, levels=[0], colors=["k"], linewidths=[0.8])
        ax.axhline(1.0, color="0.5", ls="--", lw=0.4)
        ax.set_title(f"$\\sigma={sigma:.1f}$", fontsize=6)
        if si >= 3: ax.set_xlabel("$K_3$")
        if si % 3 == 0: ax.set_ylabel("$K_2/K_c$")
    fig.suptitle("$K_3>0$ effect map (blue=helps, red=hurts)", fontsize=8, y=0.98)
    _save(fig, "K3_help_hurt_map")

# ── S17: Kc vs K3 ──
def s_Kc_K3():
    d = json.load(open("scan_K2_K3.json"))
    K2, K3, r = np.array(d["K2_list"]), np.array(d["K3_list"]), np.array(d["r"])
    Kc_list = []
    for j in range(len(K3)):
        Kc = np.nan
        for ki in range(len(K2)-1):
            if r[ki,j]<0.3<=r[ki+1,j]:
                frac = (0.3-r[ki,j])/(r[ki+1,j]-r[ki,j])
                Kc = K2[ki]+frac*(K2[ki+1]-K2[ki]); break
        Kc_list.append(Kc)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    valid = [(K3[j], Kc_list[j]) for j in range(len(K3)) if not np.isnan(Kc_list[j])]
    if valid:
        k3v, kcv = zip(*valid)
        ax.plot(k3v, kcv, "ko-", ms=3, lw=1)
    Kc_th = 2*d["sigma"]*np.sqrt(2*np.pi)/np.pi
    ax.axhline(Kc_th, color=CR, ls="--", lw=0.8, label=f"OA: $K_c={Kc_th:.3f}$")
    ax.set_xlabel("$K_3$"); ax.set_ylabel("$K_c$"); ax.legend(frameon=False)
    fig.suptitle("Effective $K_c$ vs $K_3$", fontsize=8, y=0.98)
    _save(fig, "Kc_vs_K3")

# ── S18-S25: 3D scan suite (fig1-fig8 from plot_publication.py) ──
def s_3d_suite():
    d = json.load(open("scan_sigma_K2_K3.json"))
    sigma_list = d["sigma_list"]
    K2, K3 = np.array(d["K2_list"]), np.array(d["K3_list"])
    r, basin = np.array(d["r"]), np.array(d["basin"])
    K3g, K2g = np.meshgrid(K3, K2)
    nS = len(sigma_list)

    # Phase diagram matrix
    fig, axes = plt.subplots(2, nS, figsize=(2.5*nS+1, 5.5), gridspec_kw={"hspace":0.3, "wspace":0.12})
    if nS == 1: axes = axes.reshape(2,1)
    for i, sig in enumerate(sigma_list):
        Kc = 2*sig*np.sqrt(2*np.pi)/np.pi
        im = axes[0,i].pcolormesh(K3g, K2g, r[i], cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
        axes[0,i].contour(K3g, K2g, r[i], levels=[0.5], colors=["w"], linewidths=[0.6])
        if Kc<=K2.max(): axes[0,i].axhline(Kc, color="w", ls=":", lw=0.4)
        axes[0,i].set_title(f"$\\sigma={sig:.1f}$", fontsize=6)
        if i==0: axes[0,i].set_ylabel("$K_2$")
        im2 = axes[1,i].pcolormesh(K3g, K2g, basin[i], cmap="RdBu", vmin=0, vmax=1, shading="auto", rasterized=True)
        axes[1,i].set_xlabel("$K_3$")
        if i==0: axes[1,i].set_ylabel("$K_2$")
    fig.colorbar(im, ax=axes[0,:].tolist(), shrink=0.8, pad=0.02, label="$r$")
    fig.colorbar(im2, ax=axes[1,:].tolist(), shrink=0.8, pad=0.02, label="Basin")
    fig.suptitle("Phase diagram matrix (all $\\sigma$)", fontsize=8, y=0.98)
    _save(fig, "fig1_phase_diagram_matrix")

    # Sigma effect
    fig2, axes2 = plt.subplots(1, 3, figsize=(7, 2.8), sharey=True, gridspec_kw={"wspace":0.08})
    K3_zero = np.argmin(np.abs(K3))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, nS))
    for pi, ki in enumerate([len(K2)//4, len(K2)//2, 3*len(K2)//4]):
        for si, (sig, c) in enumerate(zip(sigma_list, colors)):
            axes2[pi].plot(K3, r[si,ki,:], "-", color=c, lw=0.8, label=f"$\\sigma={sig:.1f}$" if pi==2 else None)
        axes2[pi].set_xlabel("$K_3$"); axes2[pi].set_title(f"$K_2={K2[ki]:.1f}$", fontsize=7)
    axes2[0].set_ylabel("$r$"); axes2[2].legend(frameon=False, fontsize=4)
    fig2.suptitle("$\\sigma$ effect on $r$ vs $K_3$", fontsize=8, y=0.98)
    _save(fig2, "fig2_sigma_effect")

    # Kc shift across sigma
    fig3, ax3 = plt.subplots(figsize=(4.5, 3))
    k3_targets = [0, 0.5, 1.0, -0.5, -1.0]
    k3_idxs = {t: np.argmin(np.abs(K3-t)) for t in k3_targets}
    for t in k3_targets:
        ki = k3_idxs[t]
        Kc_vals = []
        for si in range(nS):
            kc = np.nan
            for j in range(len(K2)-1):
                if r[si,j,ki]<0.3<=r[si,j+1,ki]:
                    frac = (0.3-r[si,j,ki])/(r[si,j+1,ki]-r[si,j,ki])
                    kc = K2[j]+frac*(K2[j+1]-K2[j]); break
            Kc_vals.append(kc)
        ax3.plot(sigma_list, Kc_vals, "o-", ms=3, lw=0.8, label=f"$K_3={t:+.1f}$")
    ax3.set_xlabel("$\\sigma$"); ax3.set_ylabel("$K_c$"); ax3.legend(frameon=False, fontsize=5)
    fig3.suptitle("$K_c$ shift by $\\sigma$ and $K_3$", fontsize=8, y=0.98)
    _save(fig3, "fig3_Kc_shift")

    # Triple metric
    si_mid = nS//2; sig = sigma_list[si_mid]
    fig4, axes4 = plt.subplots(1, 3, figsize=(7, 2.5), gridspec_kw={"wspace":0.25})
    for pi, (mat, cmap, lab) in enumerate([(r[si_mid],"viridis","$r$"), (basin[si_mid],"RdBu","Basin"), (np.array(d["tc"])[si_mid],"magma_r","Conv.")]):
        im = axes4[pi].pcolormesh(K3g, K2g, mat, cmap=cmap, shading="auto", rasterized=True)
        fig4.colorbar(im, ax=axes4[pi], shrink=0.85, label=lab)
        axes4[pi].set_xlabel("$K_3$"); _label(axes4[pi], chr(97+pi))
    axes4[0].set_ylabel("$K_2$")
    fig4.suptitle(f"Three metrics ($\\sigma={sig:.1f}$)", fontsize=8, y=0.98)
    _save(fig4, "fig4_triple_metric")

    # Basin vs stability
    fig5, ax5 = plt.subplots(figsize=(4, 3.5))
    for si, (sig, c) in enumerate(zip(sigma_list, colors)):
        rf, bf = r[si].flatten(), basin[si].flatten()
        mask = rf > 0.1
        ax5.scatter(bf[mask], rf[mask], c=[c], s=4, alpha=0.4, edgecolors="none", label=f"$\\sigma={sig:.1f}$")
    ax5.set_xlabel("Basin prob."); ax5.set_ylabel("$r$"); ax5.legend(frameon=False, fontsize=4, markerscale=2)
    fig5.suptitle("Accessibility vs Stability", fontsize=8, y=0.98)
    _save(fig5, "fig5_basin_vs_stability")

    # Asymmetry (using 3D data)
    fig6, ax6 = plt.subplots(figsize=(4.5, 3))
    for si, (sig, c) in enumerate(zip(sigma_list, colors)):
        K3_pos = K3[K3>0]
        K3_neg = K3[K3<0]
        n = min(len(K3_pos), len(K3_neg))
        asym_vals = []
        k3_abs = []
        for p in range(n):
            pi, ni = len(K3)-1-p, p
            if abs(abs(K3[pi])-abs(K3[ni])) < 0.1:
                asym_vals.append(np.mean(r[si,:,pi] - r[si,:,ni]))
                k3_abs.append(abs(K3[pi]))
        if asym_vals:
            ax6.plot(k3_abs, asym_vals, "-", color=c, lw=0.8, label=f"$\\sigma={sig:.1f}$")
    ax6.axhline(0, color="0.7", ls="--", lw=0.3)
    ax6.set_xlabel("$|K_3|$"); ax6.set_ylabel("$\\Delta r$"); ax6.legend(frameon=False, fontsize=4)
    fig6.suptitle("$K_3$ asymmetry", fontsize=8, y=0.98)
    _save(fig6, "fig6_asymmetry")

    # Analytic vs numeric
    fig7, ax7 = plt.subplots(figsize=(4.5, 3.5))
    ax7.pcolormesh(K3g, K2g, r[nS//2], cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
    ax7.contour(K3g, K2g, r[nS//2], levels=[0.5], colors=["w"], linewidths=[1])
    Kc = 2*sigma_list[nS//2]*np.sqrt(2*np.pi)/np.pi
    k3f = np.linspace(K3.min(), K3.max(), 100)
    Delta = oa_delta(sigma_list[nS//2])
    kc_oa = [2*Delta/(1-0.25) - k3*0.25 for k3 in k3f]
    ax7.plot(k3f, kc_oa, "c--", lw=1, label="OA $r=0.5$")
    ax7.set_xlabel("$K_3$"); ax7.set_ylabel("$K_2$"); ax7.legend(frameon=False)
    fig7.suptitle("Analytic vs Numeric boundary", fontsize=8, y=0.98)
    _save(fig7, "fig7_analytic_vs_numeric")

    # Reentrant map
    fig8, ax8 = plt.subplots(figsize=(4.5, 3.5))
    # dr/dK3 map
    dr_dk3 = np.zeros((len(K2), len(K3)))
    for ki in range(len(K2)):
        for j in range(1, len(K3)):
            dr_dk3[ki,j] = (r[nS//2,ki,j] - r[nS//2,ki,j-1]) / (K3[j]-K3[j-1])
    vmax = np.percentile(np.abs(dr_dk3), 90)
    ax8.pcolormesh(K3g, K2g, dr_dk3, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax8.contour(K3g, K2g, dr_dk3, levels=[0], colors=["k"], linewidths=[0.8])
    ax8.set_xlabel("$K_3$"); ax8.set_ylabel("$K_2$")
    fig8.suptitle("$\\partial r/\\partial K_3$ map", fontsize=8, y=0.98)
    _save(fig8, "fig8_reentrant")

# ── S26: Analytic phase diagrams ──
def s_analytic_phase():
    try:
        d = json.load(open("analytical_phase_diagram.json"))
    except: return
    K2, K3 = np.array(d["K2_range"]), np.array(d["K3_range"])
    sr = d["sigma_results"]
    keys = sorted(sr.keys(), key=lambda s: float(s.split("=")[1]))
    nS = len(keys)
    ncols = min(3, nS); nrows = (nS+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows), gridspec_kw={"hspace":0.35, "wspace":0.2})
    axes = axes.flatten()
    K3g, K2g = np.meshgrid(K3, K2)
    for i, sk in enumerate(keys):
        r_star = np.array(sr[sk]["r"])
        axes[i].pcolormesh(K3g, K2g, r_star, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
        sigma = float(sk.split("=")[1])
        axes[i].set_title(f"$\\sigma={sigma:.1f}$", fontsize=6)
        if i>=nrows*(ncols-1): axes[i].set_xlabel("$K_3$")
        if i%ncols==0: axes[i].set_ylabel("$K_2$")
    for j in range(nS, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Analytic phase diagrams (OA)", fontsize=8, y=0.98)
    _save(fig, "analytic_phase_diagrams")

# ── S27: Numerical phase K2K3 ──
def s_num_phase():
    d = json.load(open("scan_K2_K3.json"))
    K2, K3, r, basin = np.array(d["K2_list"]), np.array(d["K3_list"]), np.array(d["r"]), np.array(d["basin_prob"])
    K3g, K2g = np.meshgrid(K3, K2)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), gridspec_kw={"wspace":0.25})
    im1 = axes[0].pcolormesh(K3g, K2g, r, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
    fig.colorbar(im1, ax=axes[0], shrink=0.85, label="$r$"); axes[0].set_ylabel("$K_2$"); _label(axes[0],"a")
    im2 = axes[1].pcolormesh(K3g, K2g, basin, cmap="RdBu", vmin=0, vmax=1, shading="auto", rasterized=True)
    fig.colorbar(im2, ax=axes[1], shrink=0.85, label="Basin"); _label(axes[1],"b")
    for ax in axes: ax.set_xlabel("$K_3$")
    fig.suptitle(f"Numerical phase diagram ($N={d['N']}$, $\\sigma={d['sigma']}$)", fontsize=8, y=0.98)
    _save(fig, "numerical_phase_K2K3")

# ── S28: Finite size large N ──
def s_finite_large():
    d = json.load(open("finite_size_large_N.json"))
    results = d["results"]
    N_list = [r["N"] for r in results]
    delta = [r["K3=1.0"]-r["K3=0.0"] for r in results]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(N_list, delta, "ko-", ms=4, lw=1.2)
    ax.axhline(0, color="0.7", ls=":", lw=0.3)
    ax.set_xscale("log"); ax.set_xlabel("$N$"); ax.set_ylabel("$\\Delta r$")
    fig.suptitle("Finite-size ($N=200$-$5000$)", fontsize=8, y=0.98)
    _save(fig, "finite_size_large_N")

# ── S29: SUMMARY ──
def s_summary():
    # Just copy — summary is a composite, better regenerated from plot_summary.py
    pass


if __name__ == "__main__":
    print("Generating supplementary figures...")
    s_benchmark(); s_oa_verification(); s_K4(); s_cluster()
    s_gaussian(); s_budget(); s_freq(); s_locking_order()
    s_oa_deviation(); s_Z2(); s_corrected(); s_hyst_fine()
    s_finite_size(); s_sigma_K3(); s_dist_direction()
    s_help_hurt(); s_Kc_K3(); s_3d_suite(); s_analytic_phase()
    s_num_phase(); s_finite_large()
    print("Done.")
