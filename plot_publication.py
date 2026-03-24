"""
论文级科研绘图 — 高阶 Kuramoto 相图可视化
HaibiPlotAnalyst (agent-mn4l6mgq)

支持功能：
1. K₂×K₃ 相图矩阵（不同 σ 切片）— 序参量 r + 吸引域概率
2. σ 效应曲线 — 固定 K₂ 下 r vs K₃
3. 临界耦合偏移 — Kc vs σ（不同 K₃）
4. 三指标联合面板 — r / basin / convergence time
5. Basin vs Stability 散点图（"坑口窄但坑底深"）
6. K₃ 正负不对称性分析（KuramotoThinker 建议）
7. 解析-数值对比图（叠加 Ott-Antonsen 预测，对接 MathAgent）
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
import os

# ─── 全局论文风格 ───
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
    "axes.grid": False,
})

CMAP_R = "RdYlGn"
CMAP_BASIN = "YlGnBu"
CMAP_TC = "YlOrRd_r"
FIG_DIR = "figures"


# ─── Ott-Antonsen 解析工具（来自 MathAgent 推导） ───
def oa_delta_gaussian(sigma):
    """Gaussian 频率分布的等效半宽 Δ = σ√(2π)/π ≈ 0.798σ
    来自 g(0)匹配: Δ = 1/(π·g(0)), g(0) = 1/(σ√(2π))
    => Δ = σ√(2π)/π  (MathAgent commit df5281f 修正)
    """
    return sigma * np.sqrt(2 * np.pi) / np.pi


def oa_Kc_onset(sigma):
    """同步起始临界值 K₂c = 2Δ (K₃ 无关，leading order)"""
    return 2.0 * oa_delta_gaussian(sigma)


def oa_fixed_point_eq(r2, K2, K3, Delta):
    """不动点方程残差: Δ - (1-r²)/2*(K₂+K₃r²) = 0"""
    return Delta - (1.0 - r2) / 2.0 * (K2 + K3 * r2)


def oa_phase_boundary(K3_arr, sigma, r_thresh=0.01):
    """
    给定 K₃ 数组和 σ，求解析相边界 K₂c(K₃)
    在 r=r_thresh 处线性化: K₂c ≈ 2Δ - K₃*r_thresh²
    更精确: 解不动点方程 Δ = (1-r²)/2*(K₂+K₃r²) 在 r→0+ 极限
    K₂c = 2Δ (与 K₃ 无关 at onset)
    """
    Delta = oa_delta_gaussian(sigma)
    # Leading order: Kc = 2Δ 不依赖 K₃
    # 但对于有限 r 的等值线，K₃ 会有影响
    # 对 r=0.5 等值线求解: Δ = (1-r²)/2*(K₂+K₃r²)
    # => K₂ = 2Δ/(1-r²) - K₃r²
    r2 = r_thresh ** 2
    return 2.0 * Delta / (1.0 - r2) - K3_arr * r2


def oa_explosive_boundary(K2_arr):
    """爆炸式同步边界: K₃ = K₂ (超临界→亚临界转变)"""
    return K2_arr.copy()


def oa_reentrant_r_star(K2, K3, Delta):
    """
    求给定 (K₂, K₃) 下的同步不动点 r*
    解: Δ = (1-r²)/2·(K₂+K₃r²)  =>  K₃r⁴ - (K₂+K₃)r² + (K₂-2Δ) = 0
    """
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


def oa_dr_dK3(K2, K3, sigma):
    """
    dr*/dK₃ at fixed K₂ (MathAgent Eq. reentrant)
    = r*²(1-r*²) / [2·((K₂+3K₃r*²)(1-r*²) - 2r*²(K₂+K₃r*²))]
    Sign change marks re-entrant boundary
    """
    Delta = oa_delta_gaussian(sigma)
    r_star = oa_reentrant_r_star(K2, K3, Delta)
    if r_star < 0.01:
        return 0.0
    r2 = r_star ** 2
    numer = r2 * (1 - r2)
    denom = 2 * ((K2 + 3 * K3 * r2) * (1 - r2) - 2 * r2 * (K2 + K3 * r2))
    if abs(denom) < 1e-12:
        return np.inf
    return numer / denom


def load_data(path="scan_sigma_K2_K3.json"):
    """加载扫描数据，兼容 demo 和正式数据"""
    with open(path) as f:
        data = json.load(f)
    return {
        "sigma": np.array(data["sigma_list"]),
        "K2": np.array(data["K2_list"]),
        "K3": np.array(data["K3_list"]),
        "r": np.array(data["r"]),
        "tc": np.array(data["tc"]),
        "basin": np.array(data["basin"]),
        "N": data.get("N", 200),
        "T": data.get("T", 100),
    }


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _add_panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")


# ═══════════════════════════════════════════════════════════
# Fig 1: K₂×K₃ 相图矩阵
# ═══════════════════════════════════════════════════════════
def fig1_phase_diagram_matrix(d, save=True):
    """
    对每个 σ 画 K₂×K₃ 热力图
    上行: 序参量 r    下行: 吸引域概率 basin
    """
    _ensure_dir()
    sigma, K2, K3 = d["sigma"], d["K2"], d["K3"]
    r, basin = d["r"], d["basin"]
    nS = len(sigma)

    fig, axes = plt.subplots(2, nS, figsize=(3.2 * nS + 1, 6.5),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.15})
    if nS == 1:
        axes = axes.reshape(2, 1)

    K3g, K2g = np.meshgrid(K3, K2)

    for i, sig in enumerate(sigma):
        # --- r ---
        ax = axes[0, i]
        im = ax.pcolormesh(K3g, K2g, r[i], cmap=CMAP_R, vmin=0, vmax=1,
                           shading="auto", rasterized=True)
        # 相边界等值线
        ax.contour(K3g, K2g, r[i], levels=[0.3, 0.5, 0.7],
                   colors=["0.2"], linewidths=[0.6, 1.0, 0.6],
                   linestyles=["--", "-", "--"])
        # OA 临界线 (r=0.5 等值线)
        k3_fine = np.linspace(K3.min(), K3.max(), 200)
        Kc_oa = oa_phase_boundary(k3_fine, sig, r_thresh=0.5)
        mask_oa = (Kc_oa >= K2.min()) & (Kc_oa <= K2.max())
        if mask_oa.any():
            ax.plot(k3_fine[mask_oa], Kc_oa[mask_oa], "w--", lw=1.2, alpha=0.9)

        # 爆炸式边界 K₃ = K₂
        k2_fine = np.linspace(K2.min(), K2.max(), 100)
        mask_ex = (k2_fine >= K3.min()) & (k2_fine <= K3.max())
        if mask_ex.any():
            ax.plot(k2_fine[mask_ex], k2_fine[mask_ex], "w:", lw=0.8, alpha=0.7)

        # Kc onset 标注
        Kc = oa_Kc_onset(sig)
        if Kc <= K2.max():
            ax.axhline(Kc, color="white", ls=":", lw=0.6, alpha=0.5)
            ax.text(K3[-1] * 0.85, Kc + 0.1, f"$K_c$={Kc:.1f}",
                    color="white", fontsize=7, ha="right")
        ax.set_title(f"$\\sigma={sig:.1f}$")
        ax.set_xlabel("$K_3$")
        if i == 0:
            ax.set_ylabel("$K_2$")
            _add_panel_label(ax, chr(65 + i))  # A, B, C...
        else:
            ax.set_yticklabels([])

        # --- basin ---
        ax2 = axes[1, i]
        im2 = ax2.pcolormesh(K3g, K2g, basin[i], cmap=CMAP_BASIN, vmin=0, vmax=1,
                              shading="auto", rasterized=True)
        ax2.contour(K3g, K2g, basin[i], levels=[0.3, 0.5, 0.7],
                    colors=["0.3"], linewidths=[0.6, 1.0, 0.6],
                    linestyles=["--", "-", "--"])
        ax2.set_xlabel("$K_3$")
        if i == 0:
            ax2.set_ylabel("$K_2$")
        else:
            ax2.set_yticklabels([])

    # Colorbars
    fig.colorbar(im, ax=axes[0, :].tolist(), label="Order parameter $r$",
                 shrink=0.85, pad=0.02)
    fig.colorbar(im2, ax=axes[1, :].tolist(), label="Basin probability",
                 shrink=0.85, pad=0.02)

    fig.suptitle(f"Higher-order Kuramoto: $K_2 \\times K_3$ phase diagrams ($N={d['N']}$)",
                 fontsize=13, y=1.02)

    if save:
        path = os.path.join(FIG_DIR, "fig1_phase_diagram_matrix.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 2: σ 效应曲线
# ═══════════════════════════════════════════════════════════
def fig2_sigma_effect(d, K2_targets=None, save=True):
    """
    固定若干 K₂ 值，画 r vs K₃ 曲线（不同 σ 用不同颜色）
    """
    _ensure_dir()
    sigma, K2, K3, r = d["sigma"], d["K2"], d["K3"], d["r"]

    if K2_targets is None:
        # 取 3 个代表性 K₂：弱/中/强
        idxs = [len(K2) // 4, len(K2) // 2, 3 * len(K2) // 4]
    else:
        idxs = [np.argmin(np.abs(K2 - t)) for t in K2_targets]

    n_panels = len(idxs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4),
                              sharey=True, gridspec_kw={"wspace": 0.08})
    if n_panels == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(sigma)))

    for pi, (ax, ki) in enumerate(zip(axes, idxs)):
        K2_val = K2[ki]
        for si, (sig, c) in enumerate(zip(sigma, colors)):
            ax.plot(K3, r[si, ki, :], color=c, lw=1.8,
                    label=f"$\\sigma={sig:.1f}$", marker="o", ms=3)

        ax.axvline(0, color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.axhline(0.5, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("$K_3$")
        ax.set_title(f"$K_2 = {K2_val:.2f}$")
        _add_panel_label(ax, chr(65 + pi))

        if pi == 0:
            ax.set_ylabel("Order parameter $r$")
        if pi == n_panels - 1:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Effect of $K_3$ on synchronization at different $\\sigma$",
                 fontsize=13, y=1.02)

    if save:
        path = os.path.join(FIG_DIR, "fig2_sigma_effect.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 3: 临界耦合偏移 Kc vs σ
# ═══════════════════════════════════════════════════════════
def fig3_Kc_shift(d, r_thresh=0.3, save=True):
    """
    对每个 σ 和若干 K₃ 值，找临界 K₂（r 跨越阈值的点）
    与理论值 Kc = 2√(2π)σ 对比
    """
    _ensure_dir()
    sigma, K2, K3, r = d["sigma"], d["K2"], d["K3"], d["r"]

    def find_Kc(r_slice):
        for j in range(len(K2) - 1):
            if r_slice[j] < r_thresh <= r_slice[j + 1]:
                # 线性插值
                frac = (r_thresh - r_slice[j]) / (r_slice[j + 1] - r_slice[j])
                return K2[j] + frac * (K2[j + 1] - K2[j])
        return np.nan

    # 选几个 K₃ 值
    k3_targets = [0.0, 0.5, 1.0, -0.5, -1.0]
    k3_idxs = {t: np.argmin(np.abs(K3 - t)) for t in k3_targets}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 理论值
    sig_fine = np.linspace(sigma.min(), sigma.max(), 100)
    Kc_th = 2.0 * sig_fine * np.sqrt(2 * np.pi) / np.pi  # corrected: Δ=σ√(2π)/π
    ax.plot(sig_fine, Kc_th, "k--", lw=2, label="Theory ($K_3=0$)", zorder=10)

    markers = ["o", "s", "D", "^", "v"]
    colors_k3 = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(k3_targets)))

    for ti, (k3t, mk, clr) in enumerate(zip(k3_targets, markers, colors_k3)):
        ki = k3_idxs[k3t]
        Kc_vals = [find_Kc(r[si, :, ki]) for si in range(len(sigma))]
        ax.plot(sigma, Kc_vals, marker=mk, color=clr, lw=1.5, ms=6,
                label=f"$K_3 = {k3t:+.1f}$")

    ax.set_xlabel("$\\sigma$ (frequency spread)")
    ax.set_ylabel("$K_c$ (critical coupling)")
    ax.set_title("Critical coupling shift by higher-order interactions")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.15)

    if save:
        path = os.path.join(FIG_DIR, "fig3_Kc_shift.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 4: 三指标联合面板（固定 σ）
# ═══════════════════════════════════════════════════════════
def fig4_triple_metric(d, sigma_idx=None, save=True):
    """
    固定一个 σ，三列：r / basin / convergence time
    """
    _ensure_dir()
    sigma, K2, K3 = d["sigma"], d["K2"], d["K3"]
    r, basin, tc = d["r"], d["basin"], d["tc"]

    if sigma_idx is None:
        sigma_idx = len(sigma) // 2  # 取中间

    sig = sigma[sigma_idx]
    K3g, K2g = np.meshgrid(K3, K2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                              gridspec_kw={"wspace": 0.25})

    titles = ["Order parameter $r$", "Basin probability", "Convergence time"]
    data_list = [r[sigma_idx], basin[sigma_idx], tc[sigma_idx]]
    cmaps = [CMAP_R, CMAP_BASIN, CMAP_TC]
    labels = ["$r$", "Probability", "Time"]

    for pi, (ax, mat, cmap, title, lab) in enumerate(
            zip(axes, data_list, cmaps, titles, labels)):
        vmin = 0
        vmax = 1 if pi < 2 else np.percentile(mat, 95)
        im = ax.pcolormesh(K3g, K2g, mat, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        if pi < 2:
            ax.contour(K3g, K2g, mat, levels=[0.3, 0.5, 0.7],
                       colors=["0.2"], linewidths=[0.6, 1.0, 0.6],
                       linestyles=["--", "-", "--"])
        ax.set_xlabel("$K_3$")
        if pi == 0:
            ax.set_ylabel("$K_2$")
        ax.set_title(title)
        _add_panel_label(ax, chr(65 + pi))
        fig.colorbar(im, ax=ax, label=lab, shrink=0.9)

    fig.suptitle(f"Three synchronization metrics ($\\sigma = {sig:.1f}$, $N={d['N']}$)",
                 fontsize=13, y=1.03)

    if save:
        path = os.path.join(FIG_DIR, "fig4_triple_metric.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 5: "坑口窄但坑底深" 示意图
# ═══════════════════════════════════════════════════════════
def fig5_basin_vs_stability(d, save=True):
    """
    散点图：basin probability vs r，每个点是一个 (K₂, K₃) 组合
    不同 σ 不同颜色，揭示 "更难达到但更稳定" 的现象
    """
    _ensure_dir()
    sigma, r, basin = d["sigma"], d["r"], d["basin"]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigma)))

    for si, (sig, c) in enumerate(zip(sigma, colors)):
        r_flat = r[si].flatten()
        b_flat = basin[si].flatten()
        # 只画有同步倾向的点
        mask = r_flat > 0.1
        ax.scatter(b_flat[mask], r_flat[mask], c=[c], s=12, alpha=0.5,
                   label=f"$\\sigma={sig:.1f}$", edgecolors="none")

    ax.set_xlabel("Basin probability (accessibility)")
    ax.set_ylabel("Order parameter $r$ (stability)")
    ax.set_title("Accessibility vs. Stability of synchronization")
    ax.legend(markerscale=2, framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 标注四象限
    ax.text(0.75, 0.15, "Easy but\nunstable", ha="center", fontsize=8,
            color="0.5", style="italic")
    ax.text(0.15, 0.85, "Hard but\nstable", ha="center", fontsize=8,
            color="0.5", style="italic")
    ax.text(0.75, 0.85, "Easy &\nstable", ha="center", fontsize=8,
            color="0.4", fontweight="bold")
    ax.text(0.15, 0.15, "Hard &\nunstable", ha="center", fontsize=8,
            color="0.5", style="italic")

    ax.axhline(0.5, color="gray", ls=":", lw=0.5)
    ax.axvline(0.5, color="gray", ls=":", lw=0.5)
    ax.grid(True, alpha=0.1)

    if save:
        path = os.path.join(FIG_DIR, "fig5_basin_vs_stability.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 6: K₃ 正负不对称性分析（KuramotoThinker 建议）
# ═══════════════════════════════════════════════════════════
def fig6_asymmetry(d, save=True):
    """
    对每个 σ，计算 Δr(K₃) = r(+|K₃|) - r(-|K₃|) 的不对称性
    揭示正负 K₃ 对同步的不同效应
    """
    _ensure_dir()
    sigma, K2, K3, r = d["sigma"], d["K2"], d["K3"], d["r"]

    # 构建正负 K₃ 对
    k3_pos_mask = K3 > 0
    k3_neg = K3[K3 < 0]
    k3_pos = K3[k3_pos_mask]
    n_pairs = min(len(k3_neg), len(k3_pos))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.3})

    # Panel A: 不对称性热力图（固定中间 σ）
    ax = axes[0]
    si = len(sigma) // 2
    sig = sigma[si]

    # Δr = r(K₃>0) - r(K₃<0) for matched |K₃|
    asym = np.zeros((len(K2), n_pairs))
    k3_vals = []
    for p in range(n_pairs):
        pi = len(K3) - 1 - p  # positive side from right
        ni = p               # negative side from left
        if abs(abs(K3[pi]) - abs(K3[ni])) < 0.01:
            asym[:, p] = r[si, :, pi] - r[si, :, ni]
            k3_vals.append(abs(K3[pi]))

    if k3_vals:
        k3v = np.array(k3_vals)
        K3ag, K2ag = np.meshgrid(k3v, K2)
        vmax = max(abs(asym.min()), abs(asym.max()), 0.01)
        im = ax.pcolormesh(K3ag, K2ag, asym, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.contour(K3ag, K2ag, asym, levels=[0], colors=["k"], linewidths=[1.2])
        fig.colorbar(im, ax=ax, label="$\\Delta r = r(+K_3) - r(-K_3)$", shrink=0.9)
    ax.set_xlabel("$|K_3|$")
    ax.set_ylabel("$K_2$")
    ax.set_title(f"$K_3$ asymmetry ($\\sigma={sig:.1f}$)")
    _add_panel_label(ax, "A")

    # Panel B: 各 σ 下的平均不对称性曲线
    ax2 = axes[1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(sigma)))
    for si2, (sig2, c) in enumerate(zip(sigma, colors)):
        mean_asym = []
        for p in range(n_pairs):
            pi = len(K3) - 1 - p
            ni = p
            if abs(abs(K3[pi]) - abs(K3[ni])) < 0.01:
                mean_asym.append(np.mean(r[si2, :, pi] - r[si2, :, ni]))
        if mean_asym and k3_vals:
            ax2.plot(k3_vals[:len(mean_asym)], mean_asym, color=c, lw=1.8,
                     marker="o", ms=4, label=f"$\\sigma={sig2:.1f}$")

    ax2.axhline(0, color="gray", ls="--", lw=0.6)
    ax2.set_xlabel("$|K_3|$")
    ax2.set_ylabel("Mean $\\Delta r$")
    ax2.set_title("Average $K_3$ asymmetry across $K_2$")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.15)
    _add_panel_label(ax2, "B")

    if save:
        path = os.path.join(FIG_DIR, "fig6_asymmetry.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 7: 解析-数值对比（对接 MathAgent Ott-Antonsen 结果）
# ═══════════════════════════════════════════════════════════
def fig7_analytic_vs_numeric(d, analytic_Kc_func=None, save=True):
    """
    将 Ott-Antonsen 解析临界曲线叠加到数值相图上
    analytic_Kc_func: callable(K3, sigma) -> Kc 解析值
                      如果 None 则用 Ott-Antonsen 推导结果
    """
    _ensure_dir()
    sigma, K2, K3, r = d["sigma"], d["K2"], d["K3"], d["r"]

    nS = len(sigma)
    fig, axes = plt.subplots(1, nS, figsize=(3.5 * nS, 4),
                              sharey=True, gridspec_kw={"wspace": 0.08})
    if nS == 1:
        axes = [axes]

    K3g, K2g = np.meshgrid(K3, K2)

    for i, (ax, sig) in enumerate(zip(axes, sigma)):
        ax.pcolormesh(K3g, K2g, r[i], cmap=CMAP_R, vmin=0, vmax=1,
                      shading="auto", rasterized=True)
        ax.contour(K3g, K2g, r[i], levels=[0.5],
                   colors=["white"], linewidths=[1.5], linestyles=["-"])

        # 解析曲线 (Ott-Antonsen)
        k3_fine = np.linspace(K3.min(), K3.max(), 200)
        if analytic_Kc_func is not None:
            Kc_analytic = [analytic_Kc_func(k, sig) for k in k3_fine]
        else:
            # Ott-Antonsen: r=0.5 等值线
            Kc_analytic = oa_phase_boundary(k3_fine, sig, r_thresh=0.5)

        ax.plot(k3_fine, Kc_analytic, "w--", lw=2, label="OA (r=0.5)")

        # 爆炸式同步边界 K₃ = K₂
        k2_fine = np.linspace(K2.min(), K2.max(), 100)
        k3_explosive = oa_explosive_boundary(k2_fine)
        mask = (k3_explosive >= K3.min()) & (k3_explosive <= K3.max())
        if mask.any():
            ax.plot(k3_explosive[mask], k2_fine[mask], "w:",
                    lw=1.5, label="Explosive" if i == 0 else None)
        ax.set_xlabel("$K_3$")
        ax.set_title(f"$\\sigma={sig:.1f}$")
        if i == 0:
            ax.set_ylabel("$K_2$")
            ax.legend(loc="lower right", fontsize=8)
        _add_panel_label(ax, chr(65 + i))

    fig.suptitle("Analytic (Ott-Antonsen) vs. Numerical phase boundary",
                 fontsize=13, y=1.02)

    if save:
        path = os.path.join(FIG_DIR, "fig7_analytic_vs_numeric.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# Fig 8: Re-entrant 同步区域 (MathAgent Eq. reentrant)
# ═══════════════════════════════════════════════════════════
def fig8_reentrant(d, save=True):
    """
    在 K₂×K₃ 平面上标注 dr*/dK₃ 的符号变化
    正值区域: K₃ 增强同步; 负值区域: K₃ 抑制同步
    零等值线: re-entrant 边界
    """
    _ensure_dir()
    sigma, K2, K3 = d["sigma"], d["K2"], d["K3"]
    r = d["r"]

    # 选中间 σ
    si = len(sigma) // 2
    sig = sigma[si]

    K3g, K2g = np.meshgrid(K3, K2)
    dr_dk3 = np.zeros_like(K3g)

    for i in range(len(K2)):
        for j in range(len(K3)):
            dr_dk3[i, j] = oa_dr_dK3(K2[i], K3[j], sig)

    # clip extreme values for visualization
    vmax = np.percentile(np.abs(dr_dk3[np.isfinite(dr_dk3)]), 95)
    dr_dk3 = np.clip(dr_dk3, -vmax, vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.3})

    # Panel A: dr*/dK₃ heatmap
    ax = axes[0]
    im = ax.pcolormesh(K3g, K2g, dr_dk3, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax.contour(K3g, K2g, dr_dk3, levels=[0], colors=["k"], linewidths=[1.5])
    # explosive boundary
    k2_line = np.linspace(K2.min(), K2.max(), 100)
    mask_ex = (k2_line >= K3.min()) & (k2_line <= K3.max())
    if mask_ex.any():
        ax.plot(k2_line[mask_ex], k2_line[mask_ex], "w:", lw=1, alpha=0.7,
                label="Explosive: $K_3=K_2$")
    fig.colorbar(im, ax=ax, label="$\\partial r^*/\\partial K_3$", shrink=0.9)
    ax.set_xlabel("$K_3$")
    ax.set_ylabel("$K_2$")
    ax.set_title(f"Re-entrant map ($\\sigma={sig:.1f}$)")
    ax.legend(fontsize=8, loc="lower right")
    _add_panel_label(ax, "A")

    # Panel B: numerical r overlaid with analytic boundaries
    ax2 = axes[1]
    ax2.pcolormesh(K3g, K2g, r[si], cmap=CMAP_R, vmin=0, vmax=1,
                   shading="auto", rasterized=True)
    # r=0.5 numerical contour
    ax2.contour(K3g, K2g, r[si], levels=[0.5],
                colors=["white"], linewidths=[1.5])
    # re-entrant boundary (dr*/dK₃ = 0)
    ax2.contour(K3g, K2g, dr_dk3, levels=[0],
                colors=["yellow"], linewidths=[1.5], linestyles=["--"])
    # OA onset
    Kc = oa_Kc_onset(sig)
    ax2.axhline(Kc, color="cyan", ls=":", lw=1, alpha=0.8)

    ax2.set_xlabel("$K_3$")
    ax2.set_ylabel("$K_2$")
    ax2.set_title(f"Numerical $r$ + analytic boundaries ($\\sigma={sig:.1f}$)")
    _add_panel_label(ax2, "B")

    fig.suptitle("Re-entrant synchronization: $K_3$ first helps then hinders",
                 fontsize=13, y=1.02)

    if save:
        path = os.path.join(FIG_DIR, "fig8_reentrant.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        print(f"Saved: {path}")
    return fig


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
def generate_all(data_path="scan_sigma_K2_K3.json"):
    """一键生成全部论文图"""
    d = load_data(data_path)
    print(f"Data: σ={len(d['sigma'])}, K₂={len(d['K2'])}, K₃={len(d['K3'])}, N={d['N']}")

    fig1_phase_diagram_matrix(d)
    fig2_sigma_effect(d)
    fig3_Kc_shift(d)
    fig4_triple_metric(d)
    fig5_basin_vs_stability(d)
    fig6_asymmetry(d)
    fig7_analytic_vs_numeric(d)
    fig8_reentrant(d)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "scan_sigma_K2_K3.json"
    generate_all(path)
