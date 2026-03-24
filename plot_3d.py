"""
从 scan_sigma_K2_K3.json 生成相图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def load_data(path="scan_sigma_K2_K3.json"):
    with open(path) as f:
        data = json.load(f)
    return (
        np.array(data["sigma_list"]),
        np.array(data["K2_list"]),
        np.array(data["K3_list"]),
        np.array(data["r"]),
        np.array(data["tc"]),
        np.array(data["basin"]),
    )


def plot_K2_K3_slices(sigma_list, K2_list, K3_list, r, basin, out_dir="figures"):
    """
    对每个 σ 画 K₂×K₃ 相图（r 和 basin）
    """
    os.makedirs(out_dir, exist_ok=True)
    nS = len(sigma_list)

    fig, axes = plt.subplots(2, nS, figsize=(4 * nS, 8))
    if nS == 1:
        axes = axes.reshape(2, 1)

    K2_grid, K3_grid = np.meshgrid(K2_list, K3_list, indexing="ij")

    for i, sigma in enumerate(sigma_list):
        # 序参量 r
        ax = axes[0, i]
        im = ax.pcolormesh(K3_grid, K2_grid, r[i], cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"r  (σ={sigma:.2f})", fontsize=11)
        ax.set_xlabel("K₃")
        if i == 0:
            ax.set_ylabel("K₂")
        plt.colorbar(im, ax=ax)

        # 吸引域概率
        ax = axes[1, i]
        im2 = ax.pcolormesh(K3_grid, K2_grid, basin[i], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"basin  (σ={sigma:.2f})", fontsize=11)
        ax.set_xlabel("K₃")
        if i == 0:
            ax.set_ylabel("K₂")
        plt.colorbar(im2, ax=ax)

    plt.suptitle("高阶 Kuramoto：K₂×K₃ 相图（不同 σ）", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "phase_diagrams.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存：{path}")


def plot_sigma_effect(sigma_list, K2_list, K3_list, r, out_dir="figures"):
    """
    固定 K₂ 在中等值，画 r 随 K₃ 变化的曲线（不同 σ）
    """
    os.makedirs(out_dir, exist_ok=True)
    K2_mid_idx = len(K2_list) // 2
    K2_mid = K2_list[K2_mid_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_list)))

    for i, (sigma, color) in enumerate(zip(sigma_list, colors)):
        ax.plot(K3_list, r[i, K2_mid_idx, :], color=color, lw=2, label=f"σ={sigma:.2f}")

    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("K₃（三体耦合强度）")
    ax.set_ylabel("稳态序参量 r")
    ax.set_title(f"K₃ 对同步的影响（K₂={K2_mid:.2f}，不同 σ）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "sigma_effect.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存：{path}")


def plot_Kc_shift(sigma_list, K2_list, K3_list, r, out_dir="figures"):
    """
    对每个 σ，画 Kc（K₃=0 时）与理论值对比
    并展示不同 K₃ 下 Kc 的偏移
    """
    os.makedirs(out_dir, exist_ok=True)
    k3_zero_idx = np.argmin(np.abs(K3_list))
    k3_pos_idx = np.argmin(np.abs(K3_list - 0.5))
    k3_neg_idx = np.argmin(np.abs(K3_list + 0.5))

    Kc_theory = 2.0 * np.sqrt(2 * np.pi) / np.pi * sigma_list

    def find_Kc(r_slice, K2_list, thresh=0.3):
        for j in range(len(K2_list) - 1):
            if r_slice[j] < thresh <= r_slice[j + 1]:
                return 0.5 * (K2_list[j] + K2_list[j + 1])
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["K₃≈0", "K₃≈+0.5", "K₃≈-0.5", "理论值(K₃=0)"]
    styles = ["-o", "-s", "-^", "--k"]
    datasets = [
        [find_Kc(r[i, :, k3_zero_idx], K2_list) for i in range(len(sigma_list))],
        [find_Kc(r[i, :, k3_pos_idx], K2_list) for i in range(len(sigma_list))],
        [find_Kc(r[i, :, k3_neg_idx], K2_list) for i in range(len(sigma_list))],
        list(Kc_theory),
    ]

    for data, label, style in zip(datasets, labels, styles):
        valid = [(s, d) for s, d in zip(sigma_list, data) if d is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, style, label=label, lw=2)

    ax.set_xlabel("σ（频率分布宽度）")
    ax.set_ylabel("临界耦合强度 Kc")
    ax.set_title("σ 对临界耦合强度的影响（K₃ 的调制效果）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "Kc_vs_sigma.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存：{path}")


if __name__ == "__main__":
    sigma_list, K2_list, K3_list, r, tc, basin = load_data()

    print(f"数据维度: σ={len(sigma_list)}, K₂={len(K2_list)}, K₃={len(K3_list)}")

    plot_K2_K3_slices(sigma_list, K2_list, K3_list, r, basin)
    plot_sigma_effect(sigma_list, K2_list, K3_list, r)
    plot_Kc_shift(sigma_list, K2_list, K3_list, r)

    print("\n所有图已保存至 figures/")
