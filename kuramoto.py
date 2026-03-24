"""
高阶Kuramoto模型
dθᵢ/dt = ωᵢ + (K₂/N) Σⱼ sin(θⱼ−θᵢ) + (K₃/N²) ΣⱼΣₖ sin(2θⱼ−θₖ−θᵢ)

优化版：三体项通过均场分解从 O(N³) 降至 O(N)
sin(2θⱼ−θₖ−θᵢ) = sin(2θⱼ)cos(θₖ+θᵢ) − cos(2θⱼ)sin(θₖ+θᵢ)
预计算全局和 S₂,C₂,S₁,C₁ 后，每个振荡器仅需 O(1) 计算三体贡献
"""

import numpy as np
from numba import njit, prange


@njit
def dtheta(theta, omega, K2, K3, N):
    """计算每个振荡器的相位导数（优化版 O(N)）"""
    # 预计算全局均场量
    S1 = 0.0  # Σ sin(θⱼ)
    C1 = 0.0  # Σ cos(θⱼ)
    S2 = 0.0  # Σ sin(2θⱼ)
    C2 = 0.0  # Σ cos(2θⱼ)
    for j in range(N):
        S1 += np.sin(theta[j])
        C1 += np.cos(theta[j])
        S2 += np.sin(2.0 * theta[j])
        C2 += np.cos(2.0 * theta[j])

    dtheta_dt = np.empty(N)
    for i in range(N):
        si = np.sin(theta[i])
        ci = np.cos(theta[i])

        # 两两耦合项: (K₂/N) Σⱼ sin(θⱼ−θᵢ) = (K₂/N)(S1·ci − C1·si)
        pair = (K2 / N) * (S1 * ci - C1 * si)

        # 三体耦合项展开:
        # Σⱼ Σₖ sin(2θⱼ−θₖ−θᵢ)
        # = Σⱼ sin(2θⱼ) · Σₖ cos(θₖ+θᵢ) − Σⱼ cos(2θⱼ) · Σₖ sin(θₖ+θᵢ)
        # Σₖ cos(θₖ+θᵢ) = C1·ci − S1·si
        # Σₖ sin(θₖ+θᵢ) = S1·ci + C1·si
        sum_cos_ki = C1 * ci - S1 * si
        sum_sin_ki = S1 * ci + C1 * si
        triplet = (K3 / (N * N)) * (S2 * sum_cos_ki - C2 * sum_sin_ki)

        dtheta_dt[i] = omega[i] + pair + triplet
    return dtheta_dt


@njit
def rk4_step(theta, omega, K2, K3, N, dt):
    """4阶Runge-Kutta积分一步"""
    k1 = dtheta(theta, omega, K2, K3, N)
    k2 = dtheta(theta + 0.5 * dt * k1, omega, K2, K3, N)
    k3 = dtheta(theta + 0.5 * dt * k2, omega, K2, K3, N)
    k4 = dtheta(theta + dt * k3, omega, K2, K3, N)
    return theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def order_parameter(theta):
    """序参量 r = |1/N Σ e^{iθᵢ}|，衡量同步程度"""
    N = len(theta)
    mean = np.sum(np.exp(1j * theta)) / N
    return np.abs(mean), np.angle(mean)


class KuramotoHigherOrder:
    """高阶Kuramoto振荡器系统"""

    def __init__(self, N=200, sigma=1.0, K2=1.0, K3=0.0, seed=42):
        self.N = N
        self.sigma = sigma
        self.K2 = K2
        self.K3 = K3
        self.rng = np.random.default_rng(seed)

        # 随机自然频率 ωᵢ ~ N(0, σ²)
        self.omega = self.rng.normal(0, sigma, N)
        # 随机初始相位
        self.theta = self.rng.uniform(0, 2 * np.pi, N)
        self.r_history = []

    def reset_theta(self, seed=None):
        """重置随机初始相位"""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        self.theta = rng.uniform(0, 2 * np.pi, self.N)
        self.r_history = []

    def step(self, dt=0.01):
        """积分一步"""
        self.theta = rk4_step(self.theta, self.omega, self.K2, self.K3, self.N, dt)
        r, _ = order_parameter(self.theta)
        self.r_history.append(r)
        return r

    def simulate(self, T=100.0, dt=0.01, burn_in=10.0):
        """
        运行模拟
        T: 总时长
        dt: 积分步长
        burn_in: 预热期（前burn_in数据不计入统计）
        返回: 最终序参量 r, 收敛时间
        """
        steps = int(T / dt)
        burn_steps = int(burn_in / dt)
        self.r_history = []

        r = 0.0
        converge_step = steps
        for s in range(steps):
            r = self.step(dt)
            if s > burn_steps and r > 0.9 and converge_step == steps:
                converge_step = s

        # burn_in后的平均r
        r_final = np.mean(self.r_history[burn_steps:])
        return r_final, converge_step * dt

    def basin_probability(self, n_trials=100, T=100.0, dt=0.01):
        """
        估计从随机初始条件出发达到同步(r>0.9)的概率
        测量吸引域大小
        """
        successes = 0
        for t in range(n_trials):
            self.reset_theta(seed=t * 42 + 12345)
            r, _ = self.simulate(T=T, dt=dt)
            if r > 0.9:
                successes += 1
        return successes / n_trials


def verify_classic_Kc(sigma=1.0, N=200):
    """
    验证经典Kuramoto临界值
    Kc = 2 * sqrt(2π) * σ  (for Gaussian g(ω))
    """
    Kc_exact = 2.0 * np.sqrt(2 * np.pi) * sigma
    print(f"Kc exact (Gaussian): {Kc_exact:.4f}")

    Ks = np.linspace(0.5 * Kc_exact, 2.0 * Kc_exact, 10)
    for K in Ks:
        model = KuramotoHigherOrder(N=N, sigma=sigma, K2=K, K3=0.0, seed=0)
        r, _ = model.simulate(T=200.0, dt=0.01)
        print(f"  K={K:.3f} ({K/Kc_exact:.2f}*Kc) → r={r:.4f}")


if __name__ == '__main__':
    print("=== 经典Kuramoto临界值验证 (σ=1) ===")
    verify_classic_Kc(sigma=1.0, N=200)

    print("\n=== 高阶Kuramoto测试 (K₂=1.5, K₃变化) ===")
    K3s = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for K3 in K3s:
        model = KuramotoHigherOrder(N=200, sigma=1.0, K2=1.5, K3=K3, seed=0)
        r, tc = model.simulate(T=200.0, dt=0.01)
        print(f"  K₂=1.5, K₃={K3:+.1f} → r={r:.4f}, t_converge={tc:.1f}")
