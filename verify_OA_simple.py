"""极简OA验证"""
import numpy as np

Delta = np.sqrt(2*np.pi) / np.pi
Kc = 2 * Delta
print(f"Delta={Delta:.4f}, Kc={Kc:.4f}")

def f(r, K2, K3):
    return -Delta*r + r*(1-r**2)/2*(K2+K3*r**2)

def steady(K2, K3, r0=0.01):
    r = r0
    for _ in range(100000):
        r += 0.01 * f(r, K2, K3)
        r = max(min(r, 1.0), 0.0)
    return r

# 经典
print("\nK3=0:")
for K2 in [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]:
    print(f"  K2={K2:.1f} r*={steady(K2,0):.4f}")

# 重入测试
print(f"\nK2=1.1*Kc={1.1*Kc:.3f}, scan K3:")
for K3 in [0,0.5,1,2,3,5,10,20,50]:
    r_low = steady(1.1*Kc, K3, r0=0.01)
    r_high = steady(1.1*Kc, K3, r0=0.99)
    print(f"  K3={K3:5.1f} r*(from 0)={r_low:.4f} r*(from 1)={r_high:.4f}")
