import math
import numpy as np

zeta, wn, t_end = 0.32, 11.0, 2.0
sigma = zeta * wn
wd = wn * math.sqrt(1 - zeta ** 2)


def f(x, v, u=1.0):
    """우변: (x', v'). 스텝 입력 u=1."""
    return v, wn ** 2 * (u - x) - 2 * sigma * v


def euler(n, h):
    x = v = 0.0
    ys = [x]
    for _ in range(n):
        dx, dv = f(x, v)
        x, v = x + h * dx, v + h * dv
        ys.append(x)
    return np.array(ys)


def rk4(n, h):
    x = v = 0.0
    ys = [x]
    for _ in range(n):
        k1 = f(x, v)
        k2 = f(x + h / 2 * k1[0], v + h / 2 * k1[1])
        k3 = f(x + h / 2 * k2[0], v + h / 2 * k2[1])
        k4 = f(x + h * k3[0], v + h * k3[1])
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ys.append(x)
    return np.array(ys)


def exact(t):
    return 1 - np.exp(-sigma * t) * (np.cos(wd * t) + sigma / wd * np.sin(wd * t))


print("h [s]      euler       rk4")
for h in [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625]:
    n = round(t_end / h)
    t = np.arange(n + 1) * h
    e_eu = np.abs(euler(n, h) - exact(t)).max()
    e_rk = np.abs(rk4(n, h) - exact(t)).max()
    print(f"{h:8.6f}  {e_eu:.3e}  {e_rk:.3e}")
