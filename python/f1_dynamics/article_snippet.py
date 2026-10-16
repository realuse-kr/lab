import numpy as np
from scipy import signal
from scipy.optimize import brentq

zeta, wn = 0.32, 11.0
plant = signal.TransferFunction([wn**2], [1, 2 * zeta * wn, wn**2])
yss = plant.num[-1] / plant.den[-1]      # 정상값은 DC 이득으로 잡는다


def gain_db(w):
    return 20 * np.log10(abs(signal.freqresp(plant, [w])[1][0]))


# 대역폭: DC 이득보다 3 dB 낮아지는 첫 각주파수 [rad/s]
bw = brentq(lambda w: gain_db(w) - (gain_db(1e-6) - 3.0), 1e-3, 1e3)

t = np.linspace(0, 6, 60001)
_, y = signal.step(plant, T=t)


def crossing(level):                      # level(0~1)에 처음 닿는 시각, 선형 보간
    i = np.argmax(y >= level * yss)
    return np.interp(level * yss, y[i - 1:i + 1], t[i - 1:i + 1])


rise = crossing(0.9) - crossing(0.1)                     # 10%~90% 상승시간
outside = np.where(abs(y - yss) > 0.02 * yss)[0]
settle = t[outside[-1] + 1]                              # ±2% 정정시간
overshoot = 100 * (y.max() / yss - 1)

print(f"대역폭 {bw:.2f} rad/s = {bw / (2 * np.pi):.2f} Hz")
print(f"상승시간 {rise:.4f} s, 정정시간 {settle:.4f} s, 오버슈트 {overshoot:.2f} %")
