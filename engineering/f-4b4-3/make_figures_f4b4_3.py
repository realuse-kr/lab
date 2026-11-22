"""F-4B4.3 그림: 계산 시간 tc에 따른 위상 여유(왼쪽)와 샘플링 주기 T 두 값에서 tc = 0, T 비교(오른쪽)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from phase_margin_compute_time import margin_compute, margin_disc, crossover_and_margin, loop_base, W_HI

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
BG = "#fcfcfb"
wc0, pm0 = crossover_and_margin(loop_base, W_HI)
fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=100, facecolor=BG)
for a in ax:
    a.set_facecolor(BG)
T = 0.020
tcs = np.linspace(0, T, 41)
ax[0].plot(tcs * 1e3, [margin_compute(T, t)[1] for t in tcs], color="#1f4e79", lw=2.5, label="ZOH 응답 + 지연 tc")
ax[0].plot(tcs * 1e3, [margin_disc(T, t)[1] for t in tcs], color="#b5541c", lw=2.5, ls="--", label="이산 루프")
ax[0].plot(tcs * 1e3, pm0 - np.degrees(wc0 * (T / 2 + tcs)), color="#444444", lw=1.5, ls=":", label="72.8도 − wc·(T/2+tc)")
ax[0].set_title("샘플링 주기 T = 20 ms", fontsize=20)
ax[0].set_xlabel("계산 시간 tc [ms]", fontsize=16); ax[0].set_ylabel("위상 여유 [도]", fontsize=16)
ax[0].legend(fontsize=14); ax[0].grid(alpha=.3); ax[0].tick_params(labelsize=14)
Ts = np.arange(1, 21)
for tcf, name, c in ((0, "tc = 0", "#1f4e79"), (1, "tc = T", "#b5541c")):
    ax[1].plot(Ts, [margin_disc(t / 1e3, tcf * t / 1e3)[1] for t in Ts], color=c, lw=2.5, label=f"이산 루프, {name}")
ax[1].set_title("샘플링 주기 T에 따른 위상 여유", fontsize=20)
ax[1].set_xlabel("샘플링 주기 T [ms]", fontsize=16); ax[1].set_ylabel("위상 여유 [도]", fontsize=16)
ax[1].legend(fontsize=14); ax[1].grid(alpha=.3); ax[1].tick_params(labelsize=14)
fig.tight_layout()
fig.savefig("fig_tmp.png", facecolor=BG)
im = Image.open("fig_tmp.png").convert("RGB").resize((1600, 900), Image.LANCZOS)
im.save("fig_f4b4_3_compute_time_phase_margin.webp", lossless=True)
chk = Image.open("fig_f4b4_3_compute_time_phase_margin.webp")
print(chk.size, chk.mode, chk.getpixel((2, 2)))
