"""F-4B5.2 그림: 왼쪽은 T = 10 ms, tc = 4 ms에서 두 출력 순서의 스텝 응답, 오른쪽은 계산 시간 tc에 따른 오버슈트."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import time_response_update_order as m

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
BG = "#fcfcfb"
BLUE, ORANGE, GRAY = "#1f4e79", "#b5541c", "#444444"
T, TC, GOAL = 0.010, 0.004, 12.0
A, B = m.ORDERS

fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=100, facecolor=BG, gridspec_kw={"width_ratios": [1.1, 1]})
for a in ax:
    a.set_facecolor(BG)

# 왼쪽: 스텝 응답
a = ax[0]
t0, y0 = m.continuous_reference(t_end=0.4)
ta, ya, _ = m.step_response(A, T, TC, t_end=0.4)
tb, yb, _ = m.step_response(B, T, TC, t_end=0.4)
a.plot(t0 * 1e3, y0, color=GRAY, lw=1.8, ls="-.", label="샘플링 없는 연속 시간")
a.plot(ta * 1e3, ya, color=BLUE, lw=2.6, label="계산 직후 출력 (tc = 4 ms)")
a.plot(tb * 1e3, yb, color=ORANGE, lw=2.6, ls="--", label="다음 주기 시작에 출력")
a.axhline(1.0, color=GRAY, lw=1.0)
a.set_xlim(0, 400); a.set_ylim(0.0, 1.22)
oa, ob = 100 * (ya.max() - 1), 100 * (yb.max() - 1)
a.annotate(f"오버슈트 {oa:.1f} %", xy=(ta[np.argmax(ya)] * 1e3, ya.max()), xytext=(190, 1.115), fontsize=15, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
a.annotate(f"오버슈트 {ob:.1f} %", xy=(tb[np.argmax(yb)] * 1e3, yb.max()), xytext=(190, 1.18), fontsize=15, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
a.set_title("스텝 응답, T = 10 ms", fontsize=20)
a.set_xlabel("시간 [ms]", fontsize=16); a.set_ylabel("출력 y(t)", fontsize=16)
a.legend(fontsize=14, loc="lower right"); a.grid(alpha=.3)
a.tick_params(labelsize=14)

# 오른쪽: 오버슈트 대 tc
a = ax[1]
tcs = np.linspace(0, T, 51)
pa = [m.metrics(*m.step_response(A, T, tc)[:2])[0] for tc in tcs]
pb = m.metrics(*m.step_response(B, T, 0.0)[:2])[0]
a.plot(tcs * 1e3, pa, color=BLUE, lw=2.6, label="계산 직후 출력")
a.axhline(pb, color=ORANGE, lw=2.6, ls="--", label="다음 주기 시작에 출력")
a.axhline(GOAL, color=GRAY, lw=1.5, ls=":", label=f"{GOAL:.0f} %")
x12 = max(tc for tc, p in zip(np.arange(0, 10.0001, 0.1), [m.metrics(*m.step_response(A, T, tc / 1000)[:2])[0] for tc in np.arange(0, 10.0001, 0.1)]) if p <= GOAL)
a.annotate(f"tc {x12:.1f} ms 부근에서 {GOAL:.0f} %", xy=(x12, GOAL), xytext=(4.6, 9.4), fontsize=14, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
a.set_title("샘플링 주기 T = 10 ms", fontsize=20)
a.set_xlabel("계산 시간 tc [ms]", fontsize=16); a.set_ylabel("오버슈트 [%]", fontsize=16)
a.set_ylim(8, 18)
a.legend(fontsize=14, loc="lower right"); a.grid(alpha=.3); a.tick_params(labelsize=14)

fig.tight_layout()
fig.savefig("fig_tmp.png", facecolor=BG)
im = Image.open("fig_tmp.png").convert("RGB").resize((1600, 900), Image.LANCZOS)
im.save("fig_f4b5_2_update_order_overshoot.webp", lossless=True)
chk = Image.open("fig_f4b5_2_update_order_overshoot.webp")
print(chk.size, chk.mode, chk.getpixel((2, 2)))
