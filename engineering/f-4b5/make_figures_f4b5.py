"""F-4B5 그림: 왼쪽은 T = 10 ms, tc = 4 ms에서 두 출력 순서의 샘플 시각과 출력 갱신 시각, 오른쪽은 계산 시간 tc에 따른 두 순서의 위상 여유."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import phase_margin_update_order as m

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
BG = "#fcfcfb"
BLUE, ORANGE, GRAY = "#1f4e79", "#b5541c", "#444444"
T, TC = 0.010, 0.004

fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=100, facecolor=BG, gridspec_kw={"width_ratios": [1.05, 1]})
for a in ax:
    a.set_facecolor(BG)

# 왼쪽: 타임라인
a = ax[0]
rows = {m.ORDERS[0]: (1.3, BLUE, "계산 직후 출력"), m.ORDERS[1]: (0.0, ORANGE, "다음 주기 시작에 출력")}
for order, (y, c, name) in rows.items():
    recs = m.schedule(order, T, TC, periods=4)
    for k in range(3):  # 주기 세 개의 계산 구간
        a.barh(y, TC * 1e3, left=k * T * 1e3, height=0.16, color=c, alpha=0.35, edgecolor=c)
        a.plot(k * T * 1e3, y + 0.22, marker="v", color=GRAY, ms=11)
    for smp, out in recs[:3]:
        a.plot(out * 1e3, y - 0.22, marker="^", color=c, ms=11)
        a.annotate("", xy=(out * 1e3, y - 0.2), xytext=(smp * 1e3, y + 0.2), arrowprops=dict(arrowstyle="->", color=c, lw=1.6, alpha=0.8))
    a.text(-1.0, y + 0.42, name, fontsize=17, color=c, ha="left", fontweight="bold")
a.text(10.0, 1.3 - 0.52, "샘플에서 출력까지 tc = 4 ms", fontsize=15, color=BLUE, ha="center")
a.text(15.0, 0.0 - 0.5, "샘플에서 출력까지 T = 10 ms", fontsize=15, color=ORANGE, ha="center")
a.plot([], [], marker="v", color=GRAY, ls="", ms=11, label="샘플 읽기")
a.plot([], [], marker="^", color=GRAY, ls="", ms=11, label="출력 갱신")
from matplotlib.patches import Patch
h, l = a.get_legend_handles_labels()
h.append(Patch(facecolor=GRAY, alpha=0.35, edgecolor=GRAY)); l.append("계산 구간")
a.set_xlim(-1.5, 31); a.set_ylim(-0.8, 2.0)
a.set_yticks([]); a.set_xlabel("시간 [ms]", fontsize=16); a.tick_params(labelsize=14)
a.set_title("T = 10 ms, tc = 4 ms", fontsize=20)
a.legend(h, l, fontsize=13, loc="lower right", ncol=1)
a.grid(alpha=.3, axis="x")

# 오른쪽: 위상 여유
a = ax[1]
wc0, pm0 = m.crossover_and_margin(m.loop_base, m.W_HI)
tcs = np.linspace(0, T, 51)
pa = [m.margin_disc(T, m.ORDERS[0], t)[1] for t in tcs]
pb = [m.margin_disc(T, m.ORDERS[1], t)[1] for t in tcs]
a.plot(tcs * 1e3, pa, color=BLUE, lw=2.5, label="계산 직후 출력")
a.plot(tcs * 1e3, pb, color=ORANGE, lw=2.5, ls="--", label="다음 주기 시작에 출력")
a.axhline(60.0, color=GRAY, lw=1.5, ls=":", label="60도")
x60 = max(t for t, p in zip(tcs * 1e3, pa) if p >= 60.0)
a.annotate(f"tc {x60:.1f} ms 부근에서 60도", xy=(x60, 60.0), xytext=(5.6, 63.5), fontsize=14, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
a.set_title("샘플링 주기 T = 10 ms", fontsize=20)
a.set_xlabel("계산 시간 tc [ms]", fontsize=16); a.set_ylabel("위상 여유 [도]", fontsize=16)
a.legend(fontsize=14); a.grid(alpha=.3); a.tick_params(labelsize=14)

fig.tight_layout()
fig.savefig("fig_tmp.png", facecolor=BG)
im = Image.open("fig_tmp.png").convert("RGB").resize((1600, 900), Image.LANCZOS)
im.save("fig_f4b5_update_order_phase_margin.webp", lossless=True)
chk = Image.open("fig_f4b5_update_order_phase_margin.webp")
print(chk.size, chk.mode, chk.getpixel((2, 2)))
