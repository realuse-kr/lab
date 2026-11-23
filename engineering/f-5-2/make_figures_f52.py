"""F-5.2 그림: 왼쪽은 변환한 두 표기와 숫자를 그대로 넣은 응답, 오른쪽은 네 비교의 최대 차이(로그 눈금)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pid_form_equivalence as m

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
BG = "#fcfcfb"
BLUE, ORANGE, GRAY = "#1f4e79", "#b5541c", "#444444"

kc, ti, td = 2.0, 0.1, 0.02
kp, ki, kd = m.ideal_to_parallel(kc, ti, td)

fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=100, facecolor=BG, gridspec_kw={"width_ratios": [1, 1.15]})
for a in ax:
    a.set_facecolor(BG)

# 왼쪽: 스텝 응답 0~0.4 s
a = ax[0]
t = np.linspace(0, 0.4, 4001)
y_par, _ = m.step_tf_parallel(kp, ki, kd, t)
y_ide, _ = m.step_tf_ideal(kc, ti, td, t)
y_raw, _ = m.step_tf_parallel(kc, ti, td, t)
a.plot(t * 1e3, y_par, color=BLUE, lw=4.0, label="병렬형 Kp 2, Ki 20, Kd 0.04")
a.plot(t * 1e3, y_ide, color=ORANGE, lw=2.2, ls="--", label="이상형 Kc 2, Ti 0.1, Td 0.02")
a.plot(t * 1e3, y_raw, color=GRAY, lw=2.4, label="이상형 숫자를 병렬형에 그대로")
a.axhline(1.0, color=GRAY, lw=1.0)
a.set_xlim(0, 400); a.set_ylim(0, 1.15)
a.set_title("스텝 응답", fontsize=20)
a.set_xlabel("시간 [ms]", fontsize=16); a.set_ylabel("출력 y(t)", fontsize=16)
a.legend(fontsize=13, loc="lower right"); a.grid(alpha=.3); a.tick_params(labelsize=14)

# 오른쪽: 0~1 s 응답의 최대 차이
tt = np.linspace(0, 1.0, 100001)
yp, _ = m.step_tf_parallel(kp, ki, kd, tt)
yi, _ = m.step_tf_ideal(kc, ti, td, tt)
yp_ode = m.step_ode_parallel(kp, ki, kd, tt)
yi_ode = m.step_ode_ideal(kc, ti, td, tt)
yr, _ = m.step_tf_parallel(kc, ti, td, tt)
rows = [
    ("변환한 병렬형 대 이상형\n전달함수 경로", np.abs(yp - yi).max(), BLUE),
    ("변환한 병렬형 대 이상형\n시간 영역 경로", np.abs(yp_ode - yi_ode).max(), BLUE),
    ("같은 병렬형\n두 경로의 차이", np.abs(yp - yp_ode).max(), GRAY),
    ("이상형 숫자를 그대로 넣은\n병렬형 대 변환한 병렬형", np.abs(yr - yp).max(), ORANGE),
]
a = ax[1]
FLOOR = 1e-17
ys = np.arange(len(rows))[::-1]
for y0, (name, v, c) in zip(ys, rows):
    a.barh(y0, v - FLOOR, left=FLOOR, color=c, height=0.55)
    a.text(v * 2.5, y0, f"{v:.3f}" if v > 0.01 else f"{v:.1e}", va="center", fontsize=15)
a.set_xscale("log"); a.set_xlim(FLOOR, 30)
a.set_yticks(ys); a.set_yticklabels([r[0] for r in rows], fontsize=14)
a.set_title("0~1 s 응답의 최대 차이", fontsize=20)
a.set_xlabel("최대 차이 (log 눈금)", fontsize=16)
a.grid(alpha=.3, which="major", axis="x"); a.tick_params(axis="x", labelsize=14)

fig.tight_layout()
fig.savefig("fig_tmp.png", facecolor=BG)
im = Image.open("fig_tmp.png").convert("RGB").resize((1600, 900), Image.LANCZOS)
im.save("fig_f52_pid_form_equivalence.webp", lossless=True)
chk = Image.open("fig_f52_pid_form_equivalence.webp")
print(chk.size, chk.mode, chk.getpixel((2, 2)))