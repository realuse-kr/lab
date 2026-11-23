"""F-5 그림: 왼쪽은 세 입력의 스텝 응답, 오른쪽은 Kc 2에서 Ti에 따른 Ki 곡선."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pid_form_mismatch as m

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
BG = "#fcfcfb"
BLUE, ORANGE, GRAY = "#1f4e79", "#b5541c", "#444444"

fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=100, facecolor=BG, gridspec_kw={"width_ratios": [1.1, 1]})
for a in ax:
    a.set_facecolor(BG)

# 왼쪽: 스텝 응답 0~0.6 s
a = ax[0]
t = np.linspace(0, 0.6, 6001)
right = m.ideal_to_parallel(2.0, 0.1, 0.02)
wrong_b = (2.0, 0.1, 0.02)
wrong_c = m.ideal_to_parallel(2.0, 20.0, 0.04)
a.plot(t * 1e3, m.step_response(*right, t), color=BLUE, lw=2.8, label="이상형 숫자를 이상형으로 옮김 (Ki 20)")
a.plot(t * 1e3, m.step_response(*wrong_b, t), color=ORANGE, lw=2.8, label="이상형 숫자를 병렬형에 그대로 (Ki 0.1)")
a.plot(t * 1e3, m.step_response(*wrong_c, t), color=GRAY, lw=2.2, ls="--", label="병렬형 숫자를 이상형에 그대로 (Ki 0.1)")
a.axhline(1.0, color=GRAY, lw=1.0)
a.set_xlim(0, 600); a.set_ylim(0, 1.1)
a.set_title("스텝 응답", fontsize=20)
a.set_xlabel("시간 [ms]", fontsize=16); a.set_ylabel("출력 y(t)", fontsize=16)
a.legend(fontsize=13, loc="lower right"); a.grid(alpha=.3); a.tick_params(labelsize=14)

# 오른쪽: Kc = 2일 때 Ki = Kc / Ti
a = ax[1]
ti = np.logspace(-2, 2, 400)
a.loglog(ti, 2.0 / ti, color=GRAY, lw=2.2, label="Ki = Kc / Ti (Kc = 2)")
a.loglog([0.1], [20.0], "o", color=BLUE, ms=11, label="Ti 0.1 s → Ki 20")
a.loglog([20.0], [0.1], "s", color=GRAY, ms=11, label="Ti 20 s → Ki 0.1")
a.axhline(0.1, color=ORANGE, lw=2.0, ls=":", label="병렬형에 Ki = 0.1을 넣음")
a.set_xlim(1e-2, 1e2); a.set_ylim(3e-2, 1e3)
a.set_title("Ti가 커질수록 Ki는 작아진다", fontsize=20)
a.set_xlabel("적분 시간 Ti [s]", fontsize=16); a.set_ylabel("적분 게인 Ki", fontsize=16)
a.legend(fontsize=13, loc="upper right"); a.grid(alpha=.3, which="both"); a.tick_params(labelsize=14)

fig.tight_layout()
fig.savefig("fig_tmp.png", facecolor=BG)
im = Image.open("fig_tmp.png").convert("RGB").resize((1600, 900), Image.LANCZOS)
im.save("fig_f5_pid_form_gain_mismatch.webp", lossless=True)
chk = Image.open("fig_f5_pid_form_gain_mismatch.webp")
print(chk.size, chk.mode, chk.getpixel((2, 2)))
