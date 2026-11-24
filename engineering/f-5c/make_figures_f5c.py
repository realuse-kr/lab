"""F-5C 그림: 한도에 걸린 동안 쌓이는 적분 항과 그 때문에 커지는 오버슈트."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pid_windup import DR, UMAX, T_STEP, simulate, windup_formula

BG = "#fcfcfb"
plt.rcParams.update({
    "font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 15,
})
RED, BLUE, GRAY = "#c0392b", "#1f5fa8", "#6b6b68"
cases = [("ideal", GRAY, "-.", "한도 없음", 2.2), ("plain", RED, "-", "한도, 적분 그대로", 2.8),
         ("stop", BLUE, "-", "한도, 적분 멈춤", 2.4)]
data = {m: simulate(m) for m, *_ in cases}
k0 = data["plain"][4]
sel = slice(k0 - 10, k0 + 2501)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 9), dpi=100)
for m, col, ls, name, lw in cases:
    t, y, u, integ, _ = data[m]
    tt = t[sel] - T_STEP
    a1.plot(tt, y[sel], color=col, ls=ls, lw=lw, label=name)
    a2.plot(tt, integ[sel], color=col, ls=ls, lw=lw, label=name)
a1.axhline(DR, color="#999999", lw=1.2, ls="--")
a1.text(2.45, DR - 0.35, "설정값 8", ha="right", va="top", color="#444444")
tp, yp = data["plain"][0], data["plain"][1]
ip = int(np.argmax(yp))
a1.annotate(f"오버슈트 {yp[ip] - DR:.3f}", xy=(tp[ip] - T_STEP, yp[ip]), xytext=(1.25, 9.95),
            arrowprops=dict(arrowstyle="-", color=RED), color=RED)
a1.set_ylim(-0.3, 10.8)
a1.set_xlabel("설정값이 바뀐 뒤 시간 [s]")
a1.set_ylabel("측정값")
a1.set_title("적분을 그대로 두면 오버슈트가 커진다", fontsize=17)
a1.legend(loc="lower right", frameon=False)
a1.grid(True, color="#e3e3e0", lw=0.7)

a2.axhline(DR, color="#999999", lw=1.2, ls="--")
a2.text(2.45, DR - 0.45, "설정값을 유지하는 데 필요한 값 8", ha="right", va="top", color="#444444")
ti, ii = data["plain"][0], data["plain"][3]
im = int(np.argmax(ii))
a2.annotate(f"{ii[im]:.2f} (식 {windup_formula():.2f})", xy=(ti[im] - T_STEP, ii[im]),
            xytext=(1.0, 20.5), arrowprops=dict(arrowstyle="-", color=RED), color=RED)
a2.set_ylim(-0.5, 22.5)
a2.set_xlabel("설정값이 바뀐 뒤 시간 [s]")
a2.set_ylabel("적분 항")
a2.set_title("한도에 걸린 동안 적분 항이 필요한 값보다 쌓인다", fontsize=17)
a2.legend(loc="center right", frameon=False)
a2.grid(True, color="#e3e3e0", lw=0.7)
fig.tight_layout()
fig.savefig("fig_f5c_pid_windup.png")
img = Image.open("fig_f5c_pid_windup.png").convert("RGB").resize((1600, 900))
img.save("fig_f5c_pid_windup.webp", quality=88)
print(img.mode, img.size, img.getpixel((0, 0)))
