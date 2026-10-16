"""F-1 그림 2장(CODE_RENDER). 가상 파라미터 시뮬레이션.
fig1: 응답+주파수 응답 복합(대표·썸네일·본문 겸용), fig2: ζ 스윕 지표.
이미지 안 문구는 명사형 라벨만 쓴다(문장·경어체 없음).
"""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

from bandwidth import ROOT, second_order, first_order, bandwidth_rad_s, HALF_POWER_DB

SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
OUT = ROOT / "figures" / "F-1"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "regular", "axes.titlelocation": "left",
})


def footer(fig):
    # 하단 비정보 영역의 작고 옅은 워터마크
    fig.text(0.985, 0.022, "REALUSE", ha="right", va="bottom", fontsize=11, color=MUTED, alpha=0.55)


def to_webp(png):
    im = Image.open(png).convert("RGB")
    assert im.size == (1600, 900), im.size
    w = png.with_suffix(".webp")
    im.save(w, "WEBP", quality=90, method=6)
    return w


def cross(t, yn, level):
    i = int(np.argmax(yn >= level))
    return t[i - 1] + (level - yn[i - 1]) * (t[i] - t[i - 1]) / (yn[i] - yn[i - 1])


def fig1(p):
    z, wn = p["plant_second_order"]["zeta"], p["plant_second_order"]["wn_rad_s"]
    s = second_order(z, wn)
    t = np.linspace(0, 1.6, 16001)
    _, y = signal.step(s, T=t)
    t10, t90 = cross(t, y, 0.1), cross(t, y, 0.9)
    out = np.where(np.abs(y - 1) > 0.02)[0]
    ts = t[out[-1] + 1]
    ip = int(np.argmax(y))
    bw = bandwidth_rad_s(s, 10 * wn)  # -3 dB 문자 그대로
    w = np.logspace(0, np.log10(200), 800)
    _, h = signal.freqresp(s, w)
    mag = 20 * np.log10(np.abs(h))

    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"wspace": 0.22})
    fig.subplots_adjust(left=0.065, right=0.975, top=0.88, bottom=0.2)
    a.axhspan(0.98, 1.02, color=GRID, alpha=0.9, lw=0)
    a.axhline(1, color=BASE, lw=1)
    a.plot(t, y, color=BLUE, lw=2.4)
    a.plot([t10, t90], [0.1, 0.9], "o", color=INK, ms=9, mfc=SURF, mew=2)
    a.plot([ts], [y[out[-1] + 1]], "s", color=INK, ms=9, mfc=SURF, mew=2)
    a.plot([t[ip]], [y[ip]], "^", color=INK, ms=10, mfc=SURF, mew=2)
    a.annotate(f"상승 {t90 - t10:.3f} s\n(10%~90%)", (t90, 0.9), (0.30, 0.46), color=INK2, fontsize=14,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    a.annotate(f"정정 {ts:.3f} s\n(±2%)", (ts, 1.02), (1.05, 1.34), color=INK2, fontsize=14,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    a.annotate(f"오버슈트 {100 * (y[ip] - 1):.1f}%", (t[ip], y[ip]), (0.28, 1.5), color=INK2, fontsize=14,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    a.set_title("(a) 스텝 응답")
    a.set_xlabel("시간 t [s]")
    a.set_ylabel("정규화 출력 y/DC 이득")
    a.set_xlim(0, 1.6)
    a.set_ylim(-0.02, 1.62)

    b.semilogx(w, mag, color=BLUE, lw=2.4)
    b.axhline(-3, color=INK2, lw=1.2, ls=(0, (5, 4)))
    b.plot([bw], [-3], "o", color=INK, ms=9, mfc=SURF, mew=2)
    b.axvline(bw, color=MUTED, lw=1, ls=":")
    b.annotate(f"대역폭 {bw:.2f} rad/s\n= {bw / (2 * np.pi):.2f} Hz", (bw, -3), (2.1, -24), color=INK2, fontsize=14,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    b.text(1.15, -2.2, "-3 dB", color=INK2, fontsize=13, va="bottom")
    b.set_title("(b) 주파수 응답 크기")
    b.set_xlabel("각주파수 ω [rad/s]")
    b.set_ylabel("이득 [dB, DC 기준]")
    b.set_xlim(1, 200)
    b.set_ylim(-48, 12)
    fig.text(0.065, 0.065, f"가상 파라미터 시뮬레이션: ζ = {z}, ωn = {wn:g} rad/s", color=INK2, fontsize=14, va="bottom")
    footer(fig)
    png = OUT / "f1_fig1_response_bode.png"
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return to_webp(png)


def fig2(res):
    sw = res["sweep"]
    zs = np.array([r["zeta"] for r in sw])
    rb = np.array([r["rise_x_bw_hz"] for r in sw])
    sb = np.array([r["settle_s"] * r["bw_hz"] for r in sw])
    ref = res["first_order"]["rise_x_bw_hz"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"wspace": 0.2})
    fig.subplots_adjust(left=0.07, right=0.975, top=0.88, bottom=0.2)
    a.axhline(ref, color=INK2, lw=1.2, ls=(0, (5, 4)))
    a.plot(zs, rb, "-o", color=BLUE, lw=2.4, ms=9)
    a.text(0.1, ref + 0.004, f"1차계 {ref:.3f}", color=INK2, fontsize=13, va="bottom")
    a.set_title("(a) 상승시간 × 대역폭")
    a.set_xlabel("감쇠비 ζ")
    a.set_ylabel("상승시간 × 대역폭 [Hz·s]")
    a.set_ylim(0.24, 0.38)
    a.set_xlim(0, 2.1)
    k = int(np.argmin(abs(zs - 0.32)))
    a.annotate(f"ζ=0.32: {rb[k]:.3f}", (zs[k], rb[k]), (0.5, 0.265), color=INK2, fontsize=14,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    b.plot(zs, sb, "-s", color=ORANGE, lw=2.4, ms=9)
    b.set_title("(b) 정정시간 × 대역폭")
    b.set_xlabel("감쇠비 ζ")
    b.set_ylabel("정정시간(±2%) × 대역폭 [Hz·s]")
    b.set_xlim(0, 2.1)
    b.set_ylim(0, max(sb) * 1.15)
    kb = int(np.argmax(sb))
    b.annotate(f"최대 {sb[kb]:.1f} (ζ={zs[kb]:g})", (zs[kb], sb[kb]), (0.65, sb[kb] * 0.97), color=INK2, fontsize=14,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    fig.text(0.07, 0.065, "가상 파라미터 시뮬레이션: ωn = 11 rad/s 고정, 대역폭은 -3 dB 기준", color=INK2, fontsize=14, va="bottom")
    footer(fig)
    png = OUT / "f1_fig2_zeta_sweep.png"
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return to_webp(png)


if __name__ == "__main__":
    p = json.loads((ROOT / "params" / "f1_params.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "results" / "F-1" / "results.json").read_text(encoding="utf-8"))
    for f in (fig1(p), fig2(res)):
        print(f, f.stat().st_size, "bytes")
