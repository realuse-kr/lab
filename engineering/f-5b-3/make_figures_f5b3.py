"""F-5B.3 그림 생성기는 Drive 정본과 동일한 원본을 사용한다."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pid_derivative_filter import KP, KI, KD, TS, DR, U_MAX, simulate
BG="#fcfcfb"; plt.rcParams.update({"font.family":"Noto Sans CJK KR","axes.unicode_minus":False,"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"font.size":15})
RED,AMB,BLUE,GRAY="#c0392b","#b8860b","#1f5fa8","#6b6b68"
res=json.load(open("results_f5b3.json",encoding="utf-8")); tf_ok=res["at_tf_ok"]["tf_ms"]*1e-3
cases=[(0.0,RED,"필터 없음"),(2e-3,AMB,"Tf 2 ms"),(tf_ok,BLUE,"Tf 4.0013 ms")]
fig,(a1,a2)=plt.subplots(1,2,figsize=(16,9),dpi=100,gridspec_kw={"width_ratios":[1.1,1]})
for tf,col,name in cases:
 _,_,_,d,k0=simulate(tf); k=np.arange(k0-2,k0+21); a1.step(k-k0,d[k],where="mid",color=col,lw=2.4,label=name)
a1.axhline(0,color="#999999",lw=0.8); a1.set_xlabel("설정값이 바뀐 샘플부터 센 샘플 수 (샘플링 주기 1 ms)"); a1.set_ylabel("미분 항 출력"); a1.set_title("필터를 달면 미분 킥이 낮아지고 길어진다",fontsize=17); a1.legend(loc="upper right",frameon=False)
tt=np.linspace(0,16,300); a2.plot(tt,KD*DR/(tt*1e-3+TS),color=GRAY,lw=2.2,label="식 Kd·Δr÷(Tf+Ts)")
rows=res["rows"]; a2.plot([r["tf_ms"] for r in rows],[r["d_at_step"] for r in rows],"o",color=RED,ms=11,label="코드로 계산한 미분 항")
lim=U_MAX-KP*DR-KI*TS*DR; a2.axhline(lim,color=BLUE,lw=2,ls="--",label=f"한도 10 안에 드는 높이 {lim:.3f}"); a2.axvline(tf_ok*1e3,color=BLUE,lw=1.4,ls=":")
a2.annotate("Tf 4.0013 ms",xy=(tf_ok*1e3,lim),xytext=(6.0,22),color="#222222",arrowprops=dict(arrowstyle="-",color=BLUE))
a2.set_xlabel("미분 필터 시간상수 Tf [ms]"); a2.set_ylabel("설정값이 바뀐 샘플의 미분 항"); a2.set_title("Tf를 4.0013 ms 이상으로 두면 한도 안",fontsize=17); a2.set_ylim(0,44); a2.legend(loc="upper right",frameon=False); a2.grid(True,color="#e3e3e0",lw=0.7)
fig.tight_layout(); fig.savefig("fig_f5b3_pid_derivative_filter.png"); im=Image.open("fig_f5b3_pid_derivative_filter.png").convert("RGB").resize((1600,900)); im.save("fig_f5b3_pid_derivative_filter.webp",quality=88)
