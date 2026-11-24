"""F-5B.3: PID 미분 항에 1차 필터를 달면 설정값 계단의 미분 킥이 어떻게 달라지는지 재 보기.

PID는 병렬형이다: u = Kp*e + Ki*(e의 적분) + d. e = r - y (r 설정값, y 측정값).
미분 필터는 연속 시간식 Tf*(d의 변화 속도) + d = Kd*(e의 변화 속도)이고,
변화 속도를 backward difference로 바꾸면 다음 점화식이 된다.
  d[k] = Tf/(Tf+Ts) * d[k-1] + Kd/(Tf+Ts) * (e[k] - e[k-1])
Tf = 0이면 필터가 없는 backward difference Kd*(e[k]-e[k-1])/Ts와 같다.
제어 대상은 1차 지연 플랜트 tau*y' = -y + K*u이고, 제어 출력은 한 샘플 동안 일정하게 유지한다.
"""
import numpy as np

KP = 2.0
KI = 2.0
KD = 0.04
TAU = 0.5
K_PLANT = 1.0
TS = 1e-3
DR = 1.0
T_STEP = 0.1
T_END = 1.0
U_MAX = 10.0

def simulate(tf, on_measurement=False, ts=TS, kp=KP, ki=KI, kd=KD, dr=DR):
    n = int(round(T_END / ts)); k_step = int(round(T_STEP / ts)); a = np.exp(-ts / TAU)
    alpha = tf / (tf + ts); beta = kd / (tf + ts)
    r = np.where(np.arange(n) >= k_step, dr, 0.0); y=np.zeros(n); u=np.zeros(n); d=np.zeros(n)
    integ=x_prev=d_prev=0.0
    for k in range(n):
        e=r[k]-y[k]; integ += ki*ts*e; x=-y[k] if on_measurement else e
        d[k]=alpha*d_prev+beta*(x-x_prev); u[k]=kp*e+integ+d[k]
        if k+1<n: y[k+1]=a*y[k]+(1.0-a)*K_PLANT*u[k]
        x_prev,d_prev=x,d[k]
    return r,y,u,d,k_step

def kick_theory(kd,dr,ts,tf): return kd*dr/(tf+ts)
def first_sample_output(kp,ki,kd,dr,ts,tf): return kp*dr+ki*ts*dr+kick_theory(kd,dr,ts,tf)
def tf_min_for_step(u_max,kp,ki,kd,ts,dr): return kd*dr/(u_max-kp*dr-ki*ts*dr)-ts
def pulse_area(tf,kd=KD,dr=DR,ts=TS,m=5000):
    alpha,beta=tf/(tf+ts),kd/(tf+ts); d=x_prev=total=0.0
    for k in range(m):
        x=dr if k>=100 else 0.0; d=alpha*d+beta*(x-x_prev); x_prev=x; total+=d*ts
    return total
