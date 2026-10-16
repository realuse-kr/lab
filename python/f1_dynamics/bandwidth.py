"""F-1 대역폭: 세 정의(-3 dB, 상승시간 10~90%, 정정시간 2%)를 한 코드로 계산한다.

가상 파라미터 시뮬레이션이다. 단위: 시간 s, 각주파수 rad/s.
정상값은 DC 이득으로 잡는다(마지막 샘플 아님).
"""
import json
import pathlib
import numpy as np
from scipy import signal
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[2]


def second_order(zeta, wn):
    return signal.TransferFunction([wn**2], [1.0, 2 * zeta * wn, wn**2])


def first_order(tau):
    return signal.TransferFunction([1.0], [tau, 1.0])


def mag_db(sys, w):
    _, h = signal.freqresp(sys, [w])
    return 20 * np.log10(abs(h[0]))


HALF_POWER_DB = 10 * np.log10(2.0)  # 3.0103 dB, 이득 1/sqrt(2)


def bandwidth_rad_s(sys, w_hi, drop_db=3.0):
    """DC 이득 대비 이득이 drop_db만큼 내려가는 각주파수(단조 감소하는 계 가정).

    drop_db=3.0은 '-3 dB' 문자 그대로(이득 0.7079), HALF_POWER_DB는 반전력점(이득 0.7071)이다.
    """
    dc = 20 * np.log10(abs(sys.num[-1] / sys.den[-1]))
    f = lambda w: mag_db(sys, w) - (dc - drop_db)
    return brentq(f, 1e-6, w_hi, xtol=1e-12)


def step_metrics(sys, t_end, dt):
    t = np.arange(0.0, t_end, dt)
    _, y = signal.step(sys, T=t)
    yss = float(sys.num[-1] / sys.den[-1])  # DC 이득
    yn = y / yss
    def cross(level):
        i = int(np.argmax(yn >= level))
        # 선형 보간
        t0, t1, y0, y1 = t[i - 1], t[i], yn[i - 1], yn[i]
        return t0 + (level - y0) * (t1 - t0) / (y1 - y0)
    rise = cross(0.9) - cross(0.1)
    outside = np.where(np.abs(yn - 1.0) > 0.02)[0]
    settle = float(t[outside[-1] + 1]) if outside.size else 0.0
    overshoot = max(0.0, 100.0 * (yn.max() - 1.0))
    last_sample_yss = float(y[-1])
    return {"rise_s": rise, "settle_s": settle, "overshoot_pct": overshoot,
            "last_sample_over_dc": last_sample_yss / yss}


def closed_form_bw_second(zeta, wn):
    return wn * np.sqrt(1 - 2 * zeta**2 + np.sqrt(4 * zeta**4 - 4 * zeta**2 + 2))


def run():
    p = json.loads((ROOT / "params" / "f1_params.json").read_text(encoding="utf-8"))
    dt = p["time_grid_dt_s"]
    out = {"note": "가상 파라미터 시뮬레이션 결과", "first_order": {}, "second_order": {}, "sweep": []}

    tau = p["plant_first_order"]["tau_s"]
    s1 = first_order(tau)
    bw1 = bandwidth_rad_s(s1, 100.0)                      # -3 dB 문자 그대로
    bw1_hp = bandwidth_rad_s(s1, 100.0, HALF_POWER_DB)    # 반전력점
    m1 = step_metrics(s1, 12 * tau, dt)
    an1 = {"bw_minus3dB_rad_s": np.sqrt(10 ** 0.3 - 1) / tau, "bw_half_power_rad_s": 1.0 / tau,
           "rise_s": np.log(9) * tau, "settle_s": np.log(50) * tau}
    out["first_order"] = {"numeric": {"bw_minus3dB_rad_s": bw1, "bw_half_power_rad_s": bw1_hp, **m1},
                          "analytic": an1,
                          "rel_err": {"bw_minus3dB": abs(bw1 / an1["bw_minus3dB_rad_s"] - 1),
                                      "bw_half_power": abs(bw1_hp / an1["bw_half_power_rad_s"] - 1),
                                      "rise": abs(m1["rise_s"] / an1["rise_s"] - 1),
                                      "settle": abs(m1["settle_s"] / an1["settle_s"] - 1)}}

    z, wn = p["plant_second_order"]["zeta"], p["plant_second_order"]["wn_rad_s"]
    s2 = second_order(z, wn)
    bw2 = bandwidth_rad_s(s2, 10 * wn, HALF_POWER_DB)  # 닫힌식은 반전력점 기준
    bw2_3 = bandwidth_rad_s(s2, 10 * wn)               # -3 dB 문자 그대로
    m2 = step_metrics(s2, 10.0, dt)
    out["second_order"] = {
        "zeta": z, "wn_rad_s": wn,
        "numeric": {"bw_rad_s": bw2, "bw_hz": bw2 / (2 * np.pi), "bw_minus3dB_rad_s": bw2_3, **m2},
        "closed_form_bw_rad_s": closed_form_bw_second(z, wn),
        "overshoot_analytic_pct": 100 * np.exp(-np.pi * z / np.sqrt(1 - z**2)),
        "rise_x_bw_hz": m2["rise_s"] * bw2 / (2 * np.pi),
    }
    out["first_order"]["rise_x_bw_hz"] = m1["rise_s"] * bw1 / (2 * np.pi)  # -3 dB 기준
    out["second_order"]["rise_x_bw_hz"] = m2["rise_s"] * bw2_3 / (2 * np.pi)

    for zz in p["zeta_sweep"]:
        s = second_order(zz, wn)
        bw = bandwidth_rad_s(s, 10 * wn)  # -3 dB 문자 그대로(MathWorks·python-control 기본)
        bw_hp = bandwidth_rad_s(s, 10 * wn, HALF_POWER_DB)
        m = step_metrics(s, 40.0 / (zz * wn) if zz < 1 else 40.0 / wn, dt)
        out["sweep"].append({"zeta": zz, "bw_rad_s": bw, "bw_hz": bw / (2 * np.pi),
                             "bw_half_power_rad_s": bw_hp,
                             "bw_closed_form_half_power_rad_s": closed_form_bw_second(zz, wn),
                             "rise_s": m["rise_s"], "settle_s": m["settle_s"],
                             "overshoot_pct": m["overshoot_pct"],
                             "rise_x_bw_hz": m["rise_s"] * bw / (2 * np.pi)})
    res_dir = ROOT / "results" / "F-1"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["first_order"], indent=1, default=float))
    print(json.dumps(r["second_order"], indent=1, default=float))
    for row in r["sweep"]:
        print({k: round(float(v), 5) for k, v in row.items()})
