"""F-4B5.2: time_response_update_order.py의 결과를 results_f4b5_2.json으로 저장하고 계산이 맞는지 검사한다."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import dlti, dstep
import time_response_update_order as m

A, B = m.ORDERS
OUT = "results_f4b5_2.json"


def overshoot(order, T, tc, **kw):
    t, y, _ = m.step_response(order, T, tc, **kw)
    return m.metrics(t, y)[0]


def z_domain_samples(order, T, tc, n):
    """이산 루프에서 구한 샘플 시각의 스텝 응답. 시간 응답 계산과 독립이다."""
    d = tc if order == A else T
    c0, c1 = m.KP + m.KI * T / 2, -m.KP + m.KI * T / 2
    al = np.exp(-T / m.TAU)
    b0, b1 = 1 - np.exp(-(T - d) / m.TAU), np.exp(-(T - d) / m.TAU) * (1 - np.exp(-d / m.TAU))
    num = np.polymul([c0, c1], [b0, b1])                      # L_d = (c0 z + c1)(b0 z + b1)/((z - 1) z (z - al))
    den = np.polymul(np.polymul([1, -1], [1, 0]), [1, -al])
    cl = np.polyadd(den, num)
    stable = bool(np.max(np.abs(np.roots(cl))) < 1)
    _, yz = dstep(dlti(num, cl, dt=T), n=n)
    return np.squeeze(yz), stable


def main():
    res = {"params": {"KP": m.KP, "KI": m.KI, "TAU": m.TAU, "band": m.BAND}}
    t0, y0 = m.continuous_reference()
    o0, _, s0 = m.metrics(t0, y0)
    res["continuous"] = {"overshoot_pct": o0, "settling_ms": s0 * 1e3}

    # 표 1: T = 10 ms에서 tc를 바꾼 결과
    T = 0.010
    rows = []
    for tc_ms in (0, 2, 4, 6, 8, 10):
        row = {"tc_ms": tc_ms}
        for key, order in (("A", A), ("B", B)):
            t, y, _ = m.step_response(order, T, tc_ms / 1000)
            o, _, s = m.metrics(t, y)
            row[key] = {"overshoot_pct": o, "settling_ms": s * 1e3}
        rows.append(row)
    res["T10"] = rows

    # 표 2: 샘플링 주기 5, 10, 20 ms
    byT = {}
    for Tm in (5, 10, 20):
        Tt = Tm / 1000
        byT[str(Tm)] = {"A_tc0": overshoot(A, Tt, 0.0), "A_tcHalfT": overshoot(A, Tt, Tt / 2), "A_tcT": overshoot(A, Tt, Tt), "B": overshoot(B, Tt, 0.0)}
    res["by_T"] = byT

    # 목표 오버슈트 12 %: 0.1 ms 단위로 찾은 한도와 단조성
    goal = 12.0
    tcs = np.round(np.arange(0.0, 10.0001, 0.1), 1)
    ovA = np.array([overshoot(A, 0.010, tc / 1000) for tc in tcs])
    ok = tcs[ovA <= goal]
    Ts = np.round(np.arange(1.0, 20.0001, 0.1), 1)
    ovB = np.array([overshoot(B, T_ / 1000, 0.0) for T_ in Ts])
    okB = Ts[ovB <= goal]
    res["criterion_12"] = {
        "goal_pct": goal,
        "T10_A_max_tc_ms": float(ok.max()),
        "A_overshoot_at_max_tc": float(ovA[tcs == ok.max()][0]),
        "A_overshoot_at_next_tc": float(ovA[tcs == round(ok.max() + 0.1, 1)][0]),
        "B_max_T_ms": float(okB.max()),
        "B_overshoot_at_max_T": float(ovB[Ts == okB.max()][0]),
        "B_overshoot_at_next_T": float(ovB[Ts == round(okB.max() + 0.1, 1)][0]),
        "A_monotone_in_tc": bool(np.all(np.diff(ovA) > 0)),
        "B_monotone_in_T": bool(np.all(np.diff(ovB) > 0)),
    }

    # 범위: T 1~20 ms(1 ms 단위), tc 0, T/4, T/2, 3T/4, T
    rng = {"stable": True, "A_monotone_in_tc": True, "B_equals_A_at_tcT": 0.0, "max_sample_vs_continuous_pp": 0.0, "max_err_vs_dstep": 0.0}
    for Tm in range(1, 21):
        Tt = Tm / 1000
        prev = -1.0
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            tc = frac * Tt
            for order in (A, B):
                t, y, _ = m.step_response(order, Tt, tc)
                yz, stable = z_domain_samples(order, Tt, tc, len(y[::100]))
                rng["stable"] &= stable
                rng["max_err_vs_dstep"] = max(rng["max_err_vs_dstep"], float(np.max(np.abs(y[::100] - yz))))
                rng["max_sample_vs_continuous_pp"] = max(rng["max_sample_vs_continuous_pp"], 100 * (y.max() - y[::100].max()))
            oa = overshoot(A, Tt, tc)
            rng["A_monotone_in_tc"] &= oa > prev
            prev = oa
        rng["B_equals_A_at_tcT"] = max(rng["B_equals_A_at_tcT"], abs(overshoot(B, Tt, 0.0) - overshoot(A, Tt, Tt)))
    res["range"] = rng

    # 검사
    chk = {}
    # (a) 다음 주기 시작에 출력은 tc와 관계없다
    ref = m.step_response(B, 0.010, 0.0)[1]
    chk["check_B_independent_of_tc"] = bool(max(np.max(np.abs(m.step_response(B, 0.010, tc)[1] - ref)) for tc in (0.002, 0.005, 0.01)) < 1e-12)
    # (b) tc = T이면 두 순서가 같다
    chk["check_tcT_equal"] = bool(np.max(np.abs(m.step_response(A, 0.010, 0.010)[1] - ref)) < 1e-12)
    # (c) 샘플 시각의 출력이 이산 루프에서 구한 값과 같다(위 범위 검사의 최댓값)
    chk["check_dstep"] = bool(rng["max_err_vs_dstep"] < 1e-9)
    # (d) 샘플 사이의 출력이 solve_ivp로 구한 값과 같다(무작위 20회)
    rs = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        Tt = float(rs.uniform(0.002, 0.020))
        tc = float(rs.uniform(0.0, Tt))
        order = A if rs.random() < 0.5 else B
        d = tc if order == A else Tt
        t, y, us = m.step_response(order, Tt, tc, t_end=Tt * 20)
        up = np.concatenate([[0.0], us])

        def u_of(tt):
            k = min(int(tt // Tt), len(us) - 1)
            return up[k + 1] if (tt - k * Tt) >= d else up[k]
        sol = solve_ivp(lambda tt, x: (u_of(tt) - x) / m.TAU, (0, t[-1]), [0.0], t_eval=t, rtol=1e-11, atol=1e-13, max_step=Tt / 50)
        worst = max(worst, float(np.max(np.abs(sol.y[0] - y))))
    chk["max_err_solve_ivp"] = worst
    chk["check_solve_ivp"] = bool(worst < 1e-7)
    # (e) 격자를 10배 촘촘히 하면 오버슈트가 얼마나 바뀌나
    chk["max_grid_change_pp"] = max(abs(overshoot(o, 0.010, tc, sub=1000) - overshoot(o, 0.010, tc)) for o in (A, B) for tc in (0.0, 0.004, 0.008, 0.010))
    res["checks"] = chk
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print(f"연속 시간 폐루프: 오버슈트 {o0:.1f} %, 정정 시간 {s0 * 1e3:.0f} ms")
    c = res["criterion_12"]
    print(f"오버슈트 {goal:.0f} % 이하: T = 10 ms에서 계산 직후 출력은 tc {c['T10_A_max_tc_ms']:.1f} ms 이하(그때 {c['A_overshoot_at_max_tc']:.2f} %, 다음 0.1 ms에서 {c['A_overshoot_at_next_tc']:.2f} %)")
    print(f"오버슈트 {goal:.0f} % 이하: 다음 주기 시작에 출력은 T {c['B_max_T_ms']:.1f} ms 이하(그때 {c['B_overshoot_at_max_T']:.2f} %, 다음 0.1 ms에서 {c['B_overshoot_at_next_T']:.2f} %)")
    print(f"tc에 따라 단조 증가 {c['A_monotone_in_tc']}, T에 따라 단조 증가 {c['B_monotone_in_T']}")
    print(f"범위 T 1~20 ms: 안정 {rng['stable']}, 두 순서 tc = T 차이 {rng['B_equals_A_at_tcT']:.1e}, 이산 루프 응답과 최대 차이 {rng['max_err_vs_dstep']:.1e}, 샘플 값만 볼 때 오버슈트가 작은 최대 {rng['max_sample_vs_continuous_pp']:.2f} %p")
    print(f"검사: {{ {', '.join(f'{k}: {v}' for k, v in chk.items())} }}")


if __name__ == "__main__":
    main()
