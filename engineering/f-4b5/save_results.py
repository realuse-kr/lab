"""F-4B5: phase_margin_update_order.py의 계산을 표와 검사 값으로 저장한다. python3 save_results.py로 results_f4b5.json을 만든다."""
import json
import numpy as np
from scipy.integrate import solve_ivp
import phase_margin_update_order as m

A, B = "compute_then_write", "write_then_compute"
wc0, pm0 = m.crossover_and_margin(m.loop_base, m.W_HI)
out = {"wc0": wc0, "pm0": pm0, "table_T10": [], "criterion_60": {}}

# 1. T = 10 ms에서 계산 시간별 두 순서의 위상 여유 (본문 표)
T = 0.010
for tc_ms in (0, 2, 4, 6, 8, 10):
    tc = tc_ms / 1000
    pa, pb = m.margin_disc(T, A, tc)[1], m.margin_disc(T, B, tc)[1]
    out["table_T10"].append({"tc_ms": tc_ms, "delay_A_ms": m.output_delay(A, T, tc) * 1e3, "delay_B_ms": m.output_delay(B, T, tc) * 1e3,
                             "pm_A": pa, "pm_B": pb, "diff": pa - pb, "wc_T_minus_tc_deg": float(np.degrees(wc0 * (T - tc)))})

# 2. 범위 확인: T 1~20 ms, tc 0, T/4, T/2, 3T/4, T
max_diff_formula = 0.0  # |(pm_A - pm_B) - wc0*(T - tc)| 의 최댓값 [도]
max_disc_total = 0.0  # 이산 루프와 총 지연 루프의 위상 여유 차이의 최댓값 [도]
max_pm_formula_A = 0.0  # |pm_A - (pm0 - wc0*(T/2 + tc))| 최댓값 [도]
max_pm_formula_B = 0.0  # |pm_B - (pm0 - wc0*(T/2 + T))| 최댓값 [도]
for t_ms in range(1, 21):
    T = t_ms / 1000
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        tc = f * T
        pa, pb = m.margin_disc(T, A, tc)[1], m.margin_disc(T, B, tc)[1]
        ta, tb = m.margin_total(T, A, tc)[1], m.margin_total(T, B, tc)[1]
        max_diff_formula = max(max_diff_formula, abs((pa - pb) - np.degrees(wc0 * (T - tc))))
        max_disc_total = max(max_disc_total, abs(pa - ta), abs(pb - tb))
        max_pm_formula_A = max(max_pm_formula_A, abs(pa - (pm0 - np.degrees(wc0 * (T / 2 + tc)))))
        max_pm_formula_B = max(max_pm_formula_B, abs(pb - (pm0 - np.degrees(wc0 * (T / 2 + T)))))
out["range"] = {"T_ms": [1, 20], "tc_fractions": [0, 0.25, 0.5, 0.75, 1.0], "max_diff_formula_deg": max_diff_formula,
                "max_disc_total_deg": max_disc_total, "max_pm_formula_A_deg": max_pm_formula_A, "max_pm_formula_B_deg": max_pm_formula_B}

# 3. 60도 기준: T = 5, 10, 20 ms에서 다음 주기 시작에 출력하는 경우, T = 10 ms에서 계산 직후 출력이 허용하는 tc 최댓값
out["criterion_60"]["pm_B_by_T"] = {str(t): m.margin_disc(t / 1000, B, 0.0)[1] for t in (5, 10, 20)}
ok = [tc_ms for tc_ms in np.arange(0, 100.5, 1) if m.margin_disc(0.010, A, tc_ms / 10000)[1] >= 60.0]  # tc를 0.1 ms 단위로 (0~10 ms)
out["criterion_60"]["T10_A_max_tc_ms"] = max(ok) / 10
pm_tc_grid = [m.margin_disc(0.010, A, k / 10000)[1] for k in range(0, 101)]
out["criterion_60"]["T10_A_monotonic_decreasing"] = bool(all(x > y for x, y in zip(pm_tc_grid, pm_tc_grid[1:])))
Tgrid = [k / 10 for k in range(10, 101)]  # 1.0~10.0 ms를 0.1 ms 단위로
pmB = [m.margin_disc(t / 1000, B, 0.0)[1] for t in Tgrid]
out["criterion_60"]["B_max_T_ms"] = max(t for t, p in zip(Tgrid, pmB) if p >= 60.0)
out["criterion_60"]["B_monotonic_decreasing_in_T"] = bool(all(x > y for x, y in zip(pmB, pmB[1:])))

# 3-1. T = 5, 10, 20 ms에서 다음 주기 시작에 출력의 위상 여유와 계산 직후 출력이 60도 이상인 tc 범위(0.1 ms 단위)
by_T = {}
for t_ms in (5, 10, 20):
    Tt = t_ms / 1000
    tcs_ms = [k / 10 for k in range(0, int(t_ms * 10) + 1)]
    okA = [x for x in tcs_ms if m.margin_disc(Tt, A, x / 1000)[1] >= 60.0]
    by_T[str(t_ms)] = {"pm_B": m.margin_disc(Tt, B, 0.0)[1], "A_max_tc_ms": (max(okA) if okA else None), "A_all_tc": len(okA) == len(tcs_ms)}
out["criterion_60"]["by_T"] = by_T

# 3-2. 식으로 한도 어림: PM0 - wc0 * (T/2 + 지연) >= 60도인 총 허용 지연 [ms]와 순서별 한도
budget = float(np.radians(pm0 - 60.0) / wc0 * 1e3)  # T/2 + 지연이 넘지 않아야 하는 시간 [ms]
out["criterion_60"]["formula_budget_ms"] = budget
out["criterion_60"]["formula_T10_A_max_tc_ms"] = budget - 5.0
out["criterion_60"]["formula_B_max_T_ms"] = budget / 1.5

# 4. 검사
checks = {}
# 4-1. 순서별 지연: schedule이 만든 모든 쌍에서 계산 직후 출력은 tc, 다음 주기 시작에 출력은 T
checks["check_delay_schedule"] = all(
    abs(m.output_delay(A, 0.010, f * 0.010) - f * 0.010) < 1e-9 and abs(m.output_delay(B, 0.010, f * 0.010) - 0.010) < 1e-9
    for f in (0, 0.25, 0.5, 0.75, 1.0))
# 4-2. 다음 주기 시작에 출력하는 이산 루프: loop_disc(delay = T)와 식으로 직접 쓴 loop_disc_explicit이 같은가
w = np.linspace(1.0, 0.999 * np.pi / 0.010, 2000)
dev = float(np.max(np.abs(m.loop_disc(w, 0.010, 0.010) - m.loop_disc_explicit(w, 0.010, B, 0.0))))
out["max_abs_diff_disc_explicit"] = dev
checks["check_disc_explicit"] = dev < 1e-12
# 4-3. 정확한 이산 플랜트 식: 출력이 kT + d에 바뀌는 입력으로 solve_ivp를 돌려 샘플 값을 비교
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(20):
    T = float(rng.uniform(0.002, 0.020))
    d = float(rng.uniform(0.0, T))
    u = rng.normal(size=8)
    al = np.exp(-T / m.TAU)
    b0 = 1 - np.exp(-(T - d) / m.TAU)
    b1 = np.exp(-(T - d) / m.TAU) * (1 - np.exp(-d / m.TAU))
    x_ref, x_dis = [0.0], [0.0]
    for k in range(1, len(u)):
        uk_prev = u[k - 1]  # 구간 [(k-1)T + d, kT)에서 유지되는 입력
        u_before = u[k - 2] if k >= 2 else 0.0  # 구간 [(k-1)T, (k-1)T + d)에서 유지되는 한 입력 앞의 값. 첫 주기는 0으로 둔다.
        x = x_ref[-1]
        for (t0, t1, uu) in (((k - 1) * T, (k - 1) * T + d, u_before), ((k - 1) * T + d, k * T, uk_prev)):
            if t1 > t0:
                x = float(solve_ivp(lambda t, y: [(-y[0] + uu) / m.TAU], (t0, t1), [x], rtol=1e-12, atol=1e-14).y[0, -1])
        x_ref.append(x)
        x_dis.append(al * x_dis[-1] + b0 * uk_prev + b1 * u_before)
    worst = max(worst, max(abs(a - b) for a, b in zip(x_ref, x_dis)))
out["max_err_plant_solve_ivp"] = worst
checks["check_plant_solve_ivp"] = worst < 1e-9
# 4-4. 계산 결과 확인: 수치 범위 안에서 식과 맞는가는 값으로 저장하고 여기서는 단조 감소와 부호만 본다
checks["check_pm_positive_T10"] = all(r["pm_A"] > 0 and r["pm_B"] > 0 for r in out["table_T10"])
checks["check_A_equals_B_at_tc_T"] = abs(out["table_T10"][-1]["diff"]) < 1e-9
out.update({k: bool(v) for k, v in checks.items()})
with open("results_f4b5.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("table_T10",)}, ensure_ascii=False, indent=1))
print("모든 check_* 통과:", all(bool(v) for v in checks.values()))
