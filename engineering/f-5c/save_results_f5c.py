"""F-5C 결과를 results_f5c.json에 저장한다. 난수가 없어 같은 코드는 같은 값을 낸다."""
import json
import numpy as np
from pid_windup import (KP, KI, TAU, K_PLANT, TS, DR, T_STEP, UMAX, simulate, first_sample_output,
                        t_reach, area_a, windup_formula, metrics)


def stays_at_limit_through_cross(ki):
    """측정값이 설정값에 닿는 샘플까지 계산한 제어 출력이 한도를 넘어 있는지."""
    t, y, u, integ, k0 = simulate("plain", ki=ki)
    cross = k0 + int(np.argmax(y[k0:] >= DR))
    return bool(np.all(u[k0:cross + 1] > UMAX))


# 한도에 붙어 있는 채로 설정값에 닿는 가장 작은 Ki를 이분법으로 찾는다
lo, hi = 3.0, 8.0
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (lo, mid) if stays_at_limit_through_cross(mid) else (mid, hi)

res = {
    "params": {"Kp": KP, "Ki": KI, "tau_s": TAU, "K_plant": K_PLANT, "Ts_s": TS, "dr": DR,
               "t_step_s": T_STEP, "umax": UMAX},
    "first_sample_output": first_sample_output(),
    "t_reach_ms": t_reach() * 1e3,
    "A": area_a(),
    "needed_integ": DR / K_PLANT,
    "ki_min_formula": UMAX / area_a(),
    "ki_min_code": hi,
    "cases": {m: metrics(m) for m in ("ideal", "plain", "stop")},
    "ki_rows": [],
}
for ki in (4.0, 5.0, 6.0, 8.0, 10.0):
    formula = windup_formula(ki)
    code = metrics("plain", ki=ki)
    res["ki_rows"].append({
        "Ki": ki, "formula": formula, "ts_correction": ki * TS * DR / 2,
        "code_integ_at_cross": code["integ_at_cross"],
        "overshoot_ideal": metrics("ideal", ki=ki)["overshoot"],
        "overshoot_plain": code["overshoot"],
        "overshoot_stop": metrics("stop", ki=ki)["overshoot"],
    })
res["excess_plain"] = res["cases"]["plain"]["integ_at_cross"] - res["needed_integ"]
res["excess_formula"] = windup_formula() - res["needed_integ"]
res["after_cross_ms"] = res["cases"]["plain"]["over_limit_ms"] - res["cases"]["plain"]["cross_ms"]
json.dump(res, open("results_f5c.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
