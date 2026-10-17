"""목표 오차 탐색 검증: 각 목표에서 N-1은 목표를 넘고, N과 N+1은 목표 이내인지 확인한다.
먼저 f3_step_error.py를 실행해 results/F-3/results.json을 만든 뒤 돌린다. 결과는 results/F-3/target_check.json."""
import json

from f3_step_error import ROOT, make_system, max_error

res = json.loads((ROOT / "results" / "F-3" / "results.json").read_text(encoding="utf-8"))
p = json.loads((ROOT / "params" / "f3_params.json").read_text(encoding="utf-8"))
f, sg, wd = make_system(p["plant_second_order"]["zeta"], p["plant_second_order"]["wn_rad_s"])
out = []
for r in res["targets"]:
    row = {"target": r["target_err"]}
    for name in ("euler", "rk4"):
        n = r[name]["n_steps"]
        e_prev = max_error(name, f, sg, wd, p["t_end_s"], n - 1)[0]
        e_next = max_error(name, f, sg, wd, p["t_end_s"], n + 1)[0]
        row[name] = {"n": n, "err_n_minus_1": e_prev, "err_n": r[name]["achieved_err"], "err_n_plus_1": e_next,
                     "ok": e_prev > r["target_err"] >= r[name]["achieved_err"] and e_next <= r["target_err"]}
    out.append(row)
(ROOT / "results" / "F-3" / "target_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for row in out:
    print(row["target"], row["euler"]["ok"], row["rk4"]["ok"])
