"""F-4: s를 z로 바꾸는 세 규칙(Forward Euler, Backward Euler, Tustin)이 극점을 어디로 옮기는가.

가상 파라미터 시뮬레이션이다. 실제 장치의 제어기가 아니다.
실행: 저장소 루트에서  python python/f2_control/discretization_map.py [params.json] [results.json]
     인자를 생략하면 params/f4_poles.json을 읽고 results/F-4/results.json에 쓴다.
출력: 표준출력 요약과 results.json
필요: numpy, scipy
주의: Forward Euler의 한계 주기 식은 Re s < 0인 극점에만 쓴다. 그 밖의 극점(s=0, 불안정 극점)은
      한계 계산에서 빼고 results.json의 excluded_from_limit에 남긴다.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import cont2discrete

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "params" / "f4_poles.json"
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "F-4" / "results.json"
PARAMS = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))

T = PARAMS["T_s"]
POLES = np.array([complex(re, im) for re, im in PARAMS["poles"]])
N = len(POLES)

# 규칙: s를 받아 z를 돌려준다. exact는 샘플링을 정확히 반영하는 매핑 z = e^{sT}.
RULES = {
    "exact": lambda s, T: np.exp(s * T),
    "forward_euler": lambda s, T: 1 + s * T,
    "backward_euler": lambda s, T: 1 / (1 - s * T),
    "tustin": lambda s, T: (1 + s * T / 2) / (1 - s * T / 2),
}
SCIPY_METHOD = {"forward_euler": "euler", "backward_euler": "backward_diff", "tustin": "bilinear"}


def inside_count(z):
    return int((np.abs(z) < 1).sum())


def fwd_euler_limit(poles):
    """Forward Euler가 극점 -sigma ± j*omega를 단위원 안에 두는 최대 주기: 2*sigma / (sigma^2 + omega^2).

    sigma > 0(Re s < 0)인 극점에만 계산한다. s=0, 불안정 극점, 허수축 위 극점은 nan으로 둔다.
    """
    sigma = -poles.real
    omega = poles.imag
    out = np.full(len(poles), np.nan)
    ok = sigma > 0
    out[ok] = 2 * sigma[ok] / (sigma[ok] ** 2 + omega[ok] ** 2)
    return out


def clean(obj):
    """results.json이 엄격한 JSON이 되도록 nan, inf를 None으로 바꾼다."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def scipy_poles(poles, T, method, alpha=None):
    """극점이 poles인 C(s) = K / prod(s - p)를 cont2discrete로 옮겨 z 분모의 근을 구한다."""
    den = np.poly(poles).real
    num = np.array([den[-1] if den[-1] != 0 else 1.0])  # 직류 이득 1(s=0 극점이 있으면 분자 1)
    kw = {"alpha": alpha} if alpha is not None else {}
    num_d, den_d, _ = cont2discrete((num, den), T, method=method, **kw)
    return np.roots(np.atleast_1d(den_d))


def max_match_error(expected, found):
    """expected의 각 점에서 found 중 가장 가까운 점까지 거리의 최댓값(개수가 같은지도 확인)."""
    assert len(expected) == len(found)
    return float(max(np.min(np.abs(found - e)) for e in expected))


def run():
    out = {"T_s": T, "poles": [[p.real, p.imag] for p in POLES], "methods": {}}

    # 1) 극점별 z, |z|, 안 / 밖
    for name, f in RULES.items():
        z = f(POLES, T)
        out["methods"][name] = {
            "z": [[float(v.real), float(v.imag)] for v in z],
            "abs": [float(v) for v in np.abs(z)],
            "inside": [bool(v) for v in (np.abs(z) < 1)],
            "inside_count": inside_count(z),
            "max_abs": float(np.abs(z).max()),
            "max_abs_pole_index": int(np.abs(z).argmax()),
        }

    # 2) Forward Euler 한계 주기: Re s < 0인 극점에만 계산한다
    lim = fwd_euler_limit(POLES)
    valid = ~np.isnan(lim)
    if not valid.any():
        raise SystemExit("한계 주기를 계산할 수 있는 극점(Re s < 0)이 없다. params 파일의 극점을 확인하라.")
    t_max = float(lim[valid].min())
    out["forward_euler_limit"] = {
        "per_pole": [float(v) for v in lim],  # 제외된 극점은 nan이고 저장할 때 null이 된다
        "T_max": t_max,
        "fs_min_hz": 1 / t_max,
        "limiting_pole_index": int(np.flatnonzero(valid)[lim[valid].argmin()]),
    }
    out["excluded_from_limit"] = [
        {"index": int(i), "pole": [float(POLES[i].real), float(POLES[i].imag)],
         "reason": "Re s >= 0 (s=0, 허수축 위 또는 불안정 극점)"}
        for i in np.flatnonzero(~valid)
    ]
    # 한계 바로 아래와 위에서 안에 남는 극점 수를 실제로 센다
    below = RULES["forward_euler"](POLES, 0.99 * t_max)
    above = RULES["forward_euler"](POLES, 1.01 * t_max)
    out["forward_euler_limit"]["inside_at_0.99x"] = inside_count(below)
    out["forward_euler_limit"]["inside_at_1.01x"] = inside_count(above)

    # 3) 주기 표(Forward Euler) + Backward Euler, Tustin
    rows = []
    for Ts in PARAMS["sweep_T"]:
        r = {"T": Ts, "fs_hz": 1 / Ts}
        for name in ("forward_euler", "backward_euler", "tustin"):
            z = RULES[name](POLES, Ts)
            r[name + "_inside"] = inside_count(z)
            r[name + "_max_abs"] = float(np.abs(z).max())
        rows.append(r)
    out["sweep_table"] = rows

    # 4) 넓은 주기 범위에서 Backward Euler와 Tustin이 6개를 모두 안에 두는가
    grid = np.logspace(-4, 1, 501)
    wide = {"T_min": float(grid[0]), "T_max": float(grid[-1]), "points": int(len(grid))}
    for name in ("backward_euler", "tustin"):
        counts = [inside_count(RULES[name](POLES, g)) for g in grid]
        maxabs = [float(np.abs(RULES[name](POLES, g)).max()) for g in grid]
        wide[name] = {"min_inside": int(min(counts)), "max_abs_over_sweep": float(max(maxabs))}
    out["wide_sweep"] = wide

    # 5) SciPy cont2discrete와 대조
    sc = {}
    for name, method in SCIPY_METHOD.items():
        mapped = RULES[name](POLES, T)
        found = scipy_poles(POLES, T, method)
        sc[method] = {"max_pole_error": max_match_error(mapped, found)}
    sc["zoh"] = {"max_pole_error": max_match_error(RULES["exact"](POLES, T), scipy_poles(POLES, T, "zoh"))}
    alpha = 0.3
    gbt_expected = (1 + (1 - alpha) * POLES * T) / (1 - alpha * POLES * T)
    sc["gbt_alpha_0.3"] = {"max_pole_error": max_match_error(gbt_expected, scipy_poles(POLES, T, "gbt", alpha))}
    out["scipy_check"] = sc

    # 6) 표의 영역 주장 확인: 왼쪽 반평면의 무작위 점 20만 개를 규칙별로 옮겨 영역 조건을 검사한다
    rng = np.random.default_rng(0)
    s = -rng.uniform(1e-3, 500, 200_000) + 1j * rng.uniform(-500, 500, 200_000)
    out["region_check"] = {
        "samples": int(len(s)),
        "seed": 0,
        "forward_euler_re_z_lt_1": bool(np.all((1 + s * T).real < 1)),
        "backward_euler_in_disc_center_half_radius_half": bool(np.all(np.abs(1 / (1 - s * T) - 0.5) < 0.5)),
        "tustin_in_unit_disc": bool(np.all(np.abs((1 + s * T / 2) / (1 - s * T / 2)) < 1)),
        "exact_in_unit_disc": bool(np.all(np.abs(np.exp(s * T)) < 1)),
    }

    # 7) 연속 시간에서 불안정한 극점이 규칙별로 어디로 가는가
    u = complex(*PARAMS["unstable_example_pole"])
    out["unstable_example"] = {
        "s": [u.real, u.imag],
        "T_s": T,
        "abs": {k: float(abs(f(u, T))) for k, f in RULES.items() if k != "exact"},
        "exact_abs": float(abs(np.exp(u * T))),
        "backward_euler_inside_if_T_a_gt": 2.0,
    }
    return out


def main():
    res = run()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(clean(res), ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")

    names = {"exact": "정확 e^(sT)", "forward_euler": "Forward Euler", "backward_euler": "Backward Euler", "tustin": "Tustin"}
    print(f"T = {T} s, 극점 {N}개")
    for key, label in names.items():
        m = res["methods"][key]
        print(f"{label:14s} |z| =", np.round(m["abs"], 3), f" 안 {m['inside_count']}/{N}")
    lim = res["forward_euler_limit"]
    print(f"Forward Euler 한계 주기 {lim['T_max']:.6f} s ({lim['fs_min_hz']:.1f} Hz)")
    if res["excluded_from_limit"]:
        print("한계 계산에서 뺀 극점:", [e["pole"] for e in res["excluded_from_limit"]])


if __name__ == "__main__":
    main()
