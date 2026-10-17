# F-3 시간 간격을 줄이면 오차는 얼마나 줄까

REALUSE 엔지니어링 기초 시리즈 F-3의 코드와 결과. 전진 Euler(1차)와 RK4(4차)로 가상 2차계(ζ=0.32, ωn=11 rad/s)의 스텝 응답을 계산하고, 간격을 절반씩 줄일 때 오차가 줄어드는 배율과 같은 목표 오차에 필요한 간격·우변 계산 횟수를 비교한다. 모두 가상 파라미터 시뮬레이션이며 실제 기기의 측정값이 아니다.

## 파일 (저장소 루트 기준)
| 경로 | 내용 |
|---|---|
| `python/f3_simulation/step_error_sweep.py` | 글의 Python 코드 블록과 같은 파일(numpy만 필요) |
| `python/f3_simulation/step_error_sweep_output.txt` | 글의 출력 블록과 같은 실행 결과 |
| `matlab/f3_simulation/step_error_sweep.m` | 글의 MATLAB 접기 블록과 같은 파일 |
| `python/f3_simulation/f3_step_error.py` | 간격 스윕, 목표 오차별 필요 간격, 차수 맞춤 → `results/F-3/results.json` |
| `python/f3_simulation/f3_exercises.py` | 연습문제 1~3 해답 → `results/F-3/exercises_results.json` |
| `python/f3_simulation/f3_check_targets.py` | 목표 오차 탐색 검증 → `results/F-3/target_check.json` |
| `python/f3_simulation/requirements.txt` | numpy, scipy |
| `params/f3_params.json` | 가상 파라미터, 스윕 간격, 목표 오차 |
| `results/F-3/` | 위 스크립트가 만든 json |
| `exercises/F-3.md` | 연습문제 세 개와 해답 |

## 실행
```
pip install -r python/f3_simulation/requirements.txt

# 글의 코드 그대로
python python/f3_simulation/step_error_sweep.py

# 전체 결과 재생성(약 12초). 순서대로 실행한다
python python/f3_simulation/f3_step_error.py
python python/f3_simulation/f3_exercises.py
python python/f3_simulation/f3_check_targets.py
```
스크립트는 파일 위치를 기준으로 `params/`와 `results/F-3/`를 찾으므로 어느 폴더에서 실행해도 된다.

## 글의 숫자와 파일의 대응
| 글의 내용 | 위치 |
|---|---|
| 표 1(간격별 최대 오차), 표 2(절반 배율), 숫자 카드 2.01배·16.03배 | `results/F-3/results.json`의 `sweep` |
| 직선 맞춤 기울기 1.09, 4.02와 마지막 시각 기준 1.73, 3.82 | 같은 파일의 `fits`, `fits_final_time_error` |
| 표 3(목표 오차별 간격·계산 횟수), 숫자 카드 1,878배 | 같은 파일의 `targets` |
| 닫힌식과 행렬 지수 함수의 차이 2.46e-14 | 같은 파일의 `reference_check` |
| 안정 한계 h < 0.0582 s | 같은 파일의 `euler_stability`, 연습문제 1 |
| 연습문제 1~3 해답 | `results/F-3/exercises_results.json` |

## 정의
- 오차: 0~2 s의 모든 격자점에서 |수치해 − 닫힌식 정답|의 최댓값.
- 비용: 우변 f를 계산한 횟수. 전진 Euler는 한 단계에 1번, RK4는 4번.
- 목표 오차에 필요한 단계 수: 오차가 목표 이하가 되는 가장 작은 정수 N. 사다리와 이분법으로 찾고 `f3_check_targets.py`로 N−1은 초과, N·N+1은 이내인지 확인한다.

## 검증 범위
- Python 코드: 이 구성에서 실제로 실행해 `results/F-3/*.json` 세 개가 처음 계산한 파일과 바이트 단위로 같음을 확인했다(Python 3.13, numpy 2.5.3).
- `step_error_sweep.py`의 출력은 `step_error_sweep_output.txt`와 일치한다.
- MATLAB 코드: 실행 환경이 없어 실행하지 못했다. 코드를 읽어 확인한 수준이며, 출력이 Python과 같은지는 MATLAB에서 돌려 보아야 한다.
