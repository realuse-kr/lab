# F-4: s를 z로 바꾸는 세 규칙 (Forward Euler, Backward Euler, Tustin)

가상 파라미터 시뮬레이션이다. 실제 장치의 제어기가 아니다.

- 실행: 저장소 루트에서 `python python/f2_control/discretization_map.py`
- 다른 극점으로 돌리기: `python python/f2_control/discretization_map.py 내_params.json 내_results.json` (인자를 생략하면 `params/f4_poles.json`을 읽고 `results/F-4/results.json`에 쓴다)
- 입력: `params/f4_poles.json` (극점 6개, 샘플링 주기, 주기 표의 값)
- 출력: 표준출력 요약과 `results/F-4/results.json` (엄격한 JSON. 계산할 수 없는 값은 null)
- 필요: numpy, scipy
- Forward Euler의 한계 주기 식은 Re s < 0인 극점에만 쓴다. s=0이나 불안정 극점은 한계 계산에서 빼고 results.json의 `excluded_from_limit`에 남긴다. s=0의 z는 정확한 대응과 세 규칙 모두 1(단위원 경계)이라 단위원 안 개수에서는 밖으로 센다. Re s < 0인 극점이 하나도 없으면 이유를 적고 멈춘다.
- MATLAB: `matlab/f2_control/discretization_map.m`은 같은 식을 옮겨 적은 파일이다. MATLAB 파일은 실행하지 못했다(Python 출력만 확인).
- 연습문제와 풀이: `exercises/F-4.md`
