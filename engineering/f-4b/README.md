# engineering/f-4b — 샘플링 주기를 늘리면 제어 성능은 얼마나 떨어질까

시간 상수 0.1초인 1차 플랜트(ZOH 모델)와 Kp 2, Ki 40인 PI 제어기를 Forward Euler, Backward Euler, Tustin 세 규칙으로 이산화하고, 샘플링 비율(샘플링 주파수 ÷ 연속 시간 폐루프 −3 dB 대역폭)을 바꿔 가며 오버슈트와 연속 시간 응답과의 최대 편차를 계산한다. 가상 파라미터 계산이며 실제 기기의 값이 아니다.

## 파일
- `sweep_check.py`: 본문 코드와 같다. 대역폭을 구하고 비율 5, 10, 20, 40, 80의 결과를 출력한다. NumPy와 SciPy가 필요하다.
- `save_results.py`: 비율 4~160의 17개 값, 안정 한계, 연습문제 답을 `results_f4b.json`으로 저장한다. `sweep_check.py`와 같은 폴더에서 실행한다.
- `results_f4b.json`: 본문 표와 그림의 숫자가 나온 결과 파일이다.

## 실행
```
python -I sweep_check.py
python save_results.py
```
Python 3.13, NumPy 2.5.3, SciPy 1.18.1에서 실행했다. `save_results.py`는 같은 폴더의 `sweep_check.py`를 import하므로 `-I` 없이 실행한다. MATLAB 명령은 본문에만 있고 실행하지 않았다.

## 연습문제
상수를 바꾸려면 `sweep_check.py` 3번째 줄의 `KP, KI, TAU, TEND`만 고치면 된다. 폐루프 전달함수 `NUM`, `DEN`은 같은 줄의 상수에서 만들어지고, 대역폭과 샘플링 주기의 환산은 함수가 다시 계산한다. 파일을 바꾼 직후 다시 실행할 때 이전 실행의 캐시가 남아 있으면 `__pycache__` 폴더를 지운다.

1. 샘플링 주기 20 ms는 샘플링 비율로 얼마이고, 세 규칙의 오버슈트와 최대 편차는 얼마인가? (`metrics(0.02)`를 호출한다.)
2. Ki를 20으로 바꾸면 대역폭과 연속 시간 오버슈트는 어떻게 달라지고, 비율 20에서 Forward Euler의 최대 편차는 얼마인가?
3. Forward Euler의 최대 편차는 비율 40에서 80으로 올릴 때 몇 배로 줄어드는가?

## 풀이
1. 비율은 10.6이다. 오버슈트는 Forward Euler 17.4%, Backward Euler 8.4%, Tustin 12.4%이고 최대 편차는 0.118, 0.180, 0.124다.
2. 대역폭은 4.73 Hz에서 3.18 Hz로 줄고 연속 시간 응답이 목표값 1을 넘지 않아 연속 시간 오버슈트는 0%다. 비율 20에서 Forward Euler의 최대 편차는 0.043이다.
3. 2.06배다.
