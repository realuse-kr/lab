# F-4B2 스텝 불변 이산화는 PI 제어기의 무엇을 연속 시간과 같게 만들까

PI 제어기 C(s) = Kp + Ki/s (Kp=2, Ki=40)를 스텝 불변 이산화로 바꾸고, 스텝 입력·램프 입력·폐루프에서 연속 시간 결과와 얼마나 같은지 비교하는 Python 코드와 결과 파일이다.

## 파일

- `step_invariant.py`: 네 가지 규칙(스텝 불변, Forward Euler, Backward Euler, Tustin)의 계수와 제어기 출력 차이
- `save_results.py`: 폐루프 편차, 샘플링 비율 표, 연습문제 값을 계산해 `results_f4b2.json`에 저장
- `results_f4b2.json`: 글에 나온 숫자의 출처

## 실행

```
python step_invariant.py
python save_results.py
```

`save_results.py`는 `step_invariant.py`를 가져오므로 `python -I`로 실행하지 않는다. Python 3.13, NumPy 2.5.3, SciPy 1.18.1에서 실행했다. 실행 시간은 1초 안팎이다.

## 연습문제

1. 샘플링 주기 T를 10 ms로 바꾸면 스텝 불변 이산화의 램프 입력 차이(2초 시점)는 얼마인가.
2. Ki를 20으로 바꾸고 T = 20 ms이면 스텝 불변 계수 (c0, c1)은 무엇인가.
3. T = 2.5 ms이면 스텝 불변 이산화의 램프 입력 차이(2초 시점)는 얼마인가.

## 정답

1. 약 -0.4. 차이는 -Ki·T·t/2이므로 -40×0.01×2/2 = -0.4이다.
2. (2.0, -1.6). c0 = Kp, c1 = -Kp + Ki·T = -2 + 20×0.02이다.
3. 약 -0.1. -40×0.0025×2/2 = -0.1이다.

`results_f4b2.json`의 `exercises` 항목에 같은 값이 들어 있다.

## 실행하지 않은 것

글에 보인 MATLAB 명령(`c2d`)은 이 저장소에서 실행하지 않았다. 계수는 Python으로 계산했다.
