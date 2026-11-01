# f-4d-2: 제어기와 플랜트의 이산화 규칙 비교

시간 상수 0.1초인 1차 플랜트 P(s) = 1/(0.1s + 1)와 PI 제어기 C(s) = 2 + 40/s를 샘플링 주기 T로 이산화해 폐루프 스텝 응답을 연속 폐루프와 비교한다. 플랜트는 항상 ZOH 모델이고, 제어기만 ZOH, Forward Euler, Backward Euler, Tustin 네 규칙으로 바꾼다.

## 파일

| 파일 | 내용 |
| --- | --- |
| controller_rules.py | 제어기 계수와 연속 폐루프 대비 편차 계산. 실행하면 T = 20 ms 결과를 출력한다. |
| save_results.py | 주기별 비교(5, 10, 20, 40 ms), 플랜트 쪽 응답 오차, 검증 계산, 연습문제 답을 results_f4d2.json에 저장한다. |
| results_f4d2.json | save_results.py의 실행 결과. |

## 실행

```
pip install numpy scipy
python controller_rules.py
python save_results.py
```

Python 3.13, NumPy 2.5.3, SciPy 1.18.1에서 실행한 결과다. 연속 폐루프 기준 응답은 controller_rules.py가 TAU, KP, KI에서 직접 계산하므로 상수를 바꿔도 따로 고칠 값이 없다.

## 연습문제

1. `controller_rules.py`의 `deviation_pct`를 T = 0.01(10 ms)로 호출해 네 규칙의 최대 편차를 구하라.
2. `KI`를 20으로 바꾸고 T = 20 ms에서 네 규칙의 계수 c1과 최대 편차를 구하라.
3. PI 제어기에서 ZOH, Forward Euler, Backward Euler, Tustin의 c0, c1을 KP, KI, T로 쓰고, 본문 표(KP 2, KI 40, T 20 ms)와 맞는지 확인하라.

## 풀이

1. T = 10 ms에서 최대 편차는 ZOH와 Forward Euler가 +5.17%(90 ms), Tustin이 +5.70%(60 ms), Backward Euler가 +7.96%(40 ms)다.
2. KI = 20, T = 20 ms에서 c1은 ZOH와 Forward Euler가 −1.6, Backward Euler가 −2.0, Tustin이 −1.8이다. 최대 편차는 ZOH와 Forward Euler가 +5.66%(80 ms), Backward Euler가 +12.38%(40 ms), Tustin이 +8.81%(40 ms)다.
3. ZOH와 Forward Euler는 c0 = KP, c1 = −KP + KI·T다. Backward Euler는 c0 = KP + KI·T, c1 = −KP다. Tustin은 c0 = KP + KI·T/2, c1 = −KP + KI·T/2다. 네 규칙 모두 d1 = −1이다. KP 2, KI 40, T 0.02를 넣으면 c0, c1이 (2.0, −1.2), (2.8, −2.0), (2.4, −1.6)이 되어 본문 표와 같다.

## 실행하지 않은 것

본문에 실린 MATLAB 명령은 MathWorks 문서에서 이름과 인자를 확인했을 뿐 실행하지 않았다. 본문의 숫자와 그림은 모두 이 저장소의 Python 코드 실행값이다.
