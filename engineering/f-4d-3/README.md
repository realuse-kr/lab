# f-4d-3: 폐루프에 한 샘플 지연이 들어갔는지 극점으로 확인하기

시간 상수 0.1초인 1차 플랜트 P(s) = 1/(0.1s + 1)를 ZOH 모델로, PI 제어기 C(s) = 2 + 40/s를 Forward Euler, Backward Euler, Tustin 규칙으로 이산화해 피드백으로 잇고, 제어기 출력이 플랜트에 들어가기 전에 한 샘플 지연을 거치는 경우와 거치지 않는 경우의 스텝 응답에서 폐루프 극점을 읽는다. 읽은 극점은 특성다항식의 근(해석값)과 비교한다.

## 파일

| 파일 | 내용 |
| --- | --- |
| delay_check.py | 스텝 응답 생성, 응답에서 극점 읽기, 특성다항식의 근 계산. 실행하면 주기 20 ms의 여섯 경우를 출력한다. |
| save_results.py | 차수별 잔차, 폐루프 전달함수와의 응답 비교, 주기별 계산, 연습문제 답을 results_f4d3.json에 저장한다. |
| results_f4d3.json | save_results.py의 실행 결과. |

## 실행

```
pip install numpy scipy
python delay_check.py
python save_results.py
```

delay_check.py는 NumPy만 쓰고 save_results.py는 SciPy도 쓴다. Python 3.13, NumPy 2.5.3, SciPy 1.18.1에서 실행한 결과다. 지연 샘플 수는 `step_response(rule, delay)`와 `char_poles(rule, delay)`의 delay 인자이고, 주기와 게인은 delay_check.py 둘째 줄의 상수다. 규칙 계수 RULES가 그 상수에서 계산되므로 상수만 바꾸면 된다.

## 연습문제

1. delay_check.py 둘째 줄의 T를 0.01(10 ms)로 바꾸고 ZOH, Forward Euler에서 지연 없음과 한 샘플 지연의 극점 개수와 값을 구하라.
2. T = 0.02로 되돌리고 `read_poles(step_response("ZOH, Forward Euler", 2))`로 지연을 두 샘플로 늘렸을 때 읽은 극점 개수와 가장 큰 극점의 크기를 구하라.
3. T를 0.04(40 ms)로 바꾸고 세 규칙에서 지연 없음과 한 샘플 지연의 가장 큰 극점 크기를 구하라. 1보다 큰 경우는 어느 것인가.

## 풀이

1. 지연 없음은 극점 2개 0.8573±0.1330j, 한 샘플 지연은 극점 3개 0.8508±0.1598j와 0.2032다.
2. 극점 4개 −0.4544, 0.4889, 0.8921±0.4283j를 읽고, 가장 큰 극점의 크기는 0.9896이다.
3. 지연 없음에서 한 샘플 지연으로 갈 때 ZOH, Forward Euler는 0.7338에서 1.0731, Backward Euler는 0.4596에서 1.1207, Tustin은 0.5241에서 1.0638이다. 지연 없음은 세 규칙 모두 1보다 작고, 한 샘플 지연은 세 규칙 모두 1보다 크다.

## 실행하지 않은 것

본문에 실린 MATLAB 명령은 실행하지 않았다. pole은 MathWorks 문서에서, c2d·feedback·tf('z', Ts)는 앞 편에서 확인한 이름과 인자를 썼고, tf(num, den, Ts) 형태는 MathWorks Answers의 사용례로 확인했다. 본문의 숫자와 그림은 모두 이 저장소의 Python 코드 실행값이다.
