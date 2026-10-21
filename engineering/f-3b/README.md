# 간격을 키우면 오버슈트는 얼마나 틀릴까

가상 2차 시스템(고유각주파수 10 rad/s, 감쇠비 0.3)의 스텝 응답을 Forward Euler로 계산할 때,
적분 간격에 따라 읽히는 오버슈트가 해석값(37.23%)에서 얼마나 벗어나는지 본다.

## 실행

Python 3와 NumPy가 필요하다. MATLAB은 쓰지 않는다.

```
python overshoot_vs_dt.py   # 간격별 오버슈트와 오차를 출력한다
python save_results.py      # 같은 계산을 results_f3b.json으로 저장한다
```

## 파일

- `overshoot_vs_dt.py`: 계산 본체(함수 없는 짧은 스크립트). 간격별 오버슈트와 해석값 대비 오차를 출력한다.
- `save_results.py`: 결과 표와 오차가 TOL이 되는 간격(이분법)을 json으로 저장한다.
- `results_f3b.json`: 위 두 스크립트를 실행한 결과 파일.

## 연습문제

`save_results.py`의 `TOL`을 0.5로 바꿔 실행하면 오차가 0.5%p 이내인 가장 큰 간격은 얼마인가.

정답: 약 0.81ms(`dt_limit_ms`가 0.809 근처로 나온다). `TOL`이 1.0일 때는 1.607ms였다.
