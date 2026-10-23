# F-3C 간격이 얼마나 크면 계산이 터질까

시간상수가 10ms인 값을 Forward Euler로 계산할 때 간격에 따라 값의 크기가 어떻게 변하는지 확인하는 코드다.

## 파일
- `growth_vs_dt.py`: 간격 여섯 개에서 10걸음 뒤 값의 크기를 출력한다. Python 3 표준 라이브러리만 쓴다.
- `save_results.py`: 같은 계산에 걸음별 크기(15·20·25ms)와 연습문제 정답을 더해 `results_f3c.json`으로 저장한다.
- `results_f3c.json`: 실행 결과.

## 실행
```
python growth_vs_dt.py
python save_results.py
```
Python 3.13.17에서 실행해 확인했다.

## 핵심
식은 y' = -y/TAU, 시작값은 y(0) = 1이다. Forward Euler는 한 걸음마다 y에 M = 1 - DT/TAU를 곱한다. |M| < 1이면 값이 줄고 |M| > 1이면 커진다. 한계 간격은 DT = 2 x TAU다. 한계에서는 M이 -1이라 크기가 그대로이고 부호만 바뀐다.

## 연습 정답
`TAU = 0.050`으로 바꾸면 한계 간격은 100ms다. 10걸음 뒤 값의 크기는 DT = 99ms에서 0.817, DT = 101ms에서 1.219다.

## 참고
- 이 결과는 값이 하나뿐인 식에서 나온 것이다.
- MATLAB 명령은 넣지 않았다.
- 라이선스는 저장소의 MIT를 따른다.
