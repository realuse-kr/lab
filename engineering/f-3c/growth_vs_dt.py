"""한 줄기로 줄어드는 값을 Forward Euler로 계산할 때 간격(DT)에 따라 크기가 어떻게 변하는지 본다.

식은 y' = -y/TAU이고 시작값은 y(0) = 1이다. Forward Euler는 한 걸음마다 y에
M = 1 - DT/TAU를 곱한다. |M|이 1보다 작으면 값이 줄고, 1보다 크면 커진다.
"""
TAU = 0.010
STEPS = 10

for dt_ms in [5, 10, 19, 20, 21, 30]:
    dt = dt_ms / 1000
    y = 1.0
    for _ in range(STEPS):
        y = y + dt * (-y / TAU)
    print(f"DT={dt_ms:2d}ms  M={1 - dt / TAU:+.2f}  |y|={abs(y):9.3f}")
