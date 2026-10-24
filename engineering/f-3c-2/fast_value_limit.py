TAU_SLOW = 0.010
TAU_FAST = 0.002
T = 0.060

for dt_ms in [1, 3, 4, 5, 10, 15]:
    dt = dt_ms / 1000
    slow = fast = 1.0
    for _ in range(round(T / dt)):
        slow = slow + dt * (-slow / TAU_SLOW)
        fast = fast + dt * (-fast / TAU_FAST)
    m_fast = 1 - dt / TAU_FAST
    print(f"DT={dt_ms:2d}ms M_FAST={m_fast:+.2f} |FAST|={abs(fast):9.3f} |SLOW|={abs(slow):.6f}")
