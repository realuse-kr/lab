TAU_A = 0.010
TAU_B = 0.002
T = 0.060

for h_ms in [1, 3, 4, 5, 10, 15]:
    h = h_ms / 1000
    a = b = 1.0
    for _ in range(round(T / h)):
        a = a + h * (-a / TAU_A)
        b = b + h * (-b / TAU_B)
    g_a = 1 - h / TAU_A
    g_b = 1 - h / TAU_B
    print(f"h={h_ms:2d}ms g_A={g_a:+.2f} g_B={g_b:+.2f} |a|={abs(a):.4f} |b|={abs(b):.3f}")
