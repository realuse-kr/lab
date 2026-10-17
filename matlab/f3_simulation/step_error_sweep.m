zeta = 0.32; wn = 11; t_end = 2;
sigma = zeta*wn; wd = wn*sqrt(1 - zeta^2);
f = @(x) [x(2); wn^2*(1 - x(1)) - 2*sigma*x(2)];   % x = [위치; 속도], 스텝 입력 1
exact = @(t) 1 - exp(-sigma*t).*(cos(wd*t) + sigma/wd*sin(wd*t));

for h = [0.04 0.02 0.01 0.005 0.0025 0.00125 0.000625]
    n = round(t_end/h);  t = (0:n)*h;
    xe = [0; 0];  xr = [0; 0];  e_eu = 0;  e_rk = 0;
    for k = 1:n
        xe = xe + h*f(xe);                                  % 전진 Euler
        k1 = f(xr); k2 = f(xr + h/2*k1); k3 = f(xr + h/2*k2); k4 = f(xr + h*k3);
        xr = xr + h/6*(k1 + 2*k2 + 2*k3 + k4);              % RK4
        e_eu = max(e_eu, abs(xe(1) - exact(t(k+1))));
        e_rk = max(e_rk, abs(xr(1) - exact(t(k+1))));
    end
    fprintf('%8.6f  %.3e  %.3e\n', h, e_eu, e_rk);
end
