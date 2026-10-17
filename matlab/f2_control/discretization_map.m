% F-4: s를 z로 바꾸는 세 규칙(Forward Euler, Backward Euler, Tustin)이 극점을 어디로 옮기는가.
% 가상 파라미터 시뮬레이션이다. 실제 장치의 제어기가 아니다.
% 이 파일은 python/f2_control/discretization_map.py와 같은 식을 옮겨 적은 것이다.
T = 0.02;                                   % 샘플링 주기 [s]
poles = [-4, -25, -3+30i, -3-30i, -10+50i, -10-50i];

rules.exact          = @(s) exp(s*T);
rules.forward_euler  = @(s) 1 + s*T;
rules.backward_euler = @(s) 1 ./ (1 - s*T);
rules.tustin         = @(s) (1 + s*T/2) ./ (1 - s*T/2);

names = fieldnames(rules);
for k = 1:numel(names)
    z = rules.(names{k})(poles);
    fprintf('%-15s |z| = %s  안 %d/6\n', names{k}, mat2str(abs(z), 3), sum(abs(z) < 1));
end

sigma = -real(poles); omega = imag(poles);
limit = 2*sigma ./ (sigma.^2 + omega.^2);   % Forward Euler의 극점별 한계 주기
fprintf('Forward Euler 한계 주기 %.4f s (%.1f Hz)\n', min(limit), 1/min(limit));
