function [theta_GARD, jk_list] = GARD(X, y, n, m, eps_0)
% GARD algorithm
% Compute initial residual by projecting y onto R(X)
k = 0;
Aac = X;
In = eye(n);
z_opt = inv(Aac'*Aac) * Aac' * y; % Initial opt. projection
rk = y - Aac * z_opt;             % Initial residual
% Start the loop until the residual is small enough
jk_list = zeros(1, n);            % Record jk in each round
norm_rk_list = zeros(1, n);       % Record norm(rk) in each round
while norm(rk) > eps_0
    k = k + 1;
    [val, jk] = max(abs(rk));
    jk_list(k) = jk;
    Aac = [Aac, In(:, jk)];
    z_opt = inv(Aac'*Aac) * Aac' * y;
    rk = y - Aac * z_opt;
    norm_rk_list(k) = norm(rk);
end
theta_GARD = z_opt(1:m);
jk_list = jk_list(1:k);
end

