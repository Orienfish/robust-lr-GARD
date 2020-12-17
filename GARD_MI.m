function [theta_GARD_MI, jk_list] = GARD_MI(X, y, n, m, eps_0)
% GARD algorithm with block matrix inversion lemma
% Compute initial residual by projecting y onto R(X)
k = 0;
Aac = X;
In = eye(n);
Aac_inv = inv(Aac'*Aac);
phi = Aac_inv * Aac';             % Projection matrix
z_opt = phi * y;                  % Initial opt. projection
rk = y - Aac * z_opt;             % Initial residual
% Start the loop until the residual is small enough
jk_list = zeros(1, n);            % Record jk in each round
norm_rk_list = zeros(1, n);       % Record norm(rk) in each round
while norm(rk) > eps_0
    k = k + 1;
    [val, jk] = max(abs(rk));
    jk_list(k) = jk;
    h = phi * In(:, jk);
    tmp = Aac * h;
    lambda = 1 / (1 - tmp' * tmp);
    v = In(:, jk) - tmp;
    % Update
    Aac = [Aac, In(:, jk)];
    % Aac_inv = [Aac_inv + lambda * h * h', -lambda * h; ...
    %           -lambda * h', lambda];
    phi = [phi - lambda * h * v'; lambda * v'];
    z_opt = phi * y;
    rk = y - Aac * z_opt;
    norm_rk_list(k) = norm(rk);
end
theta_GARD_MI = z_opt(1:m);c
jk_list = jk_list(1:k);
end