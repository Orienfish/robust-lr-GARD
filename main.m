%% Initialization
% Important parameters
m = 50; % 100, 170      % Dimension of observations
n = 100;                % Number of observations, n > m
frac = 0.05;            % Outlier fraction
s = floor(n * frac);    % Number of outlier indexes

X_min = -1;             % Min bound of X
X_max = 1;              % Max bound of X
theta_mean = 0;         % Mean of the normal distribution for theta
theta_sigma = 5;        % Standard deviation of the normal distribution for theta
eps_0 = 3;              % Inlier noise bound
outErr = 25;            % Outlier noise abs value

% Generate random observation X and linear weights theta
X = repmat(X_min, n, m);
X = X + rand(n, m) * (X_max - X_min);
theta_0 = normrnd(theta_mean, theta_sigma, m, 1);
% Generate bounded inlier noise
eta = normrnd(0, 1, n, 1);
eta = min(eta, eps_0);
eta = max(eta, -eps_0);
% Generate s-sparse outlier noise
rdn_idx = randsample(n, s);
u_0 = zeros(n, 1);
for i=1:s
    sign = -1 + 2 * (rand() > 0.5);
    u_0(rdn_idx(i)) = outErr * sign;
end
% Generate final vector y
y = X * theta_0 + u_0 + eta;

% Least square
P = inv(X'*X) * X'; % Projection matrix
theta_LS = P * y;   % x* in least square
fprintf('MSE of LS: %f\n', MSE(theta_0, theta_LS));

% GARD
theta_GARD = GARD(X, y, n, m, eps_0);
fprintf('MSE of GARD: %f\n', MSE(theta_0, theta_GARD));

function theta_GARD = GARD(X, y, n, m, eps_0)
% GARD algorithm
% Compute initial residual by projecting y onto R(X)
k = 0;
Aac = X;
In = eye(n);
z_opt = inv(Aac'*Aac) * Aac' * y; % Initial opt. projection
rk = y - Aac * z_opt;             % Initial residual
% Start the loop until the residual is small enough
jk_list = zeros(1, n);            % Record jk in each round
norm_rk_list = zeros(1, n);         % Record norm(rk) in each round
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
end


function err = MSE(v1, v2)
% Calculate the mean square error between v1 and v2
err = (v1 - v2)' * (v1 - v2);
err = sum(err) / size(v1, 1);
end
