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

% Generate random ground truth X and linear weights theta
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
% Generate observation vector y
y = X * theta_0 + u_0 + eta;

