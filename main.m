%% Initialization
% Important parameters
m = 50; % 100, 170      % Dimension of observations
n = 100;                % Number of observations, n > m
frac = 0.05;            % Outlier fraction
s = floor(n * frac);    % Number of outlier indexes
eps_0 = 3;              % Inlier noise bound

% Generate random observation matrix X and linear weights theta
X = repmat(-1, n, m);
X = X + rand(n, m) * 2;
theta_0 = normrnd(0, 5, m);
eta = normrnd(0, 1, n);   % Inlier noise
% ramdomly generate the s-sparse signal with length n
comb = combnk(1:n, s);
y = X * theta_0 + u_0 + eta;
