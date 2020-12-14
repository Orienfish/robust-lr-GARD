%% Initialization
% Important parameters
m = 50; % 100, 170      % Dimension of observations
n = 600;                % Number of observations, n > m
frac = 0.4;            % Outlier fraction
s = floor(n * frac);    % Number of outlier indexes

X_min = -1;             % Min bound of X
X_max = 1;              % Max bound of X
theta_mean = 0;         % Mean of the normal distribution for theta
theta_sigma = 5;        % Standard deviation of the normal distribution for theta
eps_0 = 20;              % Inlier noise bound
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
tic;
P = inv(X'*X) * X'; % Projection matrix
theta_LS = P * y;   % x* in least square
time = toc;
fprintf('MSE of LS: %f dB, time: %f s\n', MSE(theta_0, theta_LS), time);

% MATLAB least square function regress
%theta_regress = regress(y, X);
%fprintf('MSE of regress: %f dB\n', MSE(theta_0, theta_regress));

% M-estimator
tic;
theta_Mest = robustfit(X, y);
theta_Mest = theta_Mest(2:end); % Omit the const
toc;
fprintf('MSE of M-estimator: %f dB, time: %f s\n', MSE(theta_0, theta_Mest), time);

% GARD
tic;
theta_GARD = GARD(X, y, n, m, eps_0);
time = toc;
fprintf('MSE of GARD: %f dB, time: %f s\n', MSE(theta_0, theta_GARD), time);

% GARD with QR factorization
tic;
theta_GARD_QR = GARD_QR(X, y, n, m, eps_0);
time = toc;
fprintf('MSE of GARD QR: %f dB, time: %f s\n', MSE(theta_0, theta_GARD_QR), time);

% GARD with Matrix Inversion Lemma
tic;
theta_GARD_MI = GARD_MI(X, y, n, m, eps_0);
time = toc;
fprintf('MSE of GARD MI: %f dB, time: %f s\n', MSE(theta_0, theta_GARD_MI), time);

function errlog = MSE(v1, v2)
% Calculate the mean square error between v1 and v2
err = (v1 - v2)' * (v1 - v2);
err = sum(err) / size(v1, 1);
errlog = 10 * log10(err);
end
