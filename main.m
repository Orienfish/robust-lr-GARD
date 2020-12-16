%% Initialization
% Important parameters
m_ = [50, 100, 170];    % Dimension of observations
n = 600;                % Number of observations, n > m
frac_ = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55];      
                        % Outlier fraction

X_min = -1;             % Min bound of X
X_max = 1;              % Max bound of X
theta_mean = 0;         % Mean of the normal distribution for theta
theta_sigma = 5;        % Standard deviation of the normal distribution for theta
eps_0 = 20;              % Inlier noise bound
outErr = 25;            % Outlier noise abs value

%% Exp 1: MSE of Least Square, M-est and GARD
iter = 100;             % Number of iterations to repeat per setting
for i=1:numel(m_)
    m = m_(i);
    fprintf('Start testing m=%d\n', m);
    
    % Prepare the arrays to save results
    MSE_LS = zeros(1, numel(frac_));
    MSE_Mest = zeros(1, numel(frac_));
    MSE_GARD = zeros(1, numel(frac_));
    for j = 1:numel(frac_)
        frac = frac_(j);
        s = floor(n * frac);    % Number of outlier indexes
        fprintf('Start testing frac=%f\n', frac);
        
        MSE_LS_m = 0;
        MSE_Mest_m = 0;
        MSE_GARD_m = 0;
        for it = 1:iter
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
            for t=1:s
                sign = -1 + 2 * (rand() > 0.5);
                u_0(rdn_idx(t)) = outErr * sign;
            end
            % Generate final vector y
            y = X * theta_0 + u_0 + eta;

            % Least square
            P = inv(X'*X) * X'; % Projection matrix
            theta_LS = P * y;   % x* in least square
            MSE_LS_m = MSE_LS_m + MSE(theta_0, theta_LS);
            % fprintf('MSE of LS: %f dB\n', MSE(theta_0, theta_LS));

            % M-estimator
            theta_Mest = robustfit(X, y);
            theta_Mest = theta_Mest(2:end); % Omit the const
            MSE_Mest_m = MSE_Mest_m + MSE(theta_0, theta_Mest);
            % fprintf('MSE of M-estimator: %f dB\n', MSE(theta_0, theta_Mest));

            % GARD with QR factorization
            theta_GARD_QR = GARD_QR(X, y, n, m, eps_0);
            MSE_GARD_m = MSE_GARD_m + MSE(theta_0, theta_GARD_QR);
            % fprintf('MSE of GARD: %f dB\n', MSE(theta_0, theta_GARD_QR));
        end
        
        % Convert the average MSE to log scale and save to array
        MSE_LS(j) = 10 * log10(MSE_LS_m / iter);
        MSE_Mest(j) = 10 * log10(MSE_Mest_m / iter);
        MSE_GARD(j) = 10 * log10(MSE_GARD_m / iter);
    end
    
    % Plot
    figure;
    plot(frac_, MSE_LS, 'ro', 'DisplayName','Least Square'); hold on;
    plot(frac_, MSE_Mest, 'g*', 'DisplayName','M-estimator'); hold on;
    plot(frac_, MSE_GARD, 'bd', 'DisplayName','GARD');
    legend;
    xlabel('Outlier fraction %'); ylabel('10log10(MSE)');
    title(sprintf('Total MSE comparison under m=%d', m));
end

% GARD
%tic;
%theta_GARD = GARD(X, y, n, m, eps_0);
%time = toc;
%fprintf('MSE of GARD: %f dB, time: %f s\n', MSE(theta_0, theta_GARD), time);

% GARD with QR factorization
%tic;
%theta_GARD_QR = GARD_QR(X, y, n, m, eps_0);
%time = toc;
%fprintf('MSE of GARD QR: %f dB, time: %f s\n', MSE(theta_0, theta_GARD_QR), time);

% GARD with Matrix Inversion Lemma
%tic;
%theta_GARD_MI = GARD_MI(X, y, n, m, eps_0);
%time = toc;
%fprintf('MSE of GARD MI: %f dB, time: %f s\n', MSE(theta_0, theta_GARD_MI), time);

function err = MSE(v1, v2)
% Calculate the mean square error between v1 and v2
err = (v1 - v2)' * (v1 - v2);
err = sum(err) / size(v1, 1);
end
