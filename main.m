clc;
clear;
close all;

%% Initialization
% Important parameters
p.m_ = [50, 100, 170];    % Dimension of observations
p.n = 600;                % Number of observations, n > m
p.frac_ = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55];      
                        % Outlier fraction
p.X_min = -1;             % Min bound of X
p.X_max = 1;              % Max bound of X
p.theta_mean = 0;         % Mean of the normal distribution for theta
p.theta_sigma = 5;        % Standard deviation of the normal distribution for theta
p.eps_0 = 20;              % Inlier noise bound
p.outErr = 25;            % Outlier noise abs value

%% Whether to run each experiments
exp1(p);

%% Exp 1: MSE and execution time of Least Square, M-est and GARD (3 versions)
function exp1(p)
    iter = 1;             % Number of iterations to repeat per setting
    for i=1:numel(p.m_)
        m = p.m_(i);
        fprintf('Start testing m=%d\n', m);

        % Prepare the arrays to save results
        MSE_LS = zeros(1, numel(p.frac_));
        MSE_Mest = zeros(1, numel(p.frac_));
        MSE_GARD = zeros(1, numel(p.frac_));
        T_LS = zeros(1, numel(p.frac_));
        T_Mest = zeros(1, numel(p.frac_));
        T_GARD = zeros(1, numel(p.frac_));
        T_GARD_QR = zeros(1, numel(p.frac_));
        T_GARD_MI = zeros(1, numel(p.frac_));
        for j = 1:numel(p.frac_)
            frac = p.frac_(j);
            s = floor(p.n * frac);    % Number of outlier indexes
            fprintf('Start testing frac=%f\n', frac);

            MSE_LS_m = 0;
            MSE_Mest_m = 0;
            MSE_GARD_m = 0;
            T_LS_m = 0;
            T_Mest_m = 0;
            T_GARD_m = 0;
            T_GARD_QR_m = 0;
            T_GARD_MI_m = 0;
            for it = 1:iter
                % Generate random observation X and linear weights theta
                X = repmat(p.X_min, p.n, m);
                X = X + rand(p.n, m) * (p.X_max - p.X_min);
                theta_0 = normrnd(p.theta_mean, p.theta_sigma, m, 1);
                % Generate bounded inlier noise
                eta = normrnd(0, 1, p.n, 1);
                eta = min(eta, p.eps_0);
                eta = max(eta, -p.eps_0);
                % Generate s-sparse outlier noise
                rdn_idx = randsample(p.n, s);
                u_0 = zeros(p.n, 1);
                for t=1:s
                    sign = -1 + 2 * (rand() > 0.5);
                    u_0(rdn_idx(t)) = p.outErr * sign;
                end
                % Generate final vector y
                y = X * theta_0 + u_0 + eta;

                % Least square
                tic;
                P = inv(X'*X) * X'; % Projection matrix
                theta_LS = P * y;   % x* in least square
                T_LS_m = T_LS_m + toc;
                MSE_LS_m = MSE_LS_m + MSE(theta_0, theta_LS);
                % fprintf('MSE of LS: %f dB\n', MSE(theta_0, theta_LS));

                % M-estimator
                tic;
                theta_Mest = robustfit(X, y);
                theta_Mest = theta_Mest(2:end); % Omit the const
                T_Mest_m = T_Mest_m + toc;
                MSE_Mest_m = MSE_Mest_m + MSE(theta_0, theta_Mest);
                % fprintf('MSE of M-estimator: %f dB\n', MSE(theta_0, theta_Mest));
                
                % GARD
                tic;
                theta_GARD = GARD(X, y, p.n, m, p.eps_0);
                T_GARD_m = T_GARD_m + toc;
                MSE_GARD_m = MSE_GARD_m + MSE(theta_0, theta_GARD);
                % fprintf('MSE of GARD: %f dB\n', MSE(theta_0, theta_GARD));

                % GARD with QR factorization
                tic;
                theta_GARD_QR = GARD_QR(X, y, p.n, m, p.eps_0);
                T_GARD_QR_m = T_GARD_QR_m + toc;
                % fprintf('MSE of GARD QR: %f dB\n', MSE(theta_0, theta_GARD_QR));
                
                 % GARD with Matrix Inversion Lemma
                tic;
                theta_GARD_MI = GARD_QR(X, y, p.n, m, p.eps_0);
                T_GARD_MI_m = T_GARD_MI_m + toc;
                % fprintf('MSE of GARD MI: %f dB\n', MSE(theta_0, theta_GARD_MI));
            end

            % Convert the average MSE to log scale and save to array
            MSE_LS(j) = 10 * log10(MSE_LS_m / iter);
            MSE_Mest(j) = 10 * log10(MSE_Mest_m / iter);
            MSE_GARD(j) = 10 * log10(MSE_GARD_m / iter);
            
            % Calculate the average runtime and save to array
            T_LS(j) = T_LS_m / iter;
            T_Mest(j) = T_Mest_m / iter;
            T_GARD(j) = T_GARD_m / iter;
            T_GARD_QR(j) = T_GARD_QR_m / iter;
            T_GARD_MI(j) = T_GARD_MI_m / iter; 
        end

        % Plot MSE
        figure;
        plot(p.frac_, MSE_LS, 'ro-', 'LineWidth', 2, 'DisplayName','Least Square'); 
        hold on;
        plot(p.frac_, MSE_Mest, 'g*-', 'LineWidth', 2, 'DisplayName','M-estimator');
        hold on;
        plot(p.frac_, MSE_GARD, 'bd-', 'LineWidth', 2, 'DisplayName','GARD');
        legend('Location', 'northwest', 'FontSize', 16);
        xlabel('Outlier fraction %', 'FontSize', 16); xlim([0.05, 0.55]);
        ylabel('10log10(MSE)', 'FontSize', 16);
        ax = gca; ax.FontSize = 16;
        title(sprintf('Total MSE comparison under m=%d', m), 'FontSize', 16);
        
        % Plot runtime
        figure;
        plot(p.frac_, T_LS, 'ro-', 'LineWidth', 2, 'DisplayName','Least Square'); 
        hold on;
        plot(p.frac_, T_Mest, 'g*-', 'LineWidth', 2, 'DisplayName','M-estimator');
        hold on;
        plot(p.frac_, T_GARD, 'bd-', 'LineWidth', 2, 'DisplayName','GARD');
        hold on;
        plot(p.frac_, T_GARD_QR, 'kx-', 'LineWidth', 2, 'DisplayName','GARD QR');
        hold on;
        plot(p.frac_, T_GARD_MI, 'ms-', 'LineWidth', 2, 'DisplayName','GARD MI');
        legend('Location', 'northwest', 'FontSize', 16);
        xlabel('Outlier fraction %', 'FontSize', 16); xlim([0.05, 0.55]);
        ylabel('Runtime (s)', 'FontSize', 16);
        ax = gca; ax.FontSize = 16;
        title(sprintf('Runtime comparison under m=%d', m), 'FontSize', 16);
    end
end


function err = MSE(v1, v2)
% Calculate the mean square error between v1 and v2
err = (v1 - v2)' * (v1 - v2);
err = sum(err) / size(v1, 1);
end
