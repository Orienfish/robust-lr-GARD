%% Exp C: Support Recovery Test
function expC(p, iter)
    m = p.m;

    % Prepare the arrays to save results
    P_Corr = zeros(1, numel(p.frac_));
    P_Ext = zeros(1, numel(p.frac_));
    MSE_GARD = zeros(1, numel(p.frac_));
    MSE_TH = zeros(1, numel(p.frac_));

    for j=1:numel(p.frac_)
        frac = p.frac_(j);
        s = floor(p.n * frac);    % Number of outlier indexes
        fprintf('Start testing frac=%f\n', frac);

        P_Corr_m = 0;
        P_Ext_m = 0;
        MSE_GARD_m = 0;
        MSE_TH_m = 0;

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
            % y_n = X * theta_0 + u_0;    % No inlier noise

            % Calculate theoretical bound for MSE
            delta_s = sqrt((min(abs(u_0(rdn_idx))) - (2+sqrt(6))*p.eps_0)/ (2 * norm(u_0)));
            [U, S, V] = svd(X);
            sv = diag(S);
            min_sv = min(sv(sv > 0));
            MSE_TH_m = MSE_TH_m + p.eps_0 / (min_sv * sqrt(1 - delta_s));
            % fprintf('%f\n', p.eps_0 / (min_sv * sqrt(1 - delta_s)));

            % GARD with QR acceleration
            [theta_GARD, jk_list] = GARD_QR(X, y, p.n, m, p.eps_0);
            [CorrIdx, ExtIdx] = EvalRecovery(p.n, rdn_idx, jk_list);
            P_Corr_m = P_Corr_m + CorrIdx;
            P_Ext_m = P_Ext_m + ExtIdx;
            MSE_GARD_m = MSE_GARD_m + MSE(theta_0, theta_GARD);
            % fprintf("%f %f \n", CorrIdx, ExtIdx);
        end

        P_Corr(j) = P_Corr_m / iter;
        P_Ext(j) = P_Ext_m / iter;
        MSE_GARD(j) = MSE_GARD_m / iter;
        MSE_TH(j) = MSE_TH_m / iter;
    end

    % Plot
    figure;
    subplot(2,1,1);
    plot(p.frac_, P_Corr, 'g^-', 'LineWidth', 2, 'DisplayName','Correct indices recovered');
    hold on;
    plot(p.frac_, P_Ext, 'yv-', 'LineWidth', 2, 'DisplayName','Extra indices recovered');
    legend('Location', 'West', 'FontSize', 12);
    xlabel('Outlier fraction %', 'FontSize', 12); xlim([0.0, 0.5]);
    ylabel('Support recovered %', 'FontSize', 12);
    ax = gca; ax.FontSize = 12;
    title('Recovery of the support for GARD', 'FontSize', 12);
    subplot(2,1,2);
    plot(p.frac_, MSE_GARD, 'bo-', 'LineWidth', 2, 'DisplayName','Empirical Bound');
    hold on;
    plot(p.frac_, MSE_TH, 'ks-', 'LineWidth', 2, 'DisplayName','Theoretical Bound');
    legend('Location', 'West', 'FontSize', 12);
    xlabel('Outlier fraction %', 'FontSize', 12); xlim([0.0, 0.3]);
    ylabel('MSE', 'FontSize', 12);
    ax = gca; ax.FontSize = 12;
    title('Reconstruction error', 'FontSize', 12);
end

function [CorrIdx, ExtIdx] = EvalRecovery(n, real_list, recover_list)
real_item = zeros(1, n);
recover_item = zeros(1, n);
real_item(real_list) = 1;
recover_item(recover_list) = 1;
CorrIdx = sum(real_item & recover_item) / numel(real_list) * 100;
ExtIdx = sum((recover_item - real_item) > 0) / numel(real_list) * 100;
end

function err = MSE(v1, v2)
% Calculate the mean square error between v1 and v2
err = (v1 - v2)' * (v1 - v2);
err = sum(err) / size(v1, 1);
end