%% Exp D: Phase Transition Curves
function expD(p, iter)
    m = p.m;

    % Prepare the arrays to save results
    P_LS = zeros(1, numel(p.frac_));
    P_Mest = zeros(1, numel(p.frac_));
    P_GARD = zeros(1, numel(p.frac_));

    for j=1:numel(p.frac_)
        frac = p.frac_(j);
        s = floor(p.n * frac);    % Number of outlier indexes
        fprintf('Start testing frac=%f\n', frac);

        P_LS_m = 0;
        P_Mest_m = 0;
        P_GARD_m = 0;

        %% No inlier noise
        for it=1:iter
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
            P = inv(X'*X) * X'; % Projection matrix
            theta_LS = P * y;   % x* in least square
            P_LS_m = P_LS_m + (norm(theta_0 - theta_LS) <= norm(theta_0) * p.thres);

            % M-estimator
            theta_Mest = robustfit(X, y);
            theta_Mest = theta_Mest(2:end); % Omit the const
            P_Mest_m = P_Mest_m + (norm(theta_0 - theta_Mest) <= norm(theta_0) * p.thres);

            % GARD
            theta_GARD = GARD(X, y, p.n, m, p.eps_0);
            P_GARD_m = P_GARD_m + (norm(theta_0 - theta_GARD) <= norm(theta_0) * p.thres);
        end
    P_LS(j) = P_LS_m / iter;
    P_Mest(j) = P_Mest_m / iter;
    P_GARD(j) = P_GARD_m / iter;
    end

    % Plot robability
    figure;
    plot(p.frac_, P_LS, 'ro-', 'LineWidth', 2, 'DisplayName','Least Square'); 
    hold on;
    plot(p.frac_, P_Mest, 'g*-', 'LineWidth', 2, 'DisplayName','M-estimator');
    hold on;
    plot(p.frac_, P_GARD, 'bd-', 'LineWidth', 2, 'DisplayName','GARD');
    legend('Location', 'northeast', 'FontSize', 16);
    xlabel('Outlier fraction %', 'FontSize', 16); xlim([0.0, 0.6]);
    ylabel('Recovery Rate', 'FontSize', 16);
    ax = gca; ax.FontSize = 16;
    title(sprintf('Probability of recovery for dimension of unknown vector m=%d', m), ...
        'FontSize', 16);
end
