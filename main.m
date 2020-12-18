clc;
clear;
close all;

%% Initialization
% Important parameters
p.n = 600;                % Number of observations, n > m
p.X_min = -1;             % Min bound of X
p.X_max = 1;              % Max bound of X
p.theta_mean = 0;         % Mean of the normal distribution for theta
p.theta_sigma = 5;        % Standard deviation of the normal distribution for theta
p.eps_0 = 20;             % Inlier noise bound
p.outErr = 25;            % Outlier noise abs value
p.thres = 0.03;           % Threshold to evaluate a successful recovery

%% Run each experiments in the original paper
% Reproduce exp V.A
p.m_ = [50, 100, 170];    % Dimension of observation
p.frac_ = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55];      
                          % Outlier fraction
iter = 100;              % Number of iterations to repeat per setting
%expA(p, iter);

p.m = 100;                % Dimension of observation
p.frac_ = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
                          % Outlier fraction
p.eps_0 = 28;
p.outErr = 150;
iter = 1000;
%expC(p, iter);

% Reproduce exp V.D
p.m = 100;                % Dimension of observation
p.frac_ = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
                          % Outlier fraction
iter = 200;
%expD(p, iter);

%% Apply GARD on real-world datasets
fileID = fopen('dataset/x09.txt','r');
X = fscanf(fileID, '%f %f %f %f %f', [5 Inf]);
fclose(fileID);
X = X';
y = X(:, end);
X = X(:, 2:end-1); % Omit the first and last column
%X(:, 2:end) = normalize(X(:, 2:end));
%y = normalize(y);
n = size(X, 1);
m = size(X, 2);
eps_0_09 = 5:5:400;
MSE_arr_09 = zeros(1, numel(eps_0_09));
for i=1:numel(eps_0_09)
    eps_0 = eps_0_09(i);
    [theta_GARD, jk_list] = GARD(X, y, n, m, eps_0);
    y_ = X * theta_GARD;
    MSE_arr_09(i) = MSE(y, y_);
end

fileID = fopen('dataset/x26.txt','r');
X = fscanf(fileID, repmat('%f ', 1, 28), [13 Inf]);
fclose(fileID);
X = X';
y = X(:, end);
X = X(:, 2:end-1); % Omit the first and last column
n = size(X, 1);
m = size(X, 2);
eps_0_26 = 1:30;
MSE_arr_26 = zeros(1, numel(eps_0_26));
for i=1:numel(eps_0_26)
    eps_0 = eps_0_26(i);
    [theta_GARD, jk_list] = GARD(X, y, n, m, eps_0);
    y_ = X * theta_GARD;
    MSE_arr_26(i) = MSE(y, y_);
end
subplot(2,1,1);
plot(eps_0_09, MSE_arr_09, 'bo-', 'LineWidth', 2);
xlabel('eps_0', 'FontSize', 12);
ylabel('MSE of prediction', 'FontSize', 12);
ax = gca; ax.FontSize = 12;
title('Prediction accuracy of GARD on x09.txt', 'FontSize', 12);
subplot(2,1,2);
plot(eps_0_26, MSE_arr_26, 'ks-', 'LineWidth', 2);
xlabel('eps_0', 'FontSize', 12);
ylabel('MSE of prediction', 'FontSize', 12);
ax = gca; ax.FontSize = 12;
title('Prediction accuracy of GARD on x26.txt', 'FontSize', 12);


function err = MSE(v1, v2)
% Calculate the mean square error between v1 and v2
err = (v1 - v2)' * (v1 - v2);
err = sum(err) / size(v1, 1);
end
