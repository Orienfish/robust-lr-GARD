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
p.eps_0 = 20;             % Inlier noise bound
p.outErr = 25;            % Outlier noise abs value

%% Whether to run each experiments
%iter = 100;               % Number of iterations to repeat per setting
%exp1(p, iter);

