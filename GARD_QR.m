function theta_GARD_QR = GARD_QR(X, y, n, m, eps_0)
% GARD algorithm
% Compute initial residual by projecting y onto R(X)
k = 0;
Aac = X;
In = eye(n);
[Qk, Rk] = QR(Aac);               % Initial QR factorization
z_opt = backSubstitution(Rk, Qk' * y, m); % Use back substituion to solve
                                          % initial least square
rk = y - Aac * z_opt;             % Initial residual
% Start the loop until the residual is small enough
jk_list = zeros(1, n);            % Record jk in each round
norm_rk_list = zeros(1, n);       % Record norm(rk) in each round
while norm(rk) > eps_0
    k = k + 1;
    [val, jk] = max(abs(rk));
    jk_list(k) = jk;
    uk = In(:, jk) - Qk * Qk' * In(:, jk);
    ek = uk / norm(uk);
    Rk = [Rk, Qk' * In(:, jk); ...
        zeros(1, m+k-1), norm(uk)];
    Qk = [Qk, ek];
    z_opt = backSubstitution(Rk, Qk' * y, m+k);
    Aac = [Aac, In(:, jk)];
    rk = y - Aac * z_opt;
    norm_rk_list(k) = norm(rk);
end
theta_GARD_QR = z_opt(1:m);
end

function [Q, R] = QR(A)
% Use Gram-Schmidt to decompose A = QR
% where A: nxm, Q:nxm, R:mxm, n > m
n = size(A, 1);
m = size(A, 2);
Q = zeros(n, m);
R = zeros(m, m);
for i=1:m
    R(:, i) = Q' * A(:, i);
    u = A(:, i) - Q * R(:, i); % Substitute Q'*A(:, i) with R(:, i) will 
                               % speed up a lot
    R(i, i) = norm(u);
    Q(:, i) = u / norm(u);
end
end

function x=backSubstitution(U,b,n)
% Solving an upper triangular system by back-substitution
% Input matrix U is an n by n upper triangular matrix
% Input vector b is n by 1
% Input scalar n specifies the dimensions of the arrays
% Output vector x is the solution to the linear system
% U x = b
% K. Ming Leung, 01/26/03

x=zeros(n,1);
for j=n:-1:1
    if (U(j,j)==0)
        error('Matrix is singular!');
    end
    x(j)=b(j)/U(j,j);
    b(1:j-1)=b(1:j-1)-U(1:j-1,j)*x(j);
end
end
