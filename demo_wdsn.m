function results = demo_wdsn()
%DEMO_WDSN Run ADMM and NNLS-DCA on a reproducible synthetic example.
% Run this function from the repository directory. No data download is needed.
rng(42, 'twister');
m = 48; n = 128; k = 6;
A = randn(m, n);
A = A ./ vecnorm(A);
x_true = zeros(n, 1);
x_true(randperm(n, k)) = randn(k, 1);
b = A * x_true;
Lf = norm(A, 2)^2;
pm = struct('lambda', 1e-3, 'eta', 1, 'x0', zeros(n, 1), ...
    'xg', x_true, 'max_iter', 10000);
pa = pm;
pa.rho = 0.1 * sqrt(2) * Lf;
pa.q_theory = 1.1;
pa.K0_fraction = 0.8;
pa.viol_window = 50;
pa.rel_tol = 1e-7;
pa.abs_tol = 1e-8;
[xa, ra] = mL12mL22_ADMM_twophase(A, b, pa);
pn = pm;
pn.nnls_solver = 'lh_warm';
pn.c_factor = 1;
pn.tol = 1e-9;
[xn, rn] = mL12mL22_DCA_NNLS(A, b, pn);
fprintf('Synthetic Gaussian example: seed=42, m=%d, n=%d, k=%d\n', m, n, k);
fprintf('Method         Iterations  Objective      Relative error  Stopping test\n');
fprintf('ADMM           %6d      %.6e   %.3e       %d\n', ...
    ra.iter, ra.fx, ra.error, ra.converged);
fprintf('NNLS-DCA       %6d      %.6e   %.3e       %d\n', ...
    rn.iter, rn.fx, rn.error, rn.converged);
fprintf('ADMM stage switch: k0=%d, reason=%s\n', ra.k0, ra.switch_reason);
results = struct('A', A, 'b', b, 'x_true', x_true, ...
    'x_admm', xa, 'admm', ra, 'x_nnls', xn, 'nnls', rn);
end
