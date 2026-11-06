function test_wdsn()
%TEST_WDSN Check proximal branches, NNLS optimality and solver residuals.
rng(7, 'twister');
prox = @exact_prox_L1sq_L2sq_wdsn;
assert(isequal(prox(zeros(3, 1), 0.1, 1), zeros(3, 1)));
assert(norm(prox([0.5; -2; 2; 1], 0.25, 2) - [0; -4; 0; 0]) < 1e-12);
assert(norm(prox([-(1-1e-11); 1; 0.3], 0.4, 2) - [0; 5; 0]) < 1e-12);
for trial = 1:100
    v = randn(12, 1); tau = 0.01 + 0.2*rand; eta = rand;
    x = prox(v, tau, eta);
    mu = 2*tau*norm(x, 1);
    g = (1-2*tau*eta)*x - v;
    S = abs(x) > 1e-10;
    violation = max([0; abs(g(S)+mu*sign(x(S))); max(abs(g(~S))-mu, 0)]);
    assert(violation < 1e-10, 'Proximal KKT check failed.');
end
rejected = false;
try
    prox([1; 2], 1, 2);
catch
    rejected = true;
end
assert(rejected, 'The proximal solvability guard did not reject invalid parameters.');

A = randn(15, 30); A = A ./ vecnorm(A); b = randn(15, 1);
x0 = randn(30, 1); lambda = 0.1; eta = 0.8; c = 2*lambda;
pm = struct('lambda', lambda, 'eta', eta, 'x0', x0, 'max_iter', 1);
x = mL12mL22_DCA_NNLS(A, b, pm);
g = A'*(A*x-b) - 2*lambda*eta*x0 + c*(x-x0);
mu = 2*lambda*norm(x, 1); S = abs(x) > 1e-10;
violation = max([0; abs(g(S)+mu*sign(x(S))); max(abs(g(~S))-mu, 0)]);
assert(violation < 1e-8, 'NNLS-DCA subproblem KKT check failed.');

r = demo_wdsn();
assert(all(isfinite([r.x_admm; r.x_nnls])));
assert(r.admm.converged && r.nnls.converged, 'A solver reached its iteration limit.');
assert(r.admm.r_norm(end) <= r.admm.eps_pri(end));
assert(r.admm.s_norm(end) <= r.admm.eps_dual(end));
assert(max([0; diff(r.nnls.obj_list)]) < 1e-10, 'NNLS-DCA objective increased.');
assert(r.admm.k0 > 0 && r.admm.rho_bar > sqrt(2)*r.admm.Lf);
fprintf('All WDSN checks passed.\n');
end
