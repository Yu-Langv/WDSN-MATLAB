function [x, result] = admm_zfirst_monitor(A, b, pm)
%ADMM_ZFIRST_MONITOR Copy of mL12mL22_regularized_ADMM_zfirst (revision test)
% that also records the scaled augmented Lagrangian
%   L_rho = f(x) + lambda*R_eta(z) + rho*<u, x-z> + rho/2*||x-z||^2
% and stops early (result.violated = true) once L_rho has increased in
% pm.viol_window consecutive iterations (pm.viol_window = Inf disables it).
%
% Original description:
%
% Solves the regularized WDSN model
%
%   min_x  0.5*||A*x-b||_2^2
%          + lambda*(||x||_1^2 - eta*||x||_2^2)
%
% using the same splitting and exact WDSN proximal operator as
% mL12mL22_regularized_ADMM, but swaps the Gauss-Seidel block order:
%
%   z-update: z in Prox_{(lambda/rho) R_eta}(x+u)
%   x-update: (A'*A + rho*I)x = A'*b + rho*(z-u)
%   u-update: u = u + x - z
%
% Input/output fields are aligned with mL12mL22_regularized_ADMM(A,b,pm).

if nargin < 3 || isempty(pm)
    pm = struct();
end

result = pm;
[~, N] = size(A);
b = b(:);

reject_deprecated_options(pm);

eta = require_scalar_option(pm, 'eta');
lambda = require_scalar_option(pm, 'lambda');
rho = require_scalar_option(pm, 'rho');
max_iter = scalar_option(pm, 'max_iter', 800);
rel_tol = scalar_option(pm, 'rel_tol', 1e-6);
abs_tol = scalar_option(pm, 'abs_tol', 1e-6);
restol = scalar_option(pm, 'restol', 1e-3);
verbose = scalar_option(pm, 'verbose', 0);
stop_test = [];
if isfield(pm, 'stop_test'), stop_test = pm.stop_test; end
viol_window = Inf;
if isfield(pm, 'viol_window') && ~isempty(pm.viol_window), viol_window = pm.viol_window; end

if eta < 0
    error('pm.eta must be nonnegative.');
end
if lambda <= 0
    error('The regularized WDSN model requires lambda > 0.');
end
if rho <= 0
    error('ADMM penalty rho must be positive.');
end
% Algorithm alg:admm_regularized: every proximal step is solvable iff
% rho + 2*lambda*(1-eta) > 0 (eq:prox-solvability with t = lambda/rho);
% automatic for 0 <= eta <= 1.
if ~(rho + 2 * lambda * (1 - eta) > 0)
    error('ADMM requires rho + 2*lambda*(1-eta) > 0.');
end
if max_iter < 1 || max_iter ~= floor(max_iter)
    error('pm.max_iter must be a positive integer.');
end
if rel_tol <= 0 || abs_tol <= 0
    error('pm.rel_tol and pm.abs_tol must be positive.');
end

[z, init_solver, init_message] = initial_point(A, b, pm, N);
u = get_vector_option(pm, 'u0', zeros(N, 1));
x_quad = z;

Atb = A' * b;
solve_x = build_x_solver(A, rho);

obj = @(xx) 0.5 * norm(A * xx - b, 2)^2 + ...
    lambda * (norm(xx, 1)^2 - eta * norm(xx, 2)^2);
reta = @(xx) norm(xx, 1)^2 - eta * norm(xx, 2)^2;

obj_list = zeros(max_iter, 1);
obj_z_list = zeros(max_iter, 1);
r_list = zeros(max_iter, 1);
s_list = zeros(max_iter, 1);
eps_pri_list = zeros(max_iter, 1);
eps_dual_list = zeros(max_iter, 1);
res_list = zeros(max_iter, 1);
time_list = zeros(max_iter, 1);
err_list = nan(max_iter, 1);
lag_list = zeros(max_iter, 1);
n_up = 0;
n_up_total = 0;
violated = false;
tau = lambda / rho;
converged = false;
start_time = tic;

has_xg = isfield(pm, 'xg') && ~isempty(pm.xg);
if has_xg
    xg = pm.xg(:);
else
    xg = nan(N, 1);
end

for k = 1:max_iter
    x_old = x_quad;

    z = prox_L1sq_minus_eta_L2sq(x_quad + u, tau, eta);
    h = Atb + rho * (z - u);
    x_quad = solve_x(h);
    u = u + x_quad - z;

    r = x_quad - z;
    s = rho * (x_quad - x_old);
    r_norm = norm(r, 2);
    s_norm = norm(s, 2);
    eps_pri = sqrt(N) * abs_tol + rel_tol * max(norm(x_quad, 2), norm(z, 2));
    eps_dual = sqrt(N) * abs_tol + rel_tol * norm(rho * u, 2);

    r_list(k) = r_norm;
    s_list(k) = s_norm;
    eps_pri_list(k) = eps_pri;
    eps_dual_list(k) = eps_dual;
    obj_list(k) = obj(x_quad);
    obj_z_list(k) = obj(z);
    res_list(k) = norm(A * z - b, 2) / max(norm(b, 2), eps);
    stop = k > 1 && r_norm <= eps_pri && s_norm <= eps_dual;
    if ~isempty(stop_test), stop = stop_test(z); end
    time_list(k) = toc(start_time);
    lag_list(k) = obj_list(k) - lambda * reta(x_quad) + lambda * reta(z) ...
        + rho * (u' * r) + 0.5 * rho * r_norm^2;
    if k > 1 && lag_list(k) > lag_list(k - 1)
        n_up = n_up + 1;
        n_up_total = n_up_total + 1;
    else
        n_up = 0;
    end

    if has_xg
        err_list(k) = norm(z - xg, 2) / max(norm(xg, 2), eps);
    end

    if verbose && (k == 1 || mod(k, 50) == 0)
        fprintf('RWDSN-ADMM-zfirst iter %4d: r=%.3e, s=%.3e, obj=%.6e\n', ...
            k, r_norm, s_norm, obj_z_list(k));
    end

    if stop
        converged = true;
        break;
    end
    if n_up >= viol_window
        violated = true;
        break;
    end
end

% Return z for consistency with the original ADMM solver: it is the sparse
% prox block and agrees with x_quad at convergence.
x = z;

result.i = k;
result.j = 0;
result.iter = k;
result.converged = converged;
result.violated = violated;
result.n_up_total = n_up_total;
result.lag_list = lag_list(1:k);
result.lambda = lambda;
result.eta = eta;
result.rho = rho;
result.tau = tau;
result.max_iter = max_iter;
result.rel_tol = rel_tol;
result.abs_tol = abs_tol;
result.init_solver = init_solver;
result.init_message = init_message;
result.method = 'RWDSN-ADMM-zfirst';
result.dual_residual_definition = 'rho*(x_k-x_{k-1}) because x is the second block';
result.x_quadratic = x_quad;
result.z = z;
result.u = u;
result.obj_list = obj_list(1:k);
result.obj_z_list = obj_z_list(1:k);
result.r_norm = r_list(1:k);
result.s_norm = s_list(1:k);
result.eps_pri = eps_pri_list(1:k);
result.eps_dual = eps_dual_list(1:k);
result.res_list = res_list(1:k);
result.time_list = time_list(1:k);
result.error_list = err_list(1:k);
result.reta = reta(x);
result.fx = obj(x);
result.residual = norm(A * x - b, 2);
result.relative_residual = result.residual / max(norm(b, 2), eps);
result.nnz = nnz(abs(x) > 1e-8);
result.norm1 = norm(x, 1);
result.norm2 = norm(x, 2);

if has_xg
    fxg = obj(xg);
    result.fxg = fxg;
    result.reta_xg = reta(xg);
    result.error = norm(x - xg, 2) / max(norm(xg, 2), eps);
    result.xnorm2 = norm(x, 2) - norm(xg, 2);
    result.xnorm2_r = result.xnorm2 / max(norm(xg, 2), eps);

    if result.error < restol
        result.rate = 1;
    elseif result.fx + 1e-12 < fxg
        result.rate = -1;
    else
        result.rate = -2;
    end
else
    result.fxg = NaN;
    result.reta_xg = NaN;
    result.error = NaN;
    result.xnorm2 = NaN;
    result.xnorm2_r = NaN;
    result.rate = NaN;
end

end

function reject_deprecated_options(pm)
deprecated_names = {'lambda_wdsn', 'alpha_wdsn', 'rho_wdsn', ...
    'maxit', 'iterADMM', 'reltol', 'abstol', ...
    'eps_abs', 'eps_pri', 'reletive_error'};
bad = {};
for ii = 1:numel(deprecated_names)
    if isfield(pm, deprecated_names{ii})
        bad{end + 1} = deprecated_names{ii}; %#ok<AGROW>
    end
end
if ~isempty(bad)
    error(['Deprecated pm field(s): %s. Use only canonical fields ', ...
        'pm.lambda, pm.rho, pm.max_iter, pm.rel_tol, and pm.abs_tol.'], ...
        strjoin(bad, ', '));
end
end

function value = require_scalar_option(pm, name)
if ~isfield(pm, name) || isempty(pm.(name))
    error('Missing required option pm.%s.', name);
end
value = pm.(name);
if ~isscalar(value) || ~isnumeric(value) || ~isfinite(value)
    error('pm.%s must be a finite numeric scalar.', name);
end
end

function value = scalar_option(pm, name, default_value)
if isfield(pm, name) && ~isempty(pm.(name))
    value = pm.(name);
else
    value = default_value;
end
if ~isscalar(value) || ~isnumeric(value) || ~isfinite(value)
    error('pm.%s must be a finite numeric scalar.', name);
end
end

function value = get_vector_option(pm, name, default_value)
if isfield(pm, name) && ~isempty(pm.(name))
    value = pm.(name)(:);
else
    value = default_value;
end
end

function [z, solver_name, message] = initial_point(A, b, pm, N)
message = '';
if isfield(pm, 'z0') && ~isempty(pm.z0)
    z = pm.z0(:);
    solver_name = 'pm.z0';
elseif isfield(pm, 'x0') && ~isempty(pm.x0)
    z = pm.x0(:);
    solver_name = 'pm.x0';
elseif isfield(pm, 'xr') && ~isempty(pm.xr)
    z = pm.xr(:);
    solver_name = 'pm.xr';
else
    try
        z = mL1_constrained_LP_Gurobi(A, b);
        z = z(:);
        solver_name = 'mL1_constrained_LP_Gurobi';
    catch ME
        z = minimum_norm_feasible(A, b, N);
        solver_name = 'minimum_norm_feasible';
        message = sprintf('mL1_constrained_LP_Gurobi unavailable or failed: %s', ME.message);
        warning('%s', message);
    end
end

if numel(z) ~= N
    error('Initial point length is %d, expected %d.', numel(z), N);
end
end

function x = minimum_norm_feasible(A, b, N)
AAt = A * A';
if rcond(full(AAt)) > 1e-12
    x = A' * (AAt \ b);
else
    x = A' * (pinv(full(AAt)) * b);
end
if isempty(x)
    x = zeros(N, 1);
end
end

function solve_x = build_x_solver(A, rho)
[M, N] = size(A);
if M < N
    C = eye(M) + (A * A') / rho;
    [L, p] = chol(C, 'lower');
    if p > 0
        C = C + 1e-12 * eye(M);
        L = chol(C, 'lower');
    end
    U = L';
    solve_x = @(h) h / rho - A' * (U \ (L \ (A * h))) / rho^2;
else
    H = A' * A + rho * eye(N);
    [L, p] = chol(H, 'lower');
    if p > 0
        H = H + 1e-12 * eye(N);
        L = chol(H, 'lower');
    end
    U = L';
    solve_x = @(h) U \ (L \ h);
end
end

function x = prox_L1sq_minus_eta_L2sq(v, tau, eta)
% One element of Prox_{tau*(||.||_1^2 - eta*||.||_2^2)}(v), computed as in
% Algorithm alg:prox_l1sq_l2sq_compact of the manuscript. Requires eta >= 0,
% tau > 0 and a_t = 1 + 2*tau*(1-eta) > 0 (eq:prox-solvability).
v = v(:);
n = length(v);
w = abs(v);
a_t = 1 + 2 * tau * (1 - eta);
if ~(a_t > 0)
    error('Empty WDSN proximal set: 1 + 2*tau*(1-eta) must be positive.');
end
if all(w == 0)
    x = zeros(n, 1);
    return;
end

c = 1 - 2 * tau * eta;
if c > 0
    % Theorem thm:prox-c-positive, written with the gaps d_i = ||w||_inf - w_i
    % so that it stays accurate when c is close to 0: with D_k = sum_{j<=k} d_(j),
    % k* = max{k : c*w_(k) > 2*tau*(k*d_(k) - D_k)} and
    % |x_i| = max{0, (c*w_i + 2*tau*(D_k* - k*d_i)) / (c*(c + 2*tau*k*))},
    % which equals max{0, (w_i - 2*tau*S*)/c} with S* = Q_k*/(c + 2*tau*k*).
    w_sort = sort(w, 'descend');
    d_sort = w_sort(1) - w_sort;
    D = cumsum(d_sort);
    kk = (1:n)';
    k_star = find(c * w_sort > 2 * tau * (kk .* d_sort - D), 1, 'last');
    if isempty(k_star), k_star = 1; end  % k = 1 always qualifies when c > 0
    d = w_sort(1) - w;
    x = sign(v) .* max(0, (c * w + 2 * tau * (D(k_star) - k_star * d)) ...
        / (c * (c + 2 * tau * k_star)));
else
    % c <= 0: smallest index of the exact maximal set M = {i : w_i = ||w||_inf}
    % (max returns the first maximizer); for c = 0, a_t = 2*tau.
    [w_inf, i_star] = max(w);
    x = zeros(n, 1);
    x(i_star) = sign(v(i_star)) * w_inf / a_t;
end
end
