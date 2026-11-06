function [x, result] = mL12mL22_DCA_NNLS(A, b, pm)
%ML12ML22_DCA_NNLS Proximal DCA for the regularized WDSN model, one NNLS per step.
%
% Solves (to a critical point)
%
%   min_x  0.5*||A*x-b||_2^2 + lambda*(||x||_1^2 - eta*||x||_2^2)
%
% With x = u - v and z = [u; v] >= 0, the convex part
% 0.5*||Ax-b||^2 + lambda*||x||_1^2 is one NNLS residual (Foucart-Koslicki).
% The concave part -lambda*eta*||x||^2 is linearized at x^k and the proximal
% term (c/2)*||x-x^k||^2 is added, so every step is the standard NNLS
%
%   min_{z>=0} || [A, -A; s*1', s*1'; sqrt(c)*I, -sqrt(c)*I] z
%                 - [b; 0; sqrt(c)*t^k] ||_2^2,
%   s = sqrt(2*lambda),  t^k = (1 + 2*lambda*eta/c) * x^k,
%
% and F(x^{k+1}) <= F(x^k) - (c/2)*||x^{k+1}-x^k||^2.
%
% Options: pm.lambda, pm.eta (required); pm.c_factor (1, c = c_factor*2*lambda),
% pm.max_iter (2000), pm.tol (1e-9, relative change of x), pm.nnls_solver
% ('lh_warm' = warm-started Lawson-Hanson on the implicit matrix, or
% 'lsqnonneg' = MATLAB lsqnonneg on the explicit matrix, small sizes only,
% 'gurobi' = Gurobi on an equivalent sparse QP; pm.gurobi_method picks the
% Gurobi algorithm, default -1 = automatic),
% pm.z0 / pm.x0 / pm.xr warm start, pm.xg and pm.restol for the success test.
% Output fields follow mL12mL22_regularized_ADMM_zfirst.

if nargin < 3 || isempty(pm)
    pm = struct();
end

result = pm;
[M, N] = size(A);
b = b(:);

lambda = pm.lambda;
eta = pm.eta;
c_factor = get_opt(pm, 'c_factor', 1);
max_iter = get_opt(pm, 'max_iter', 2000);
tol = get_opt(pm, 'tol', 1e-9);
restol = get_opt(pm, 'restol', 1e-3);
nnls_solver = get_opt(pm, 'nnls_solver', 'lh_warm');
verbose = get_opt(pm, 'verbose', 0);
stop_test = get_opt(pm, 'stop_test', []); % optional common benchmark criterion

if lambda <= 0, error('pm.lambda must be positive.'); end
% Algorithm alg:nnls-dca: the DCA step is valid for every eta >= 0. For
% eta > 1 the model is bounded below only under eq:coercivity-condition;
% blow-up of the iterates is reported below. (pm.allow_eta_gt1 is ignored.)
if eta < 0
    error('pm.eta must be nonnegative.');
end

s = sqrt(2 * lambda);
c = c_factor * 2 * lambda;

x = initial_point(pm, N);
z = [max(x, 0); max(-x, 0)];
Atb = A' * b;
nnls_tol = 1e-13 * max(1, norm(Atb, inf));

if strcmpi(nnls_solver, 'lsqnonneg')
    C = [A, -A; s * ones(1, 2 * N); sqrt(c) * [speye(N), -speye(N)]];
    C = full(C);
    opts = optimset('TolX', nnls_tol);
elseif strcmpi(nnls_solver, 'gurobi')
    % same NNLS as a separable convex QP in w = [u; v; x; r; q]:
    %   min ||r||^2 + s^2*q^2 + c*||x||^2 - 2c*t'*x
    %   s.t. x - u + v = 0,  A*x - r = b,  q - 1'u - 1'v = 0,  u, v >= 0,
    % which equals ||C*z - d||^2 up to a constant; only the linear term
    % changes between DCA steps.
    I = speye(N);
    grb.A = [-I, I, I, sparse(N, M + 1); ...
        sparse(M, 2 * N), sparse(A), -speye(M), sparse(M, 1); ...
        -ones(1, 2 * N), sparse(1, N + M), 1];
    grb.rhs = [zeros(N, 1); b; 0];
    grb.sense = '=';
    grb.lb = [zeros(2 * N, 1); -inf(N + M + 1, 1)];
    grb.Q = spdiags([zeros(2 * N, 1); c * ones(N, 1); ones(M, 1); s^2], 0, 3 * N + M + 1, 3 * N + M + 1);
    % barrier returns an interior point (no crossover for QP): every u_i, v_i
    % is a tiny positive number off the support; simplex needs the tightest
    % tolerances, at the defaults its NNLS objective is off by about 1e-4
    grb_params = struct('OutputFlag', 0, 'BarConvTol', 1e-12, ...
        'OptimalityTol', 1e-9, 'FeasibilityTol', 1e-9, ...
        'Method', get_opt(pm, 'gurobi_method', -1), 'Threads', maxNumCompThreads);
end

obj = @(xx) 0.5 * norm(A * xx - b, 2)^2 + ...
    lambda * (norm(xx, 1)^2 - eta * norm(xx, 2)^2);

obj_list = zeros(max_iter, 1);
time_list = zeros(max_iter, 1);
lh_list = zeros(max_iter, 1);
has_xg = isfield(pm, 'xg') && ~isempty(pm.xg);
if has_xg
    xg = pm.xg(:);
    err_list = zeros(max_iter, 1);
else
    xg = nan(N, 1);
    err_list = nan(max_iter, 1);
end

converged = false;
start_time = tic;
for k = 1:max_iter
    x_old = x;
    t = (1 + 2 * lambda * eta / c) * x_old;
    if strcmpi(nnls_solver, 'lsqnonneg')
        z = lsqnonneg(C, [b; 0; sqrt(c) * t], opts);
        n_lh = NaN;
    elseif strcmpi(nnls_solver, 'gurobi')
        grb.obj = [zeros(2 * N, 1); -2 * c * t; zeros(M + 1, 1)];
        out = gurobi(grb, grb_params);
        if ~strcmp(out.status, 'OPTIMAL')
            error('Gurobi NNLS step: status %s.', out.status);
        end
        z = max(out.x(1:2 * N), 0);
        n_lh = out.itercount;
    else
        [z, n_lh] = nnls_lh_warm(A, b, Atb, s, c, t, z, nnls_tol);
    end
    x = z(1:N) - z(N + 1:end);
    if any(~isfinite(x))
        % divergence: keep the last finite iterate and stop
        x = x_old;
        obj_list(k) = obj(x);
        time_list(k) = toc(start_time);
        lh_list(k) = n_lh;
        if has_xg, err_list(k) = norm(x - xg) / max(norm(xg), eps); end
        break;
    end

    obj_list(k) = obj(x);
    stop = norm(x - x_old) <= tol * max(1, norm(x_old));
    if ~isempty(stop_test), stop = stop_test(x); end
    time_list(k) = toc(start_time);
    lh_list(k) = n_lh;
    if has_xg
        err_list(k) = norm(x - xg) / max(norm(xg), eps);
    end
    if verbose && (k == 1 || mod(k, 50) == 0)
        fprintf('DCA-NNLS iter %4d: F=%.10e, LH=%d\n', k, obj_list(k), n_lh);
    end
    if stop
        converged = true;
        break;
    end
end

result.iter = k;
result.converged = converged;
result.lambda = lambda;
result.eta = eta;
result.c = c;
result.method = 'RWDSN-DCA-NNLS';
result.nnls_solver = nnls_solver;
result.obj_list = obj_list(1:k);
result.time_list = time_list(1:k);
result.lh_iters = lh_list(1:k);
result.error_list = err_list(1:k);
result.fx = obj(x);
result.residual = norm(A * x - b, 2);
result.relative_residual = result.residual / max(norm(b, 2), eps);
result.nnz = nnz(abs(x) > 1e-8);
result.z = z;
result.M = M;

if has_xg
    fxg = obj(xg);
    result.fxg = fxg;
    result.error = norm(x - xg) / max(norm(xg), eps);
    if result.error < restol
        result.rate = 1;
    elseif result.fx + 1e-12 < fxg
        result.rate = -1;
    else
        result.rate = -2;
    end
else
    result.fxg = NaN;
    result.error = NaN;
    result.rate = NaN;
end
end

function [z, n_iter] = nnls_lh_warm(A, b, Atb, s, c, t, z, tol)
% Lawson-Hanson active set for min_{z>=0} ||C*z - d||_2 with the implicit
%   C = [A, -A; s*1', s*1'; sqrt(c)*I, -sqrt(c)*I],  d = [b; 0; sqrt(c)*t],
% started from the feasible point z (warm start: passive set = supp(z)).
% Indices 1..N are u_i, N+1..2N are v_i.
N = size(A, 2);
z = max(z, 0);
P = find(z > 0);
n_iter = 0;
if ~isempty(P)
    [z, P] = inner_loop(z, P);
end
last_j = 0;
while n_iter < 10 * N
    n_iter = n_iter + 1;
    w = gradient_w(z);
    w(P) = -inf;
    [wmax, j] = max(w);
    if wmax <= tol
        break;
    end
    if j == last_j
        % added index was dropped at once (degenerate step): stop
        break;
    end
    P = [P; j]; %#ok<AGROW>
    z_before = z;
    [z, P] = inner_loop(z, P);
    if any(P == j) || norm(z - z_before, inf) > 0
        last_j = 0;
    else
        last_j = j;
    end
end

    function w = gradient_w(zz)
        xx = zz(1:N) - zz(N + 1:end);
        r1 = b - A * xx;
        r0 = -s * sum(zz);
        g = A' * r1 + c * (t - xx);
        w = [g + s * r0; -g + s * r0];
    end

    function [zz, PP] = inner_loop(zz, PP)
        while true
            y = ls_on_set(PP);
            if any(~isfinite(y))
                % iterates have blown up (model unbounded below, eta > 1)
                zz(:) = NaN;
                return;
            end
            if all(y > 0)
                zz(PP) = y;
                return;
            end
            neg = y <= 0;
            zP = zz(PP);
            neg_idx = find(neg);
            [alpha, ii] = min(zP(neg) ./ (zP(neg) - y(neg)));
            zP = zP + alpha * (y - zP);
            zP(neg_idx(ii)) = 0;
            drop = zP <= 1e-15 * max(1, max(zP));
            zP(drop) = 0;
            zz(PP) = zP;
            PP = PP(~drop);
            if isempty(PP)
                return;
            end
        end
    end

    function y = ls_on_set(PP)
        I = mod(PP - 1, N) + 1;
        sg = 1 - 2 * (PP > N);
        AI = A(:, I) .* sg';
        G = AI' * AI + s^2 + c * (sg * sg') .* (I == I');
        rhs = sg .* (Atb(I) + c * t(I));
        [R, p] = chol(G);
        if p == 0
            y = R \ (R' \ rhs);
        else
            y = (G + 1e-14 * trace(G) / numel(PP) * eye(numel(PP))) \ rhs;
        end
    end
end

function x = initial_point(pm, N)
if isfield(pm, 'z0') && ~isempty(pm.z0)
    x = pm.z0(:);
elseif isfield(pm, 'x0') && ~isempty(pm.x0)
    x = pm.x0(:);
elseif isfield(pm, 'xr') && ~isempty(pm.xr)
    x = pm.xr(:);
else
    x = zeros(N, 1);
end
end

function v = get_opt(pm, name, default_value)
if isfield(pm, name) && ~isempty(pm.(name))
    v = pm.(name);
else
    v = default_value;
end
end
