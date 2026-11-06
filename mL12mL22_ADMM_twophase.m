function [x, result] = mL12mL22_ADMM_twophase(A, b, pm)
%ML12ML22_ADMM_TWOPHASE Two-stage penalty ADMM for the regularized WDSN model.
%
% Same interface as mL12mL22_regularized_ADMM_zfirst(A, b, pm).
%
% Stage 1 runs the z-first ADMM with the given penalty rho0 = pm.rho and
% monitors the scaled augmented Lagrangian L_rho. It switches at
%     k0 = min{k_stop, k_viol, K0},
%   k_stop  first iteration meeting the stopping test,
%   k_viol  first iteration after which L_rho increased in viol_window
%           consecutive iterations,
%   K0      floor(K0_fraction * max_iter).
% Stage 2 restarts from the stage-1 iterate with the fixed penalty
%     rho_bar = q_theory * sqrt(2) * L_f,  L_f = ||A'A||_2,
% so the convergence theorem (rho > sqrt(2) L_f) applies from k0 on. The
% scaled dual is rescaled as u <- u*rho0/rho_bar, which keeps rho*u.
%
% Optional fields: pm.q_theory (1.1), pm.K0_fraction (0.8),
% pm.viol_window (50). The returned result has the fields of the original
% solver for the final iterate, with iteration counts and histories of
% both stages concatenated, plus k0, switch_reason, n_up_total, rho0 and
% rho_bar.

q_theory = get_opt(pm, 'q_theory', 1.1);
K0_fraction = get_opt(pm, 'K0_fraction', 0.8);
viol_window = get_opt(pm, 'viol_window', 50);
max_iter = get_opt(pm, 'max_iter', 800);
pm = strip_fields(pm, {'q_theory', 'K0_fraction', 'viol_window'});

[M, N] = size(A);
if M < N
    Lf = max(eig(full(A * A')));
else
    Lf = max(eig(full(A' * A)));
end
rho0 = pm.rho;
rho_bar = q_theory * sqrt(2) * Lf;

if rho0 >= rho_bar
    % the given penalty already satisfies the theorem: single stage
    [x, result] = mL12mL22_regularized_ADMM_zfirst(A, b, pm);
    result.k0 = 0;
    result.switch_reason = 'none';
    result.n_up_total = NaN;
    result.rho0 = rho0;
    result.rho_bar = rho0;
    result.Lf = Lf;
    return;
end

% stage 1
p1 = pm;
p1.max_iter = max(1, floor(K0_fraction * max_iter));
p1.viol_window = viol_window;
[~, r1] = admm_zfirst_monitor(A, b, p1);
if r1.converged
    reason = 'stop';
elseif r1.violated
    reason = 'viol';
else
    reason = 'cap';
end

% stage 2
p2 = strip_fields(pm, {'x0', 'xr'});
p2.rho = rho_bar;
p2.max_iter = max(1, max_iter - r1.iter);
p2.z0 = r1.z;
p2.u0 = r1.u * rho0 / rho_bar;
[x, result] = mL12mL22_regularized_ADMM_zfirst(A, b, p2);

% merge histories and counts
k1 = r1.iter;
result.i = k1 + result.i;
result.iter = k1 + result.iter;
result.max_iter = max_iter;
list_names = {'obj_list', 'obj_z_list', 'r_norm', 's_norm', 'eps_pri', ...
    'eps_dual', 'res_list', 'error_list'};
for ii = 1:numel(list_names)
    nm = list_names{ii};
    if isfield(r1, nm) && isfield(result, nm)
        result.(nm) = [r1.(nm)(:); result.(nm)(:)];
    end
end
if isfield(r1, 'time_list') && isfield(result, 'time_list')
    t1 = r1.time_list(:);
    result.time_list = [t1; t1(end) + result.time_list(:)];
end
result.init_solver = r1.init_solver;
result.method = 'RWDSN-ADMM-zfirst-twophase';
result.k0 = k1;
result.switch_reason = reason;
result.n_up_total = r1.n_up_total;
result.stage1_converged = r1.converged;
result.rho0 = rho0;
result.rho_bar = rho_bar;
result.rho = rho_bar;
result.Lf = Lf;
end

function v = get_opt(pm, name, default_value)
if isfield(pm, name) && ~isempty(pm.(name))
    v = pm.(name);
else
    v = default_value;
end
end

function pm = strip_fields(pm, names)
for ii = 1:numel(names)
    if isfield(pm, names{ii}), pm = rmfield(pm, names{ii}); end
end
end
