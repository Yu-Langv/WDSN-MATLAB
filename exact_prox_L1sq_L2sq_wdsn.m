function x = exact_prox_L1sq_L2sq_wdsn(v, tau, eta)
%EXACT_PROX_L1SQ_L2SQ_WDSN Prox for tau*(||x||_1^2 - eta*||x||_2^2).
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
