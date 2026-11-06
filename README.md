# WDSN: ADMM, NNLS-DCA and the proximal operator

MATLAB implementations for the regularized weighted difference of squared norms (WDSN) model:

```text
min_x  0.5 * ||A*x - b||_2^2 + lambda * (||x||_1^2 - eta * ||x||_2^2).
```

This repository provides selected core algorithms associated with
*Sparse Signal Recovery via l1^2 - eta l2^2 Minimization: Recovery Guarantees and Efficient Algorithms with Global Convergence*,
by Lang Yu and Nanjing Huang.

## Quick start

Download the repository using **Code > Download ZIP**, extract it, and set the MATLAB current folder to the extracted directory. Run:

```matlab
results = demo_wdsn;
test_wdsn;
```

The demo generates a column-normalized Gaussian matrix, a sparse signal and noiseless observations using `rng(42, 'twister')`. The generated data and both solver outputs are returned in `results`; no external dataset is needed. This small example demonstrates the algorithms and is separate from the full experiment suite in the paper.

## Files

| File | Purpose |
| --- | --- |
| `mL12mL22_ADMM_twophase.m` | Two-stage ADMM with a fixed penalty in each stage |
| `mL12mL22_regularized_ADMM_zfirst.m` | Fixed-penalty ADMM with the proximal block updated first |
| `admm_zfirst_monitor.m` | Stage-one augmented-Lagrangian monitor used by two-stage ADMM |
| `mL12mL22_DCA_NNLS.m` | Proximal DCA using a warm-started Lawson-Hanson NNLS solver |
| `exact_prox_L1sq_L2sq_wdsn.m` | One exact minimizer of the WDSN proximal subproblem |
| `demo_wdsn.m` | Reproducible synthetic-data example |
| `test_wdsn.m` | Proximal branch, subproblem KKT, descent and ADMM residual checks |

The five algorithm files retain the source implementations and function names. ADMM includes a local copy of the proximal routine; the standalone proximal file exposes the same computation for direct use.

## Calling the algorithms

Supply a real matrix `A` of size `m` by `n` and a real column vector `b` of length `m`:

```matlab
pm = struct('lambda', 1e-3, 'eta', 1, ...
    'x0', zeros(size(A, 2), 1), 'max_iter', 10000);
[x_nnls, info_nnls] = mL12mL22_DCA_NNLS(A, b, pm);

pm.rho = 1.1 * sqrt(2) * norm(A, 2)^2;
pm.rel_tol = 1e-7;
pm.abs_tol = 1e-8;
[x_admm, info_admm] = mL12mL22_regularized_ADMM_zfirst(A, b, pm);

% For two-stage ADMM, start from a smaller penalty:
pm.rho = 0.1 * sqrt(2) * norm(A, 2)^2;
[x_two_stage, info_two_stage] = mL12mL22_ADMM_twophase(A, b, pm);

% Standalone proximal operator; tau is the proximal step size:
x_prox = exact_prox_L1sq_L2sq_wdsn([1; -0.5; 0.2], 0.1, 1);
```

Always supply `pm.x0` for a self-contained ADMM run. The original ADMM initialization also supports an optional external `mL1_constrained_LP_Gurobi` helper; that helper is outside this release. If no initial point is supplied and the helper is unavailable, ADMM warns and falls back to a minimum-norm least-squares initialization.

NNLS-DCA uses the included `lh_warm` backend by default. Its optional `lsqnonneg` and `gurobi` backends require those functions to be available in the user's environment. The demo and checks use `lh_warm` and do not require Gurobi.

## Parameters and output

- `lambda > 0`, `eta >= 0`. For NNLS-DCA, `c_factor > 0` and `tol > 0`; the proximal coefficient is `c = 2*lambda*c_factor`.
- The proximal routine requires `tau > 0` and `1 + 2*tau*(1-eta) > 0`. It returns a column vector and selects the first maximal-magnitude index when the solution is nonunique in the nonpositive quadratic-coefficient branch.
- ADMM requires `rho > 0` and `rho + 2*lambda*(1-eta) > 0`. Two-stage ADMM defaults to `q_theory = 1.1`, `K0_fraction = 0.8` and `viol_window = 50`; use `q_theory > 1` and `0 < K0_fraction < 1`.
- The convergence results require the assumptions stated in the paper. In particular, `eta > 1` requires a separate boundedness/coercivity condition on the regularized model. The demo uses `eta = 1`.
- `info.converged` reports whether the numerical stopping test was met. It does not certify a global minimizer. Both solvers return objective and timing histories; ADMM also returns primal/dual residuals and their thresholds. The ADMM output `x` is the sparse proximal block `z`.
- The optional `pm.xg` is a ground-truth signal used only for reporting recovery error; it is not needed by the solver.

## Release scope and data

This release contains ADMM, NNLS-DCA and the WDSN proximal operator, with a runnable synthetic-data example. The paper uses synthetic data. Full parameter sweeps, comparison-method implementations, plotting scripts and archived experiment outputs are outside this selected release.

## Validation

Tested with MATLAB R2025b on Windows after `restoredefaultpath`, using only this repository and the standard MATLAB path. `test_wdsn` passed. In the included example, ADMM stopped after 115 iterations and NNLS-DCA after 6 iterations; both returned an objective of approximately `1.846203e-02` and a relative recovery error of approximately `1.232e-02`. These are results for this example, not a general performance comparison.

Run `test_wdsn` to check the zero input, tied and nearly tied maxima, positive/zero/negative quadratic-coefficient proximal branches, the proximal solvability guard, a one-step NNLS-DCA KKT residual, objective descent, and final ADMM residual thresholds on the demo.
