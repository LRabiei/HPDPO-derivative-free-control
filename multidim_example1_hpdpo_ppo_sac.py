"""
HPDPO benchmark: 2-state exact Automatica example
=================================================

Reference:
F. Blanchini, P. Bolzern, P. Colaneri, G. De Nicolao, G. Giordano,
"Optimal control of compartmental models: The exact solution,"
Automatica 147 (2023), 110680.

Paper Example 1:
    x1' = -alpha x1
    x2' =  alpha x1 - beta x2

    alpha(t) in [1,4]
    beta(t)  in [2,3]
    T = 1

    minimize J = x2(T)

The paper proves that the optimal control is bang-bang and independent of
the nonnegative initial state.  For a reproducible scalar benchmark here,
we choose

    x(0) = (1,0).

For this initial state, the exact minimizing policy is

    alpha*(t) = 4,  0 <= t < t_s
                1,  t_s <= t <= 1

    beta*(t)  = 3,

with

    t_s = 1 - log(3)/2 = 0.450693855665945...

and

    J_exact = x2(1) = 0.103977422086339...

The HPDPO solver below DOES NOT use the known switching time during search.
At every time knot it uses exactly the same joint-search structure intended
for the two-control benchmarks:

    21x21 global joint grid
        -> 11x11 local joint refinement
        -> 11x11 local joint refinement.

Both control components are optimized simultaneously at every stage.

The state trajectory is propagated with RK4.  The HPDPO score uses the
integrated-by-parts saddle expression

    L = Phi(x_M)
        + sum_k lambda_k^T b_k
        + beta_u * sum ||u_{k+1}-u_k||^2
        - beta_lambda_s * sum ||lambda_{k+1}-lambda_k||^2
        - beta_lambda_2 * sum ||lambda_k||^2,

where

    b_0 = dt f(x_0,u_0),
    b_k = dt f(x_k,u_k) + x_{k-1} - x_k,     k=1,...,M-1,
    b_M = x_{M-1} - x_M.

The U block is minimized with Lambda fixed.  Then the Lambda block is
maximized by backward cyclic coordinate updates with U fixed.

The control smoothing penalty beta_u is set to zero by default so that the
benchmark objective is exactly the paper's objective J=x2(T), without
biasing the known bang-bang solution.

This file is Spyder-friendly: edit the SETTINGS section and press Run.
"""

import csv
import math
import time
from pathlib import Path

import numpy as np


# ======================================================================
# SETTINGS -- edit here in Spyder
# ======================================================================

T = 1.0
CONTROL_DT = 0.01             # 100 control intervals
OUTER_ITERS = 12

# Consistent JOINT search used for the 2-control benchmarks:
#   21x21 global grid over the full admissible rectangle,
#   followed by two 11x11 local joint refinements.
GLOBAL_Q = 21                 # 21x21 = 441 simultaneous action pairs
LOCAL_Q = 11                  # 11x11 = 121 simultaneous action pairs
LOCAL_LEVELS = 2

# Algorithmic regularization.
BETA_U = 0.0                  # keep exact benchmark objective unmodified
BETA_LAMBDA_S = 1.0
BETA_LAMBDA_2 = 0.05
LAMBDA_BOUND = 25.0
LAMBDA_SWEEPS = 3
LAMBDA_RELAX = 0.20

# Initial HPDPO control.  This is deliberately interior, not the exact policy.
INITIAL_ACTION = np.array([2.5, 2.5], dtype=float)

# Reproducible nonnegative initial state chosen for the scalar J comparison.
X0 = np.array([1.0, 0.0], dtype=float)

OUTPUT_DIR = "automatica_2d_exact_hpdpo_results"

SAVE_PLOTS = True


# ======================================================================
# PROBLEM
# ======================================================================

ALPHA_MIN, ALPHA_MAX = 1.0, 4.0
BETA_MIN, BETA_MAX = 2.0, 3.0

M = int(round(T / CONTROL_DT))
if abs(M * CONTROL_DT - T) > 1e-14:
    raise ValueError("T must be an integer multiple of CONTROL_DT.")


def f(x, action):
    """Vector field for one state/action pair."""
    alpha, beta = action
    x1, x2 = x
    return np.array(
        [
            -alpha * x1,
            alpha * x1 - beta * x2,
        ],
        dtype=float,
    )


def f_batch(X, A):
    """
    Vectorized f.

    X: (P,2)
    A: (P,2), columns [alpha,beta]
    """
    alpha = A[:, 0]
    beta = A[:, 1]

    out = np.empty_like(X)
    out[:, 0] = -alpha * X[:, 0]
    out[:, 1] = alpha * X[:, 0] - beta * X[:, 1]
    return out


def rk4_step(x, action):
    dt = CONTROL_DT

    k1 = f(x, action)
    k2 = f(x + 0.5 * dt * k1, action)
    k3 = f(x + 0.5 * dt * k2, action)
    k4 = f(x + dt * k3, action)

    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def rk4_step_batch(X, A):
    dt = CONTROL_DT

    k1 = f_batch(X, A)
    k2 = f_batch(X + 0.5 * dt * k1, A)
    k3 = f_batch(X + 0.5 * dt * k2, A)
    k4 = f_batch(X + dt * k3, A)

    return X + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def rollout(U):
    X = np.empty((M + 1, 2), dtype=float)
    X[0] = X0

    for k in range(M):
        X[k + 1] = rk4_step(X[k], U[k])

    return X


def objective(X):
    """Paper objective for this example: J = x2(T)."""
    return float(X[-1, 1])


# ======================================================================
# EXACT ANALYTIC REFERENCE
# ======================================================================

def exact_switch_time():
    return 1.0 - 0.5 * math.log(3.0)


def exact_objective_x0_10():
    """
    Closed-form J for X0=(1,0), with:
        alpha=4 before t_s,
        alpha=1 after t_s,
        beta=3 throughout.
    """
    ts = exact_switch_time()

    # Contribution from [0,ts]:
    # 4 exp(-4t) exp[-3(1-t)] = 4 exp(-3) exp(-t)
    J1 = 4.0 * math.exp(-3.0) * (1.0 - math.exp(-ts))

    # Contribution from [ts,1]:
    J2 = (
        0.5
        * math.exp(-3.0 - 3.0*ts)
        * (math.exp(2.0) - math.exp(2.0*ts))
    )

    return J1 + J2


def constant_segment_exact(x_start, duration, alpha, beta):
    """Exact state evolution for constant alpha,beta over one duration."""
    x10, x20 = x_start

    x1 = x10 * math.exp(-alpha * duration)

    if abs(beta - alpha) > 1e-14:
        x2 = (
            x20 * math.exp(-beta * duration)
            + alpha * x10
            * (
                math.exp(-alpha * duration)
                - math.exp(-beta * duration)
            )
            / (beta - alpha)
        )
    else:
        # Limit beta -> alpha.
        x2 = (
            x20 * math.exp(-beta * duration)
            + alpha * x10 * duration * math.exp(-alpha * duration)
        )

    return np.array([x1, x2], dtype=float)


def exact_state_at(t):
    """Exact state for the analytic optimal control and X0=(1,0)."""
    ts = exact_switch_time()

    if t <= ts:
        return constant_segment_exact(
            X0,
            t,
            alpha=4.0,
            beta=3.0,
        )

    xs = constant_segment_exact(
        X0,
        ts,
        alpha=4.0,
        beta=3.0,
    )

    return constant_segment_exact(
        xs,
        t - ts,
        alpha=1.0,
        beta=3.0,
    )


# ======================================================================
# HPDPO SADDLE TERMS
# ======================================================================

def coupling_vectors(U, X):
    """
    b vectors multiplying Lambda in the integrated-by-parts saddle.
    """
    B = np.empty((M + 1, 2), dtype=float)

    B[0] = CONTROL_DT * f(X[0], U[0])

    for k in range(1, M):
        B[k] = (
            CONTROL_DT * f(X[k], U[k])
            + X[k - 1]
            - X[k]
        )

    B[M] = X[M - 1] - X[M]

    return B


def lambda_regularization(Lambda):
    return (
        -BETA_LAMBDA_S
        * float(np.sum((Lambda[1:] - Lambda[:-1])**2))
        -BETA_LAMBDA_2
        * float(np.sum(Lambda**2))
    )


def control_regularization(U):
    if BETA_U == 0.0:
        return 0.0

    return BETA_U * float(
        np.sum((U[1:] - U[:-1])**2)
    )


def full_saddle_value(U, X, Lambda):
    B = coupling_vectors(U, X)

    return (
        objective(X)
        + float(np.sum(Lambda * B))
        + control_regularization(U)
        + lambda_regularization(Lambda)
    )


# ======================================================================
# JOINT 2D ACTION SEARCH
# ======================================================================

def joint_global_grid():
    alphas = np.linspace(
        ALPHA_MIN,
        ALPHA_MAX,
        GLOBAL_Q,
    )

    betas = np.linspace(
        BETA_MIN,
        BETA_MAX,
        GLOBAL_Q,
    )

    return np.array(
        [
            [a, b]
            for a in alphas
            for b in betas
        ],
        dtype=float,
    )


JOINT_GRID = joint_global_grid()


def joint_local_grid(center, half_width):
    """Joint 2D local grid around center, clipped to admissible bounds."""
    center = np.asarray(center, dtype=float)
    half_width = np.asarray(half_width, dtype=float)

    lo = np.maximum(
        np.array([ALPHA_MIN, BETA_MIN], dtype=float),
        center - half_width,
    )
    hi = np.minimum(
        np.array([ALPHA_MAX, BETA_MAX], dtype=float),
        center + half_width,
    )

    alphas = np.linspace(lo[0], hi[0], LOCAL_Q)
    betas = np.linspace(lo[1], hi[1], LOCAL_Q)

    grid = np.array(
        [[a, b] for a in alphas for b in betas],
        dtype=float,
    )

    return grid, lo, hi


def unique_rows(A):
    rounded = np.round(
        np.asarray(A, dtype=float),
        14,
    )

    _, idx = np.unique(
        rounded,
        axis=0,
        return_index=True,
    )

    return A[np.sort(idx)]


def local_control_penalty_for_candidates(U, k, C):
    """
    Returns the complete smoothness penalty for each candidate C at time k.

    Only the two neighboring differences can change, so we update the
    baseline penalty locally.
    """
    P = C.shape[0]

    if BETA_U == 0.0:
        return np.zeros(P, dtype=float)

    base = control_regularization(U)

    old_local = 0.0
    new_local = np.zeros(P, dtype=float)

    if k > 0:
        old_local += BETA_U * float(
            np.sum((U[k] - U[k - 1])**2)
        )

        new_local += BETA_U * np.sum(
            (C - U[k - 1])**2,
            axis=1,
        )

    if k < M - 1:
        old_local += BETA_U * float(
            np.sum((U[k + 1] - U[k])**2)
        )

        new_local += BETA_U * np.sum(
            (U[k + 1] - C)**2,
            axis=1,
        )

    return base - old_local + new_local


def candidate_scores(U, X, Lambda, k, C):
    """
    Evaluate all candidate (alpha,beta) pairs at one time index jointly.

    Lambda is FIXED during the complete U sweep.

    Prefix terms before k are identical for every candidate and therefore
    are omitted because they cannot affect argmin selection.
    """
    P = C.shape[0]

    # State x_k is common to all candidates.
    x_prev = np.repeat(
        X[k][None, :],
        P,
        axis=0,
    )

    dual_suffix = np.zeros(P, dtype=float)

    # b_k
    if k == 0:
        b = CONTROL_DT * f_batch(
            x_prev,
            C,
        )
    else:
        b = (
            CONTROL_DT * f_batch(x_prev, C)
            + X[k - 1][None, :]
            - x_prev
        )

    dual_suffix += b @ Lambda[k]

    # Propagate x_{k+1}.
    x_curr = rk4_step_batch(
        x_prev,
        C,
    )

    # Subsequent controls are common across candidates.
    for j in range(k + 1, M):
        A_j = np.repeat(
            U[j][None, :],
            P,
            axis=0,
        )

        b = (
            CONTROL_DT * f_batch(x_curr, A_j)
            + x_prev
            - x_curr
        )

        dual_suffix += b @ Lambda[j]

        x_next = rk4_step_batch(
            x_curr,
            A_j,
        )

        x_prev, x_curr = x_curr, x_next

    # Terminal multiplier coefficient.
    b_M = x_prev - x_curr
    dual_suffix += b_M @ Lambda[M]

    terminal_cost = x_curr[:, 1]

    return (
        terminal_cost
        + dual_suffix
        + local_control_penalty_for_candidates(U, k, C)
    )


def reroll_suffix_in_place(U, X, k, chosen):
    U[k] = chosen

    for j in range(k, M):
        X[j + 1] = rk4_step(
            X[j],
            U[j],
        )


def joint_U_sweep(U, X, Lambda):
    """
    One Gauss-Seidel time sweep using the SAME 2D search protocol as the
    Alzheimer two-control benchmark:

        21x21 global joint grid
            -> 11x11 local joint refinement
            -> 11x11 local joint refinement

    At every stage, alpha and beta are optimized simultaneously.
    Lambda is fixed for the entire U sweep.
    """
    evaluations = 0

    global_spacing = np.array(
        [
            (ALPHA_MAX - ALPHA_MIN) / (GLOBAL_Q - 1),
            (BETA_MAX - BETA_MIN) / (GLOBAL_Q - 1),
        ],
        dtype=float,
    )

    for k in range(M):
        # ----------------------------------------------------------
        # Stage 1: full 21x21 global joint grid.
        # Include current action explicitly so "do not change" is
        # always available even if it is off-grid.
        # ----------------------------------------------------------
        candidates = unique_rows(
            np.vstack(
                [
                    JOINT_GRID,
                    U[k][None, :],
                ]
            )
        )

        scores = candidate_scores(
            U,
            X,
            Lambda,
            k,
            candidates,
        )

        evaluations += len(candidates)

        best = candidates[
            int(np.argmin(scores))
        ].copy()

        best_score = float(
            np.min(scores)
        )

        # ----------------------------------------------------------
        # Stages 2-3: two local 11x11 JOINT refinements.
        # Initial local half-width is one global-grid spacing in
        # each control coordinate.
        # ----------------------------------------------------------
        half_width = global_spacing.copy()

        for _ in range(LOCAL_LEVELS):
            local_grid, lo, hi = joint_local_grid(
                best,
                half_width,
            )

            local_candidates = unique_rows(
                np.vstack(
                    [
                        local_grid,
                        best[None, :],
                        U[k][None, :],
                    ]
                )
            )

            local_scores = candidate_scores(
                U,
                X,
                Lambda,
                k,
                local_candidates,
            )

            evaluations += len(local_candidates)

            jbest = int(
                np.argmin(local_scores)
            )

            if local_scores[jbest] < best_score:
                best_score = float(
                    local_scores[jbest]
                )
                best = local_candidates[
                    jbest
                ].copy()

            # Next refinement radius = one grid spacing of the
            # current local rectangle in each coordinate.
            half_width = np.maximum(
                (hi - lo) / (LOCAL_Q - 1),
                np.finfo(float).eps,
            )

        # Immediate Gauss-Seidel acceptance at time k.
        reroll_suffix_in_place(
            U,
            X,
            k,
            best,
        )

    return U, X, evaluations


# ======================================================================
# LAMBDA MAXIMIZATION
# ======================================================================

def update_lambda(U, X, Lambda):
    """
    Backward cyclic coordinate maximization of Lambda with U fixed.
    """
    B = coupling_vectors(U, X)

    L = Lambda.copy()

    for _ in range(LAMBDA_SWEEPS):
        for k in range(M, -1, -1):

            if k == 0:
                hat = (
                    B[0]
                    + 2.0*BETA_LAMBDA_S*L[1]
                ) / (
                    2.0*BETA_LAMBDA_S
                    + 2.0*BETA_LAMBDA_2
                )

            elif k == M:
                hat = (
                    B[M]
                    + 2.0*BETA_LAMBDA_S*L[M - 1]
                ) / (
                    2.0*BETA_LAMBDA_S
                    + 2.0*BETA_LAMBDA_2
                )

            else:
                hat = (
                    B[k]
                    + 2.0*BETA_LAMBDA_S
                    * (L[k - 1] + L[k + 1])
                ) / (
                    4.0*BETA_LAMBDA_S
                    + 2.0*BETA_LAMBDA_2
                )

            hat = np.clip(
                hat,
                -LAMBDA_BOUND,
                LAMBDA_BOUND,
            )

            L[k] = (
                (1.0 - LAMBDA_RELAX)*L[k]
                + LAMBDA_RELAX*hat
            )

    return L


# ======================================================================
# DIAGNOSTICS
# ======================================================================

def alpha_switch_times(U):
    changes = np.where(
        np.abs(
            np.diff(U[:, 0])
        ) > 1e-10
    )[0]

    return [
        (int(k) + 1) * CONTROL_DT
        for k in changes
    ]


def beta_switch_times(U):
    changes = np.where(
        np.abs(
            np.diff(U[:, 1])
        ) > 1e-10
    )[0]

    return [
        (int(k) + 1) * CONTROL_DT
        for k in changes
    ]


def count_unique_actions(U):
    return np.unique(
        np.round(U, 12),
        axis=0,
    )


# ======================================================================
# SOLVER
# ======================================================================

def solve():
    U = np.repeat(
        INITIAL_ACTION[None, :],
        M,
        axis=0,
    )

    X = rollout(U)

    Lambda = np.zeros(
        (M + 1, 2),
        dtype=float,
    )

    history = []
    score_evaluations = 0

    best = {
        "J": objective(X),
        "U": U.copy(),
        "X": X.copy(),
        "outer": 0,
    }

    tic = time.perf_counter()

    print("="*72)
    print("AUTOMATICA 2D EXACT BENCHMARK — JOINT HPDPO")
    print("="*72)
    print(f"T                       = {T}")
    print(f"control_dt              = {CONTROL_DT}")
    print(f"M                       = {M}")
    print(f"state dimension         = 2")
    print(f"control dimension       = 2")
    print(f"alpha bounds            = [{ALPHA_MIN}, {ALPHA_MAX}]")
    print(f"beta bounds             = [{BETA_MIN}, {BETA_MAX}]")
    print(f"x0                      = {X0}")
    print(f"objective               = minimize x2(T)")
    print(f"joint global grid       = {GLOBAL_Q} x {GLOBAL_Q} = {GLOBAL_Q**2}")
    print(f"joint local grid        = {LOCAL_Q} x {LOCAL_Q} = {LOCAL_Q**2}")
    print(f"local refinement levels = {LOCAL_LEVELS}")
    print(f"beta_u                  = {BETA_U}")
    print(f"beta_lambda_s           = {BETA_LAMBDA_S}")
    print(f"beta_lambda_2           = {BETA_LAMBDA_2}")
    print(f"lambda_relax            = {LAMBDA_RELAX}")
    print("="*72)

    for outer in range(1, OUTER_ITERS + 1):

        # --------------------------------------------------------------
        # U minimization block -- Lambda fixed during the full sweep.
        # --------------------------------------------------------------
        Lambda_fixed = Lambda.copy()

        U, X, ne = joint_U_sweep(
            U,
            X,
            Lambda_fixed,
        )

        score_evaluations += ne

        J = objective(X)

        if J < best["J"]:
            best = {
                "J": float(J),
                "U": U.copy(),
                "X": X.copy(),
                "outer": int(outer),
            }

        # --------------------------------------------------------------
        # Lambda maximization block -- U fixed.
        # --------------------------------------------------------------
        Lambda = update_lambda(
            U,
            X,
            Lambda,
        )

        switches_a = alpha_switch_times(U)
        switches_b = beta_switch_times(U)

        row = {
            "outer": outer,
            "J": float(J),
            "best_J": float(best["J"]),
            "lambda_l2": float(np.linalg.norm(Lambda)),
            "lambda_max_abs": float(np.max(np.abs(Lambda))),
            "alpha_switch_count": len(switches_a),
            "first_alpha_switch": (
                switches_a[0]
                if switches_a
                else np.nan
            ),
            "beta_switch_count": len(switches_b),
            "score_evaluations": score_evaluations,
            "seconds": time.perf_counter() - tic,
        }

        history.append(row)

        print(
            f"[outer {outer:02d}] "
            f"J={J:.12f}  "
            f"best={best['J']:.12f}  "
            f"||lambda||={row['lambda_l2']:.6f}  "
            f"alpha_switches={switches_a}  "
            f"beta_switches={switches_b}"
        )

    return {
        "best": best,
        "Lambda": Lambda,
        "history": history,
        "score_evaluations": score_evaluations,
        "seconds": time.perf_counter() - tic,
    }


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(result):
    out = Path(OUTPUT_DIR)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    best = result["best"]
    U = best["U"]
    X = best["X"]

    exact_ts = exact_switch_time()
    exact_J = exact_objective_x0_10()

    # History.
    with open(
        out / "history.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=list(
                result["history"][0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            result["history"]
        )

    # Best trajectory and exact state sampled at the same knots.
    with open(
        out / "best_trajectory.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as fobj:
        writer = csv.writer(fobj)

        writer.writerow(
            [
                "k",
                "time",
                "x1_hpdpo",
                "x2_hpdpo",
                "x1_exact",
                "x2_exact",
                "alpha_hpdpo",
                "beta_hpdpo",
                "alpha_exact",
                "beta_exact",
            ]
        )

        for k in range(M + 1):
            t = k * CONTROL_DT
            xe = exact_state_at(t)

            if k < M:
                ah = U[k, 0]
                bh = U[k, 1]
                ae = 4.0 if t < exact_ts else 1.0
                be = 3.0
            else:
                ah = np.nan
                bh = np.nan
                ae = np.nan
                be = np.nan

            writer.writerow(
                [
                    k,
                    t,
                    X[k, 0],
                    X[k, 1],
                    xe[0],
                    xe[1],
                    ah,
                    bh,
                    ae,
                    be,
                ]
            )

    np.savez(
        out / "best_result.npz",
        U=U,
        X=X,
        Lambda=result["Lambda"],
        best_J=best["J"],
        best_outer=best["outer"],
        exact_J=exact_J,
        exact_switch_time=exact_ts,
        x0=X0,
        dt=CONTROL_DT,
    )

    # Short text summary.
    switches = alpha_switch_times(U)
    hpdpo_ts = (
        switches[0]
        if switches
        else float("nan")
    )

    rel_J_pct = (
        100.0
        * abs(best["J"] - exact_J)
        / abs(exact_J)
    )

    switch_abs_error = abs(
        hpdpo_ts - exact_ts
    )

    lines = [
        "AUTOMATICA 2D EXACT BENCHMARK",
        "",
        f"Exact switch time          = {exact_ts:.12f}",
        f"HPDPO first alpha switch   = {hpdpo_ts:.12f}",
        f"Absolute switch error      = {switch_abs_error:.12e}",
        "",
        f"Exact J                    = {exact_J:.12f}",
        f"HPDPO J                    = {best['J']:.12f}",
        f"Relative J difference (%)  = {rel_J_pct:.9f}",
        "",
        f"Best outer                 = {best['outer']}",
        f"Score evaluations          = {result['score_evaluations']}",
        f"Wall time (s)              = {result['seconds']:.6f}",
        "",
        f"Unique HPDPO actions:",
        str(count_unique_actions(U)),
    ]

    (out / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    if SAVE_PLOTS:
        try:
            import matplotlib.pyplot as plt

            t_control = np.arange(M) * CONTROL_DT
            t_state = np.arange(M + 1) * CONTROL_DT

            alpha_exact = np.where(
                t_control < exact_ts,
                4.0,
                1.0,
            )

            beta_exact = np.full(
                M,
                3.0,
            )

            plt.figure(figsize=(8, 4.5))
            plt.step(
                t_control,
                U[:, 0],
                where="post",
                label="HPDPO alpha",
            )
            plt.step(
                t_control,
                alpha_exact,
                where="post",
                linestyle="--",
                label="Exact alpha",
            )
            plt.axvline(
                exact_ts,
                linestyle=":",
                label="Exact switch",
            )
            plt.xlabel("time")
            plt.ylabel("alpha")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                out / "alpha_control.png",
                dpi=180,
            )
            plt.close()

            plt.figure(figsize=(8, 4.5))
            plt.step(
                t_control,
                U[:, 1],
                where="post",
                label="HPDPO beta",
            )
            plt.step(
                t_control,
                beta_exact,
                where="post",
                linestyle="--",
                label="Exact beta",
            )
            plt.xlabel("time")
            plt.ylabel("beta")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                out / "beta_control.png",
                dpi=180,
            )
            plt.close()

            X_exact = np.array(
                [
                    exact_state_at(t)
                    for t in t_state
                ]
            )

            plt.figure(figsize=(8, 4.5))
            plt.plot(
                t_state,
                X[:, 0],
                label="HPDPO x1",
            )
            plt.plot(
                t_state,
                X_exact[:, 0],
                linestyle="--",
                label="Exact x1",
            )
            plt.plot(
                t_state,
                X[:, 1],
                label="HPDPO x2",
            )
            plt.plot(
                t_state,
                X_exact[:, 1],
                linestyle="--",
                label="Exact x2",
            )
            plt.xlabel("time")
            plt.ylabel("state")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                out / "states.png",
                dpi=180,
            )
            plt.close()

        except Exception as exc:
            print(
                "Plotting skipped:",
                exc,
            )

    return out



# ======================================================================
# MATCHED PPO / SAC COMPARISON + PUBLICATION TABLE
# ======================================================================
#
# This section extends the original HPDPO benchmark to a matched comparison
# with PPO and SAC.  All methods use:
#   * the same 2-state dynamics,
#   * the same T=1 and dt=0.01,
#   * the same RK4 transition,
#   * the same physical action bounds,
#   * the same initial condition x(0)=(1,0),
#   * the same terminal objective J=x2(T).
#
# PPO/SAC follow the settings used in the manuscript's scalar RL baselines:
#   MlpPolicy, 64x64 networks, lr=3e-4, gamma=1, CPU,
#   250,000 environment transitions, deterministic final evaluation.
#
# For this terminal-cost-only benchmark, the training reward is the
# telescoping quantity
#       r_k = -SCALE * (x2_{k+1} - x2_k).
# Since gamma=1 and x2(0)=0,
#       sum_k r_k = -SCALE * x2(T),
# so this is exactly equivalent to minimizing the original objective while
# providing denser credit assignment than a reward only at the final step.
# The table objective is always recomputed in the ORIGINAL, unscaled units.
# ======================================================================

import os
import platform
import random

try:
    import pandas as pd
    import gymnasium as gym
    from gymnasium import spaces
    import torch
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, SAC
except ImportError as exc:
    raise ImportError(
        "This comparison requires pandas, gymnasium, torch, and stable-baselines3.\n"
        "Install with:\n"
        "  pip install numpy pandas matplotlib gymnasium stable-baselines3 torch\n"
    ) from exc


# -------------------------- comparison settings --------------------------
TOTAL_TIMESTEPS = 250_000
RL_SEED = 0
REWARD_SCALE = 100.0
DEVICE = "cpu"

# Use one CPU thread for more interpretable wall-clock comparisons.
# Remove these two lines if you deliberately want multi-threaded timing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

COMPARISON_OUTPUT_DIR = Path("automatica_multidim1_hpdpo_ppo_sac")
SAVE_MODELS = True
SAVE_COMPARISON_PLOTS = True


# -------------------------- RL environment -------------------------------
class CompartmentalEnv(gym.Env):
    """Gymnasium environment for the exact 2-state / 2-control benchmark."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # SB3 acts in a normalized square [-1,1]^2.  We map to
        # alpha in [1,4] and beta in [2,3].
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Observation = (x1, x2, normalized time k/M).
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.x = None
        self.k = None

    @staticmethod
    def normalized_to_physical(action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        alpha = 2.5 + 1.5 * a[0]   # [-1,1] -> [1,4]
        beta = 2.5 + 0.5 * a[1]    # [-1,1] -> [2,3]
        return np.array([alpha, beta], dtype=float)

    def _obs(self):
        return np.array(
            [self.x[0], self.x[1], self.k / M],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = X0.copy()
        self.k = 0
        return self._obs(), {}

    def step(self, action):
        physical_action = self.normalized_to_physical(action)

        old_x2 = float(self.x[1])
        self.x = rk4_step(self.x, physical_action)
        self.k += 1
        new_x2 = float(self.x[1])

        # Exact telescoping reward for gamma=1:
        # sum r_k = -REWARD_SCALE * x2(T), because x2(0)=0.
        reward = -REWARD_SCALE * (new_x2 - old_x2)

        terminated = self.k >= M
        truncated = False

        info = {
            "physical_action": physical_action.copy(),
            "objective_if_terminal": new_x2 if terminated else None,
        }
        return self._obs(), float(reward), terminated, truncated, info


# -------------------------- references / metrics --------------------------
def exact_control_grid_reference():
    """
    Nearest-grid representation of the exact bang-bang control.

    The continuous switch t_s=0.450693... lies between grid points.  With
    dt=0.01, the nearest admissible switching boundary is 0.45.  This is the
    appropriate control reference for RMSE of piecewise-constant numerical
    policies.  Objective error is still reported against the continuous J*.
    """
    ts = exact_switch_time()
    grid_switch = round(ts / CONTROL_DT) * CONTROL_DT

    t = np.arange(M, dtype=float) * CONTROL_DT
    Ue = np.empty((M, 2), dtype=float)
    Ue[:, 0] = np.where(t < grid_switch, 4.0, 1.0)
    Ue[:, 1] = 3.0
    return Ue, float(grid_switch)


def exact_state_knots():
    t = np.arange(M + 1, dtype=float) * CONTROL_DT
    return np.vstack([exact_state_at(float(tt)) for tt in t])


def joint_control_rmse(U, U_ref):
    # sqrt( mean over time and over the two control components )
    return float(np.sqrt(np.mean((np.asarray(U) - np.asarray(U_ref)) ** 2)))


def joint_state_rmse(X, X_ref):
    # sqrt( mean over time and over the two state components )
    return float(np.sqrt(np.mean((np.asarray(X) - np.asarray(X_ref)) ** 2)))


def estimate_alpha_switch(U, threshold=2.5):
    """
    Estimate a high-to-low alpha switch by midpoint crossing.
    Returns NaN if no such crossing occurs.
    """
    a = np.asarray(U, dtype=float)[:, 0]
    for k in range(1, len(a)):
        if a[k - 1] > threshold and a[k] <= threshold:
            return float(k * CONTROL_DT)
    return float("nan")


def evaluate_policy(model):
    env = CompartmentalEnv()
    obs, _ = env.reset(seed=RL_SEED)

    U = np.empty((M, 2), dtype=float)
    X = np.empty((M + 1, 2), dtype=float)
    X[0] = X0.copy()

    for k in range(M):
        action, _ = model.predict(obs, deterministic=True)
        physical = env.normalized_to_physical(action)
        U[k] = physical

        obs, _, terminated, truncated, _ = env.step(action)
        X[k + 1] = env.x.copy()

        if terminated or truncated:
            if k != M - 1:
                raise RuntimeError("RL episode terminated before the expected horizon.")
            break

    return U, X


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------- PPO / SAC training ----------------------------
def train_ppo():
    set_all_seeds(RL_SEED)
    env = CompartmentalEnv()

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    tic = time.perf_counter()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2000,
        batch_size=100,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=RL_SEED,
        device=DEVICE,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    seconds = time.perf_counter() - tic

    U, X = evaluate_policy(model)
    return model, U, X, float(seconds)


def train_sac():
    set_all_seeds(RL_SEED)
    env = CompartmentalEnv()

    policy_kwargs = dict(net_arch=[64, 64])

    tic = time.perf_counter()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=250_000,
        learning_starts=2000,
        batch_size=256,
        tau=0.005,
        gamma=1.0,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=1,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        seed=RL_SEED,
        device=DEVICE,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    seconds = time.perf_counter() - tic

    U, X = evaluate_policy(model)
    return model, U, X, float(seconds)


# -------------------------- table creation --------------------------------
def method_metrics(name, U, X, seconds, U_ref, X_ref, exact_J, exact_ts):
    J = float(X[-1, 1])
    sw = estimate_alpha_switch(U)

    return {
        "Method": name,
        "Objective J": J,
        "Absolute objective difference": abs(J - exact_J),
        "Relative objective difference (%)": 100.0 * abs(J - exact_J) / abs(exact_J),
        "Control RMSE (2D)": joint_control_rmse(U, U_ref),
        "State RMSE (2D)": joint_state_rmse(X, X_ref),
        "Alpha switch time": sw,
        "Absolute switch-time error": abs(sw - exact_ts) if np.isfinite(sw) else np.nan,
        "Computation time (s)": float(seconds),
    }


def build_and_save_table(hpdpo_result, ppo_pack, sac_pack):
    out = COMPARISON_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    exact_J = exact_objective_x0_10()
    exact_ts = exact_switch_time()
    U_ref, grid_switch = exact_control_grid_reference()
    X_ref = exact_state_knots()

    # Exact/reference row.
    rows = [
        {
            "Method": "Exact",
            "Objective J": exact_J,
            "Absolute objective difference": 0.0,
            "Relative objective difference (%)": 0.0,
            "Control RMSE (2D)": 0.0,
            "State RMSE (2D)": 0.0,
            "Alpha switch time": exact_ts,
            "Absolute switch-time error": 0.0,
            "Computation time (s)": np.nan,
        }
    ]

    # HPDPO row.
    hbest = hpdpo_result["best"]
    rows.append(
        method_metrics(
            "HPDPO",
            hbest["U"],
            hbest["X"],
            hpdpo_result["seconds"],
            U_ref,
            X_ref,
            exact_J,
            exact_ts,
        )
    )

    # PPO and SAC rows.
    _, U_ppo, X_ppo, t_ppo = ppo_pack
    _, U_sac, X_sac, t_sac = sac_pack

    rows.append(method_metrics("PPO", U_ppo, X_ppo, t_ppo, U_ref, X_ref, exact_J, exact_ts))
    rows.append(method_metrics("SAC", U_sac, X_sac, t_sac, U_ref, X_ref, exact_J, exact_ts))

    raw = pd.DataFrame(rows).set_index("Method")
    raw.to_csv(out / "comparison_table_raw.csv")

    # Publication orientation: quantities as rows, methods as columns.
    pub = raw.T
    pub.to_csv(out / "comparison_table.csv")
    pub.to_latex(
        out / "comparison_table.tex",
        float_format=lambda z: f"{z:.8g}",
        na_rep="--",
        escape=False,
    )

    # Save all trajectories/actions for later plotting or manuscript figures.
    np.savez(
        out / "comparison_results.npz",
        U_exact_grid=U_ref,
        X_exact=X_ref,
        exact_J=exact_J,
        exact_switch_time=exact_ts,
        grid_switch_time=grid_switch,
        U_hpdpo=hbest["U"],
        X_hpdpo=hbest["X"],
        U_ppo=U_ppo,
        X_ppo=X_ppo,
        U_sac=U_sac,
        X_sac=X_sac,
        hpdpo_seconds=hpdpo_result["seconds"],
        ppo_seconds=t_ppo,
        sac_seconds=t_sac,
        dt=CONTROL_DT,
    )

    # Timing/context metadata: absolute wall times are machine dependent.
    with open(out / "run_metadata.txt", "w", encoding="utf-8") as fobj:
        fobj.write(f"Platform: {platform.platform()}\n")
        fobj.write(f"Processor: {platform.processor()}\n")
        fobj.write(f"Python: {platform.python_version()}\n")
        fobj.write(f"NumPy: {np.__version__}\n")
        fobj.write(f"PyTorch: {torch.__version__}\n")
        fobj.write(f"Stable-Baselines3: {sb3.__version__}\n")
        fobj.write(f"Device: {DEVICE}\n")
        fobj.write(f"Torch threads: {torch.get_num_threads()}\n")
        fobj.write(f"RL seed: {RL_SEED}\n")
        fobj.write(f"RL transitions per method: {TOTAL_TIMESTEPS}\n")
        fobj.write(f"Reward scale: {REWARD_SCALE}\n")
        fobj.write(f"dt: {CONTROL_DT}\n")
        fobj.write(f"M: {M}\n")
        fobj.write(f"Exact continuous switch: {exact_ts:.12f}\n")
        fobj.write(f"Nearest grid switch used for control RMSE: {grid_switch:.12f}\n")

    if SAVE_COMPARISON_PLOTS:
        try:
            import matplotlib.pyplot as plt

            tc = np.arange(M) * CONTROL_DT
            tx = np.arange(M + 1) * CONTROL_DT

            # Alpha controls.
            plt.figure(figsize=(8, 4.5))
            plt.step(tc, U_ref[:, 0], where="post", label="Exact/grid alpha")
            plt.step(tc, hbest["U"][:, 0], where="post", label="HPDPO alpha")
            plt.plot(tc, U_ppo[:, 0], label="PPO alpha", alpha=0.85)
            plt.plot(tc, U_sac[:, 0], label="SAC alpha", alpha=0.85)
            plt.axvline(exact_ts, linestyle=":", label="Exact switch")
            plt.xlabel("time")
            plt.ylabel("alpha")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "alpha_comparison.png", dpi=180)
            plt.close()

            # Beta controls.
            plt.figure(figsize=(8, 4.5))
            plt.step(tc, U_ref[:, 1], where="post", label="Exact beta")
            plt.step(tc, hbest["U"][:, 1], where="post", label="HPDPO beta")
            plt.plot(tc, U_ppo[:, 1], label="PPO beta", alpha=0.85)
            plt.plot(tc, U_sac[:, 1], label="SAC beta", alpha=0.85)
            plt.xlabel("time")
            plt.ylabel("beta")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "beta_comparison.png", dpi=180)
            plt.close()

            # State trajectories.
            plt.figure(figsize=(8, 4.5))
            plt.plot(tx, X_ref[:, 1], label="Exact x2")
            plt.plot(tx, hbest["X"][:, 1], linestyle="--", label="HPDPO x2")
            plt.plot(tx, X_ppo[:, 1], label="PPO x2")
            plt.plot(tx, X_sac[:, 1], label="SAC x2")
            plt.xlabel("time")
            plt.ylabel("x2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "x2_comparison.png", dpi=180)
            plt.close()

        except Exception as exc:
            print("Comparison plotting skipped:", exc)

    return raw, pub, out


def print_publication_table(pub):
    def fmt(v):
        if pd.isna(v):
            return "--"
        av = abs(float(v))
        if av != 0.0 and (av < 1e-4 or av >= 1e4):
            return f"{float(v):.4e}"
        return f"{float(v):.8f}".rstrip("0").rstrip(".")

    display = pub.copy()
    for c in display.columns:
        display[c] = display[c].map(fmt)

    print("\n" + "=" * 100)
    print("MULTIDIMENSIONAL EXAMPLE 1 — EXACT / HPDPO / PPO / SAC")
    print("=" * 100)
    print(display.to_string())
    print("=" * 100)


# -------------------------- complete experiment ---------------------------
def comparison_main():
    print("\nRunning HPDPO ...")
    hpdpo_result = solve()

    print("\nRunning PPO on CPU ...")
    ppo_pack = train_ppo()

    print("\nRunning SAC on CPU ...")
    sac_pack = train_sac()

    if SAVE_MODELS:
        COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ppo_pack[0].save(COMPARISON_OUTPUT_DIR / "ppo_model")
        sac_pack[0].save(COMPARISON_OUTPUT_DIR / "sac_model")

    raw, pub, out = build_and_save_table(hpdpo_result, ppo_pack, sac_pack)
    print_publication_table(pub)

    print(f"\nSaved table: {out / 'comparison_table.csv'}")
    print(f"Saved LaTeX: {out / 'comparison_table.tex'}")
    print(f"Saved raw metrics: {out / 'comparison_table_raw.csv'}")
    print(f"Saved trajectories: {out / 'comparison_results.npz'}")
    print(f"Saved run metadata: {out / 'run_metadata.txt'}")


if __name__ == "__main__":
    comparison_main()
