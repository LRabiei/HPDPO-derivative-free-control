"""
Published Alzheimer CMO benchmark solved with joint HPDPO
=========================================================

Reference
---------
I. M. Bulai, F. Ferraresso, F. Gladiali,
"Optimal control of monomers and oligomers degradation in an Alzheimer's
disease model,"
Journal of Mathematical Biology 91, 27 (2025).
DOI: 10.1007/s00285-025-02256-3

This script reproduces the paper's CMO problem:
    two DIFFERENT simultaneous treatments
        v(t): monomer degradation
        u(t): oligomer degradation

and compares a genuinely JOINT HPDPO action search against the published
numerical results.

No component-wise action update is used in the main solvers.

Published CMO reference (Table 5)
---------------------------------
    J(v*,u*)      = 0.1148
    dose(v*)      = 1.439
    dose(u*)      = 2.306

Published baselines:
    no treatment:
        J = 0.2529
        dose(v)=dose(u)=0

    maximum treatment:
        J = 3.619
        dose(v)=dose(u)=29.97

The maximum therapy rate used in the numerical experiments is 0.03 s^-1.
The 29.97 reported maximum dose is consistent with the paper's numerical
time discretization over T=1000 s.

States
------
    x = [M, U2, U3, U4, O]

M  : monomers
U2 : proto-oligomers of length 2
U3 : proto-oligomers of length 3
U4 : proto-oligomers of length 4
O  : oligomers (length >= 5)

Published baseline numerical setup
----------------------------------
    T = 1000 s

    s = 1e-4
    K = 1
    m = 1e-4

    r2 = r3 = r4 = 100
    b = 1e-3
    beta = 1e-4
    delta = 5e-4

    a11 = 10
    a22 = 2.5
    a23 = 1.67
    a24 = 1.25
    a33 = 1.11
    a34 = 0.83
    a44 = 0.625

Initial state:
    M(0)  = 1e-3
    U2(0) = 0
    U3(0) = 0
    U4(0) = 0
    O(0)  = 0

CMO objective
-------------
Minimize

    J(v,u)
      = c1*M(T) + c2*O(T)
        + integral_0^T [
              c3*M + c4*O
            + c5*v + c6*v^2
            + c7*u + c8*u^2
            + c9*v*u
          ] dt

with

    c1=c2=c3=c4 = 1
    c5=c7=c9    = 1e-5
    c6=c8       = 2

and

    0 <= v(t) <= 0.03
    0 <= u(t) <= 0.03.

The interaction term c9*v*u means the two controls are explicitly coupled,
so a joint vector search is the natural multidimensional HPDPO test.

HPDPO methods
-------------
1. joint
       Full q x q joint grid at each time point.
       Default q=21:
           21^2 = 441 simultaneous 2D action candidates.

2. joint_refine
       q x q global joint grid
       + two local 11 x 11 joint refinements.

3. joint_powell
       5 x 5 coarse JOINT grid
       + Powell derivative-free local optimization in R^2.

The code evaluates each candidate by replacing the full vector
    (v_k,u_k)
at one time point and rerolling the downstream nonlinear ODE suffix.

Numerical integration
---------------------
The treatment policy is piecewise constant over a policy interval
`control_dt`.

Default:
    control_dt = 5 s

Within every policy interval, RK4 uses internal substeps no larger than
    0.5 s.

This separates policy resolution from ODE integration accuracy.

IMPORTANT NUMERICAL CORRECTION IN THIS VERSION
------------------------------------------------
The HPDPO candidate search and the final reported objective now use the SAME
running-cost quadrature.

For every piecewise-constant control interval, the state and the accumulated
running cost are integrated together with the same internal RK4 substeps
(maximum internal step = 0.5 s).

Thus HPDPO no longer selects candidates using a left-endpoint approximation
while reporting them with a more accurate RK4-integrated objective.

The saddle score now contains

    C_k = integral_{t_k}^{t_{k+1}} ell(x(t),a_k) dt

for the running-cost contribution of interval k, while the HPDPO multiplier
term retains the original left-endpoint discretization

    control_dt * lambda_k^T f(x_k,a_k).

This is the version recommended for the final paper comparison.

Recommended usage
-----------------
Smoke test:
    python alzheimer_cmo_joint_hpdpo.py --method joint --quick

Main published comparison:
    python alzheimer_cmo_joint_hpdpo.py --method joint

Other joint solvers:
    python alzheimer_cmo_joint_hpdpo.py --method joint_refine
    python alzheimer_cmo_joint_hpdpo.py --method joint_powell

Run all:
    python alzheimer_cmo_joint_hpdpo.py --method all

Finer control grid:
    python alzheimer_cmo_joint_hpdpo.py --method joint --control_dt 2.5
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ============================================================
# 1. PUBLISHED MODEL PARAMETERS
# ============================================================

TF = 1000.0

S = 1e-4
K = 1.0

R2 = 100.0
R3 = 100.0
R4 = 100.0

B = 1e-3
BETA = 1e-4
DELTA = 5e-4

A11 = 10.0
A22 = 2.5
A23 = 1.67
A24 = 1.25
A33 = 1.11
A34 = 0.83
A44 = 0.625

M_COMP = 1e-4

X0 = np.array(
    [1e-3, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)

STATE_DIM = 5
ACTION_DIM = 2

# Physical treatment-rate bounds.
U_LO = np.array([0.0, 0.0], dtype=float)       # [v,u]
U_HI = np.array([0.03, 0.03], dtype=float)


# ============================================================
# 2. PUBLISHED CMO OBJECTIVE WEIGHTS
# ============================================================

C1 = 1.0
C2 = 1.0
C3 = 1.0
C4 = 1.0

C5 = 1e-5
C6 = 2.0

C7 = 1e-5
C8 = 2.0

C9 = 1e-5


# Published Table 5 references
PAPER_J_OPT = 0.1148
PAPER_DOSE_V_OPT = 1.439
PAPER_DOSE_U_OPT = 2.306

PAPER_J_NO_CONTROL = 0.2529
PAPER_J_MAX_CONTROL = 3.619

PAPER_DOSE_MAX = 29.97


# ============================================================
# 3. NONLINEAR CMO ODE SYSTEM
# ============================================================

def rhs(x, action):
    """
    Published model (1), CMO case alpha0=alpha1=1.

    action = [v,u]
        v: monomer degradation
        u: oligomer degradation
    """
    M, U2, U3, U4, O = x
    v, u = action

    dM = (
        S * M * (1.0 - M / K)
        - M * (R2*U2 + R3*U3 + R4*U4)
        + (B + 2.0*BETA) * U3
        + (B + 2.0*BETA) * U4
        + 2.0*BETA * U2
        - A11 * M*M
        - DELTA * M
        - v * M
    )

    dU2 = (
        -R2 * M * U2
        - BETA * U2
        + (B + 2.0*BETA) * U3
        + 2.0*BETA * U4
        - A23 * U2 * U3
        - A24 * U2 * U4
        - A22 * U2*U2
        + A11 * M*M
    )

    dU3 = (
        -M * (R3*U3 - R2*U2)
        - (2.0*BETA + B) * U3
        + (B + 2.0*BETA) * U4
        - A23 * U2 * U3
        - A34 * U3 * U4
        - A33 * U3*U3
    )

    dU4 = (
        -M * (R4*U4 - R3*U3)
        - (B + 3.0*BETA) * U4
        - A24 * U2 * U4
        - A34 * U3 * U4
        - A44 * U4*U4
        + A22 * U2*U2
    )

    dO = (
        R4 * M * U4
        + A23 * U2 * U3
        + A24 * U2 * U4
        + A34 * U3 * U4
        + A33 * U3*U3
        + A44 * U4*U4
        - M_COMP * O*O
        - u * O
    )

    return np.array(
        [dM, dU2, dU3, dU4, dO],
        dtype=float,
    )


def rhs_batch(X, A):
    """
    Vectorized state RHS.

    X shape (...,5)
    A shape (...,2)
    """
    M = X[..., 0]
    U2 = X[..., 1]
    U3 = X[..., 2]
    U4 = X[..., 3]
    O = X[..., 4]

    v = A[..., 0]
    u = A[..., 1]

    dM = (
        S * M * (1.0 - M / K)
        - M * (R2*U2 + R3*U3 + R4*U4)
        + (B + 2.0*BETA) * U3
        + (B + 2.0*BETA) * U4
        + 2.0*BETA * U2
        - A11 * M*M
        - DELTA * M
        - v * M
    )

    dU2 = (
        -R2 * M * U2
        - BETA * U2
        + (B + 2.0*BETA) * U3
        + 2.0*BETA * U4
        - A23 * U2 * U3
        - A24 * U2 * U4
        - A22 * U2*U2
        + A11 * M*M
    )

    dU3 = (
        -M * (R3*U3 - R2*U2)
        - (2.0*BETA + B) * U3
        + (B + 2.0*BETA) * U4
        - A23 * U2 * U3
        - A34 * U3 * U4
        - A33 * U3*U3
    )

    dU4 = (
        -M * (R4*U4 - R3*U3)
        - (B + 3.0*BETA) * U4
        - A24 * U2 * U4
        - A34 * U3 * U4
        - A44 * U4*U4
        + A22 * U2*U2
    )

    dO = (
        R4 * M * U4
        + A23 * U2 * U3
        + A24 * U2 * U4
        + A34 * U3 * U4
        + A33 * U3*U3
        + A44 * U4*U4
        - M_COMP * O*O
        - u * O
    )

    return np.stack(
        [dM, dU2, dU3, dU4, dO],
        axis=-1,
    )


# ============================================================
# 4. OBJECTIVE
# ============================================================

def running_cost(x, action):
    M = x[0]
    O = x[4]

    v, u = action

    return float(
        C3*M
        + C4*O
        + C5*v
        + C6*v*v
        + C7*u
        + C8*u*u
        + C9*v*u
    )


def running_cost_batch(X, A):
    M = X[..., 0]
    O = X[..., 4]

    v = A[..., 0]
    u = A[..., 1]

    return (
        C3*M
        + C4*O
        + C5*v
        + C6*v*v
        + C7*u
        + C8*u*u
        + C9*v*u
    )


def terminal_cost(x):
    return float(
        C1*x[0] + C2*x[4]
    )


# ============================================================
# 5. POLICY GRID AND RK4
# ============================================================

MAX_INTERNAL_STEP = 0.5


def make_grid(control_dt):
    M = int(round(TF / control_dt))

    if not np.isclose(M*control_dt, TF):
        raise ValueError(
            "Choose control_dt so that 1000/control_dt is an integer."
        )

    times = np.linspace(
        0.0,
        TF,
        M + 1,
    )

    return M, times


def substep_info(control_dt):
    n = int(
        math.ceil(
            control_dt / MAX_INTERNAL_STEP
        )
    )

    return n, control_dt / n


def augmented_rhs(y, action):
    """
    Augmented state:
        y = [x, accumulated_running_cost].

    The final component satisfies
        dI/dt = ell(x,action).
    """
    x = y[:STATE_DIM]

    return np.r_[
        rhs(x, action),
        running_cost(x, action),
    ]


def augmented_rhs_batch(Y, A):
    """
    Vectorized augmented dynamics.

    Y shape (..., STATE_DIM+1)
    A shape (..., ACTION_DIM)
    """
    X = Y[..., :STATE_DIM]

    return np.concatenate(
        [
            rhs_batch(X, A),
            running_cost_batch(X, A)[..., None],
        ],
        axis=-1,
    )


def rk4_aug_step(y, action, h):
    k1 = augmented_rhs(y, action)
    k2 = augmented_rhs(
        y + 0.5*h*k1,
        action,
    )
    k3 = augmented_rhs(
        y + 0.5*h*k2,
        action,
    )
    k4 = augmented_rhs(
        y + h*k3,
        action,
    )

    return (
        y
        + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
    )


def rk4_aug_step_batch(Y, A, h):
    k1 = augmented_rhs_batch(Y, A)
    k2 = augmented_rhs_batch(
        Y + 0.5*h*k1,
        A,
    )
    k3 = augmented_rhs_batch(
        Y + 0.5*h*k2,
        A,
    )
    k4 = augmented_rhs_batch(
        Y + h*k3,
        A,
    )

    return (
        Y
        + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
    )


def interval_step_and_cost(x, action, control_dt):
    """
    Integrate one policy interval with the action held constant.

    Returns
    -------
    x_next : state at the end of the interval
    cost   : integral of running_cost over the same interval
    """
    n, h = substep_info(control_dt)

    y = np.r_[
        np.asarray(x, dtype=float),
        0.0,
    ]

    for _ in range(n):
        y = rk4_aug_step(
            y,
            action,
            h,
        )

    return (
        y[:STATE_DIM].copy(),
        float(y[-1]),
    )


def interval_step_and_cost_batch(X, A, control_dt):
    """
    Vectorized interval integration.

    X shape (N,STATE_DIM)
    A shape (N,ACTION_DIM)

    Returns
    -------
    X_next : (N,STATE_DIM)
    C      : (N,) integrated running cost for this interval
    """
    n, h = substep_info(control_dt)

    Y = np.concatenate(
        [
            np.asarray(X, dtype=float),
            np.zeros(
                (X.shape[0], 1),
                dtype=float,
            ),
        ],
        axis=1,
    )

    for _ in range(n):
        Y = rk4_aug_step_batch(
            Y,
            A,
            h,
        )

    return (
        Y[:, :STATE_DIM].copy(),
        Y[:, -1].copy(),
    )


def interval_step(x, action, control_dt):
    x_next, _ = interval_step_and_cost(
        x,
        action,
        control_dt,
    )
    return x_next


def interval_step_batch(X, A, control_dt):
    X_next, _ = interval_step_and_cost_batch(
        X,
        A,
        control_dt,
    )
    return X_next


def rollout_with_costs(Actions, control_dt):
    """
    Roll out the complete policy and store the RK4-integrated running cost
    of every policy interval.
    """
    M = Actions.shape[0]

    X = np.empty(
        (M + 1, STATE_DIM),
        dtype=float,
    )

    C = np.empty(
        M,
        dtype=float,
    )

    X[0] = X0

    for k in range(M):
        X[k + 1], C[k] = interval_step_and_cost(
            X[k],
            Actions[k],
            control_dt,
        )

    return X, C


def rollout(Actions, control_dt):
    X, _ = rollout_with_costs(
        Actions,
        control_dt,
    )
    return X


# ============================================================
# 6. CONSISTENT OBJECTIVE EVALUATION
# ============================================================

def objective_discrete(
    Actions,
    control_dt,
    X=None,
    interval_costs=None,
):
    """
    Objective used BOTH for HPDPO optimization and final reporting.

    Running cost is integrated through the same internal RK4 substeps used
    to propagate the state:

        J = Phi(x_M) + sum_k C_k,

    where
        C_k = integral_{t_k}^{t_{k+1}} ell(x(t),a_k) dt.

    The historical function name `objective_discrete` is retained only to
    minimize changes elsewhere in the script.
    """
    if X is None or interval_costs is None:
        X, interval_costs = rollout_with_costs(
            Actions,
            control_dt,
        )

    J = (
        terminal_cost(X[-1])
        + np.sum(interval_costs)
    )

    return float(J), X


def objective_report(Actions, control_dt):
    """
    Final paper-comparison objective.

    It is now exactly the same numerical objective used to rank HPDPO
    candidates.
    """
    X, C = rollout_with_costs(
        Actions,
        control_dt,
    )

    return float(
        terminal_cost(X[-1])
        + np.sum(C)
    )


def dose(Actions, control_dt):
    return (
        control_dt
        * np.sum(
            Actions,
            axis=0,
        )
    )


# ============================================================
# 7. PUBLISHED BASELINE VALIDATION
# ============================================================

def baseline_validation(control_dt):
    M, _ = make_grid(
        control_dt
    )

    A_zero = np.zeros(
        (M, ACTION_DIM),
        dtype=float,
    )

    A_max = np.tile(
        U_HI,
        (M, 1),
    )

    J_zero = objective_report(
        A_zero,
        control_dt,
    )

    J_max = objective_report(
        A_max,
        control_dt,
    )

    d_zero = dose(
        A_zero,
        control_dt,
    )

    d_max = dose(
        A_max,
        control_dt,
    )

    return {
        "J_zero": J_zero,
        "J_max": J_max,
        "dose_zero": d_zero,
        "dose_max": d_max,
    }


# ============================================================
# 8. HPDPO REGULARIZED SADDLE SCORE
# ============================================================

# No additional control smoothness: the published objective already contains
# treatment and quadratic side-effect costs.
BETA_ACTION_SMOOTH = 0.0

BETA_LAMBDA_SMOOTH = 1.0
BETA_LAMBDA_L2 = 0.05
LAMBDA_BOUND = 25.0
LAMBDA_RELAX = 0.20


def L_ibp(
    Actions,
    Lambda,
    control_dt,
    X=None,
    interval_costs=None,
):
    """
    Integrated-by-parts HPDPO score.

    Numerical correction:
        the running contribution is sum_k interval_costs[k], where each
        interval cost is integrated with the same internal RK4 quadrature
        used for final reporting.

    The multiplier/Hamiltonian term remains the original left-endpoint
    discretization:
        control_dt * lambda_k^T f(x_k,a_k).
    """
    if X is None or interval_costs is None:
        X, interval_costs = rollout_with_costs(
            Actions,
            control_dt,
        )

    M = Actions.shape[0]

    value = float(
        np.sum(interval_costs)
    )

    for k in range(M):
        f_k = rhs(
            X[k],
            Actions[k],
        )

        value += (
            control_dt
            * Lambda[k]
            @ f_k

            + (
                Lambda[k + 1]
                - Lambda[k]
            )
            @ X[k]
        )

    value += (
        -Lambda[M] @ X[M]
        + Lambda[0] @ X[0]
        + terminal_cost(X[M])
    )

    return float(value)


def L_reg(
    Actions,
    Lambda,
    control_dt,
    X=None,
    interval_costs=None,
):
    if X is None or interval_costs is None:
        X, interval_costs = rollout_with_costs(
            Actions,
            control_dt,
        )

    value = L_ibp(
        Actions,
        Lambda,
        control_dt,
        X,
        interval_costs,
    )

    if BETA_ACTION_SMOOTH > 0.0:
        value += (
            BETA_ACTION_SMOOTH
            * np.sum(
                (
                    Actions[1:]
                    - Actions[:-1]
                )**2
            )
        )

    value -= (
        BETA_LAMBDA_SMOOTH
        * np.sum(
            (
                Lambda[1:]
                - Lambda[:-1]
            )**2
        )
    )

    value -= (
        BETA_LAMBDA_L2
        * np.sum(
            Lambda**2
        )
    )

    return float(value)


def L_reg_batch(
    A_batch,
    X_batch,
    C_batch,
    Lambda,
    control_dt,
):
    """
    Vectorized L_reg for candidate policies.

    C_batch[n,k] is the RK4-integrated running cost of interval k for
    candidate policy n.
    """
    M = A_batch.shape[1]

    Xk = X_batch[:, :M, :]

    F = rhs_batch(
        Xk,
        A_batch,
    )

    running = np.sum(
        C_batch,
        axis=1,
    )

    lambda_f = (
        control_dt
        * np.einsum(
            "ti,nti->n",
            Lambda[:M],
            F,
        )
    )

    dLambda = (
        Lambda[1:]
        - Lambda[:-1]
    )

    lambda_x = np.einsum(
        "ti,nti->n",
        dLambda,
        Xk,
    )

    boundary = (
        -X_batch[:, M, :]
        @ Lambda[M]
        + Lambda[0] @ X0
        + C1*X_batch[:, M, 0]
        + C2*X_batch[:, M, 4]
    )

    value = (
        running
        + lambda_f
        + lambda_x
        + boundary
    )

    if BETA_ACTION_SMOOTH > 0.0:
        value += (
            BETA_ACTION_SMOOTH
            * np.sum(
                (
                    A_batch[:, 1:, :]
                    - A_batch[:, :-1, :]
                )**2,
                axis=(1, 2),
            )
        )

    value += (
        -BETA_LAMBDA_SMOOTH
        * np.sum(
            (
                Lambda[1:]
                - Lambda[:-1]
            )**2
        )
        -BETA_LAMBDA_L2
        * np.sum(
            Lambda**2
        )
    )

    return value


# ============================================================
# 9. JOINT CANDIDATE SUFFIX EVALUATION
# ============================================================

def evaluate_candidates(
    Actions,
    X,
    interval_costs,
    Lambda,
    k,
    candidates,
    control_dt,
):
    """
    candidates shape (N,2).

    The COMPLETE action pair (v_k,u_k) is replaced jointly.

    Both state trajectories and RK4-integrated interval running costs are
    rerolled from time k onward.
    """
    N = candidates.shape[0]
    M = Actions.shape[0]

    A_batch = np.repeat(
        Actions[None, :, :],
        N,
        axis=0,
    )

    A_batch[:, k, :] = candidates

    X_batch = np.repeat(
        X[None, :, :],
        N,
        axis=0,
    )

    C_batch = np.repeat(
        interval_costs[None, :],
        N,
        axis=0,
    )

    X_batch[:, k, :] = X[k]

    for j in range(
        k,
        M,
    ):
        (
            X_batch[:, j + 1, :],
            C_batch[:, j],
        ) = interval_step_and_cost_batch(
            X_batch[:, j, :],
            A_batch[:, j, :],
            control_dt,
        )

    scores = L_reg_batch(
        A_batch,
        X_batch,
        C_batch,
        Lambda,
        control_dt,
    )

    idx = int(
        np.argmin(scores)
    )

    return (
        candidates[idx].copy(),
        float(scores[idx]),
        X_batch[idx].copy(),
        C_batch[idx].copy(),
        int(N),
    )


def reroll_suffix(
    X,
    interval_costs,
    Actions,
    k,
    control_dt,
):
    X_new = X.copy()
    C_new = interval_costs.copy()

    for j in range(
        k,
        Actions.shape[0],
    ):
        X_new[j + 1], C_new[j] = interval_step_and_cost(
            X_new[j],
            Actions[j],
            control_dt,
        )

    return X_new, C_new


# ============================================================
# 10. CLOSED-FORM REGULARIZED DUAL UPDATE
# ============================================================

def update_lambda(
    Actions,
    Lambda,
    X,
    control_dt,
    sweeps=3,
):
    M = Actions.shape[0]

    bvec = np.zeros(
        (M + 1, STATE_DIM),
        dtype=float,
    )

    bvec[0] = (
        control_dt
        * rhs(
            X[0],
            Actions[0],
        )
    )

    for k in range(
        1,
        M,
    ):
        bvec[k] = (
            control_dt
            * rhs(
                X[k],
                Actions[k],
            )
            + X[k - 1]
            - X[k]
        )

    bvec[M] = (
        X[M - 1]
        - X[M]
    )

    L = Lambda.copy()

    for _ in range(sweeps):
        for k in range(
            M,
            -1,
            -1,
        ):
            if k == 0:
                hat = (
                    bvec[0]
                    + 2.0
                    * BETA_LAMBDA_SMOOTH
                    * L[1]
                ) / (
                    2.0
                    * BETA_LAMBDA_SMOOTH
                    + 2.0
                    * BETA_LAMBDA_L2
                )

            elif k == M:
                hat = (
                    bvec[M]
                    + 2.0
                    * BETA_LAMBDA_SMOOTH
                    * L[M - 1]
                ) / (
                    2.0
                    * BETA_LAMBDA_SMOOTH
                    + 2.0
                    * BETA_LAMBDA_L2
                )

            else:
                hat = (
                    bvec[k]
                    + 2.0
                    * BETA_LAMBDA_SMOOTH
                    * (
                        L[k - 1]
                        + L[k + 1]
                    )
                ) / (
                    4.0
                    * BETA_LAMBDA_SMOOTH
                    + 2.0
                    * BETA_LAMBDA_L2
                )

            L[k] = np.clip(
                hat,
                -LAMBDA_BOUND,
                LAMBDA_BOUND,
            )

    return L


# ============================================================
# 11. ACTION GRID HELPERS
# ============================================================

def joint_grid(q, lo=None, hi=None):
    if lo is None:
        lo = U_LO

    if hi is None:
        hi = U_HI

    v_vals = np.linspace(
        lo[0],
        hi[0],
        q,
    )

    u_vals = np.linspace(
        lo[1],
        hi[1],
        q,
    )

    V, U = np.meshgrid(
        v_vals,
        u_vals,
        indexing="ij",
    )

    return np.column_stack(
        [
            V.ravel(),
            U.ravel(),
        ]
    )


def initialize_solver(control_dt):
    M, times = make_grid(
        control_dt
    )

    # Use a modest interior initial pair, not the paper solution.
    Actions = np.tile(
        np.array([0.01, 0.01]),
        (M, 1),
    )

    X, interval_costs = rollout_with_costs(
        Actions,
        control_dt,
    )

    Lambda = np.zeros(
        (M + 1, STATE_DIM),
        dtype=float,
    )

    return (
        M,
        times,
        Actions,
        X,
        interval_costs,
        Lambda,
    )


def initialize_best(
    Actions,
    X,
    interval_costs,
    control_dt,
):
    J, _ = objective_discrete(
        Actions,
        control_dt,
        X,
        interval_costs,
    )

    return {
        "J": J,
        "Actions": Actions.copy(),
        "X": X.copy(),
        "interval_costs": interval_costs.copy(),
        "outer": 0,
    }


def update_best(
    best,
    Actions,
    X,
    interval_costs,
    control_dt,
    outer,
):
    J, _ = objective_discrete(
        Actions,
        control_dt,
        X,
        interval_costs,
    )

    if J < best["J"]:
        return {
            "J": J,
            "Actions": Actions.copy(),
            "X": X.copy(),
            "interval_costs": interval_costs.copy(),
            "outer": int(outer),
        }

    return best


# ============================================================
# 12. MAIN JOINT q x q HPDPO
# ============================================================

def solve_joint(
    control_dt,
    outer_iters=12,
    q=21,
):
    (
        M,
        times,
        Actions,
        X,
        interval_costs,
        Lambda,
    ) = initialize_solver(
        control_dt
    )

    candidates = joint_grid(q)

    evaluations = 0
    history = []

    best = initialize_best(
        Actions,
        X,
        interval_costs,
        control_dt,
    )

    tic = time.perf_counter()

    for outer in range(
        1,
        outer_iters + 1,
    ):
        A_before = Actions.copy()
        L_before = Lambda.copy()

        for k in range(M):
            (
                best_action,
                _,
                X,
                interval_costs,
                n_eval,
            ) = evaluate_candidates(
                Actions,
                X,
                interval_costs,
                Lambda,
                k,
                candidates,
                control_dt,
            )

            evaluations += n_eval
            Actions[k] = best_action

        Lambda_exact = update_lambda(
            Actions,
            Lambda,
            X,
            control_dt,
            sweeps=3,
        )

        Lambda = (
            (1.0 - LAMBDA_RELAX)
            * Lambda
            + LAMBDA_RELAX
            * Lambda_exact
        )

        Jd, X = objective_discrete(
            Actions,
            control_dt,
            X,
            interval_costs,
        )

        best = update_best(
            best,
            Actions,
            X,
            interval_costs,
            control_dt,
            outer,
        )

        history.append(
            [
                outer,
                Jd,
                best["J"],
                np.max(
                    np.abs(
                        Actions
                        - A_before
                    )
                ),
                np.max(
                    np.abs(
                        Lambda
                        - L_before
                    )
                ),
            ]
        )

    seconds = (
        time.perf_counter()
        - tic
    )

    return package_result(
        "joint",
        f"joint {q}x{q} global grid",
        times,
        best,
        Actions,
        X,
        interval_costs,
        Lambda,
        history,
        evaluations,
        seconds,
        control_dt,
    )


# ============================================================
# 13. JOINT GRID + LOCAL JOINT REFINEMENT
# ============================================================

def solve_joint_refined(
    control_dt,
    outer_iters=12,
    global_q=21,
    refine_q=11,
    refine_levels=2,
):
    (
        M,
        times,
        Actions,
        X,
        interval_costs,
        Lambda,
    ) = initialize_solver(
        control_dt
    )

    global_candidates = joint_grid(
        global_q
    )

    global_spacing = (
        U_HI - U_LO
    ) / (
        global_q - 1
    )

    evaluations = 0
    history = []

    best = initialize_best(
        Actions,
        X,
        interval_costs,
        control_dt,
    )

    tic = time.perf_counter()

    for outer in range(
        1,
        outer_iters + 1,
    ):
        A_before = Actions.copy()
        L_before = Lambda.copy()

        for k in range(M):
            (
                best_action,
                best_score,
                X,
                interval_costs,
                n_eval,
            ) = evaluate_candidates(
                Actions,
                X,
                interval_costs,
                Lambda,
                k,
                global_candidates,
                control_dt,
            )

            evaluations += n_eval
            Actions[k] = best_action

            half_width = global_spacing.copy()

            for _ in range(
                refine_levels
            ):
                lo = np.maximum(
                    U_LO,
                    best_action
                    - half_width,
                )

                hi = np.minimum(
                    U_HI,
                    best_action
                    + half_width,
                )

                local_candidates = joint_grid(
                    refine_q,
                    lo=lo,
                    hi=hi,
                )

                (
                    local_action,
                    local_score,
                    local_X,
                    local_interval_costs,
                    n_eval,
                ) = evaluate_candidates(
                    Actions,
                    X,
                    interval_costs,
                    Lambda,
                    k,
                    local_candidates,
                    control_dt,
                )

                evaluations += n_eval

                if (
                    local_score
                    <= best_score
                ):
                    best_action = (
                        local_action
                    )
                    best_score = (
                        local_score
                    )
                    Actions[k] = (
                        best_action
                    )
                    X = local_X
                    interval_costs = local_interval_costs

                half_width = (
                    np.maximum(
                        (
                            hi - lo
                        ) / (
                            refine_q - 1
                        ),
                        np.finfo(float).eps,
                    )
                )

        Lambda_exact = update_lambda(
            Actions,
            Lambda,
            X,
            control_dt,
            sweeps=3,
        )

        Lambda = (
            (1.0 - LAMBDA_RELAX)
            * Lambda
            + LAMBDA_RELAX
            * Lambda_exact
        )

        Jd, X = objective_discrete(
            Actions,
            control_dt,
            X,
            interval_costs,
        )

        best = update_best(
            best,
            Actions,
            X,
            interval_costs,
            control_dt,
            outer,
        )

        history.append(
            [
                outer,
                Jd,
                best["J"],
                np.max(
                    np.abs(
                        Actions
                        - A_before
                    )
                ),
                np.max(
                    np.abs(
                        Lambda
                        - L_before
                    )
                ),
            ]
        )

    seconds = (
        time.perf_counter()
        - tic
    )

    return package_result(
        "joint_refine",
        (
            f"joint {global_q}x{global_q}"
            f" + {refine_levels} local {refine_q}x{refine_q}"
        ),
        times,
        best,
        Actions,
        X,
        interval_costs,
        Lambda,
        history,
        evaluations,
        seconds,
        control_dt,
    )


# ============================================================
# 14. COARSE JOINT GRID + POWELL
# ============================================================

def solve_joint_powell(
    control_dt,
    outer_iters=12,
    coarse_q=5,
):
    (
        M,
        times,
        Actions,
        X,
        interval_costs,
        Lambda,
    ) = initialize_solver(
        control_dt
    )

    coarse_candidates = joint_grid(
        coarse_q
    )

    evaluations = 0
    history = []

    best = initialize_best(
        Actions,
        X,
        interval_costs,
        control_dt,
    )

    tic = time.perf_counter()

    for outer in range(
        1,
        outer_iters + 1,
    ):
        A_before = Actions.copy()
        L_before = Lambda.copy()

        for k in range(M):
            (
                coarse_action,
                coarse_score,
                X,
                interval_costs,
                n_eval,
            ) = evaluate_candidates(
                Actions,
                X,
                interval_costs,
                Lambda,
                k,
                coarse_candidates,
                control_dt,
            )

            evaluations += n_eval
            Actions[k] = coarse_action

            counter = [0]

            def score_fun(a):
                counter[0] += 1

                a = np.clip(
                    np.asarray(
                        a,
                        dtype=float,
                    ),
                    U_LO,
                    U_HI,
                )

                A_test = Actions.copy()
                A_test[k] = a

                X_test, C_test = reroll_suffix(
                    X,
                    interval_costs,
                    A_test,
                    k,
                    control_dt,
                )

                return L_reg(
                    A_test,
                    Lambda,
                    control_dt,
                    X_test,
                    C_test,
                )

            result = minimize(
                score_fun,
                x0=coarse_action,
                method="Powell",
                bounds=[
                    (
                        float(U_LO[0]),
                        float(U_HI[0]),
                    ),
                    (
                        float(U_LO[1]),
                        float(U_HI[1]),
                    ),
                ],
                options={
                    "xtol": 1e-8,
                    "ftol": 1e-10,
                    "maxiter": 80,
                    "disp": False,
                },
            )

            evaluations += counter[0]

            a_powell = np.clip(
                result.x,
                U_LO,
                U_HI,
            )

            A_test = Actions.copy()
            A_test[k] = a_powell

            X_test, C_test = reroll_suffix(
                X,
                interval_costs,
                A_test,
                k,
                control_dt,
            )

            powell_score = L_reg(
                A_test,
                Lambda,
                control_dt,
                X_test,
                C_test,
            )

            evaluations += 1

            if (
                powell_score
                <= coarse_score
            ):
                Actions = A_test
                X = X_test
                interval_costs = C_test

        Lambda_exact = update_lambda(
            Actions,
            Lambda,
            X,
            control_dt,
            sweeps=3,
        )

        Lambda = (
            (1.0 - LAMBDA_RELAX)
            * Lambda
            + LAMBDA_RELAX
            * Lambda_exact
        )

        Jd, X = objective_discrete(
            Actions,
            control_dt,
            X,
            interval_costs,
        )

        best = update_best(
            best,
            Actions,
            X,
            interval_costs,
            control_dt,
            outer,
        )

        history.append(
            [
                outer,
                Jd,
                best["J"],
                np.max(
                    np.abs(
                        Actions
                        - A_before
                    )
                ),
                np.max(
                    np.abs(
                        Lambda
                        - L_before
                    )
                ),
            ]
        )

    seconds = (
        time.perf_counter()
        - tic
    )

    return package_result(
        "joint_powell",
        f"joint {coarse_q}x{coarse_q} + Powell",
        times,
        best,
        Actions,
        X,
        interval_costs,
        Lambda,
        history,
        evaluations,
        seconds,
        control_dt,
    )


# ============================================================
# 15. RESULT PACKAGING AND PAPER COMPARISON
# ============================================================

def package_result(
    name,
    label,
    times,
    best,
    final_A,
    final_X,
    final_interval_costs,
    Lambda,
    history,
    evaluations,
    seconds,
    control_dt,
):
    # In this corrected version, the optimized objective and the reported
    # objective are exactly the same RK4-integrated quantity.
    best_report_J = float(
        best["J"]
    )

    d = dose(
        best["Actions"],
        control_dt,
    )

    final_J, _ = objective_discrete(
        final_A,
        control_dt,
        final_X,
        final_interval_costs,
    )

    return {
        "name": name,
        "label": label,
        "times": times,
        "best_A": best["Actions"],
        "best_X": best["X"],
        "best_interval_costs": best["interval_costs"],
        "best_J_discrete": best["J"],
        "best_J_report": best_report_J,
        "best_outer": best["outer"],
        "dose_v": float(d[0]),
        "dose_u": float(d[1]),
        "final_A": final_A,
        "final_X": final_X,
        "final_interval_costs": final_interval_costs,
        "final_J_discrete": final_J,
        "Lambda": Lambda,
        "history": np.asarray(history),
        "evaluations": evaluations,
        "seconds": seconds,
    }



def comparison_row(result):
    J = result["best_J_report"]

    return {
        "method": result["label"],
        "best_outer": result["best_outer"],
        "J_hpdpo": J,
        "J_paper": PAPER_J_OPT,
        "J_relative_difference_percent": (
            100.0
            * (
                J - PAPER_J_OPT
            )
            / PAPER_J_OPT
        ),
        "dose_v_hpdpo": result["dose_v"],
        "dose_v_paper": PAPER_DOSE_V_OPT,
        "dose_v_relative_difference_percent": (
            100.0
            * (
                result["dose_v"]
                - PAPER_DOSE_V_OPT
            )
            / PAPER_DOSE_V_OPT
        ),
        "dose_u_hpdpo": result["dose_u"],
        "dose_u_paper": PAPER_DOSE_U_OPT,
        "dose_u_relative_difference_percent": (
            100.0
            * (
                result["dose_u"]
                - PAPER_DOSE_U_OPT
            )
            / PAPER_DOSE_U_OPT
        ),
        "M_final": float(
            result["best_X"][-1, 0]
        ),
        "O_final": float(
            result["best_X"][-1, 4]
        ),
        "score_evaluations": result["evaluations"],
        "wall_time_seconds": result["seconds"],
    }


def print_result(row):
    print(
        "\n------------------------------------------------------------"
    )
    print(row["method"])
    print(
        "------------------------------------------------------------"
    )

    print(
        f"best outer iteration       = {row['best_outer']}"
    )

    print(
        f"HPDPO J                    = {row['J_hpdpo']:.10f}"
    )

    print(
        f"published J                = {row['J_paper']:.10f}"
    )

    print(
        f"relative J difference      = "
        f"{row['J_relative_difference_percent']:.6f}%"
    )

    print(
        f"HPDPO dose(v)              = {row['dose_v_hpdpo']:.10f}"
    )

    print(
        f"published dose(v)          = {row['dose_v_paper']:.10f}"
    )

    print(
        f"HPDPO dose(u)              = {row['dose_u_hpdpo']:.10f}"
    )

    print(
        f"published dose(u)          = {row['dose_u_paper']:.10f}"
    )

    print(
        f"M(T)                       = {row['M_final']:.8e}"
    )

    print(
        f"O(T)                       = {row['O_final']:.8e}"
    )

    print(
        f"score evaluations          = {row['score_evaluations']}"
    )

    print(
        f"wall time (s)              = {row['wall_time_seconds']:.4f}"
    )


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

def save_summary(rows, path):
    if not rows:
        return

    fields = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def save_trajectory(
    result,
    path,
):
    times = result["times"]
    X = result["best_X"]
    A = result["best_A"]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "time",
                "M",
                "U2",
                "U3",
                "U4",
                "O",
                "v",
                "u",
            ]
        )

        for k in range(
            A.shape[0]
        ):
            writer.writerow(
                [
                    times[k],
                    *X[k],
                    *A[k],
                ]
            )

        writer.writerow(
            [
                times[-1],
                *X[-1],
                "",
                "",
            ]
        )


# ============================================================
# 17. OPTIONAL PLOTS
# ============================================================

def make_plots(
    results,
    out_dir,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(
            "matplotlib unavailable; skipping plots."
        )
        return

    # v(t)
    plt.figure(
        figsize=(8, 4.5)
    )

    for r in results:
        plt.step(
            r["times"][:-1],
            r["best_A"][:, 0],
            where="post",
            label=r["name"],
        )

    plt.xlabel("time (s)")
    plt.ylabel("v(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "v_control.png",
        dpi=180,
    )
    plt.close()

    # u(t)
    plt.figure(
        figsize=(8, 4.5)
    )

    for r in results:
        plt.step(
            r["times"][:-1],
            r["best_A"][:, 1],
            where="post",
            label=r["name"],
        )

    plt.xlabel("time (s)")
    plt.ylabel("u(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "u_control.png",
        dpi=180,
    )
    plt.close()

    # M and O
    for idx, name in [
        (0, "M"),
        (4, "O"),
    ]:
        plt.figure(
            figsize=(8, 4.5)
        )

        for r in results:
            plt.plot(
                r["times"],
                r["best_X"][:, idx],
                label=r["name"],
            )

        plt.xlabel("time (s)")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"{name}_trajectory.png",
            dpi=180,
        )
        plt.close()


# ============================================================
# 18. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Published 2025 Alzheimer CMO benchmark "
            "with genuinely joint HPDPO action search."
        )
    )

    parser.add_argument(
        "--method",
        choices=[
            "joint",
            "joint_refine",
            "joint_powell",
            "all",
        ],
        default="joint_refine",
    )

    parser.add_argument(
        "--control_dt",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--outer_iters",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--q",
        type=int,
        default=21,
        help=(
            "Values per dimension for the main joint grid. "
            "q=21 gives 441 joint pairs/time point."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke test: control_dt=20 s, outer_iters=2, q=9."
        ),
    )

    parser.add_argument(
        "--out_dir",
        default="alzheimer_cmo_joint_hpdpo_results",
    )

    args = parser.parse_args()

    if args.quick:
        control_dt = 20.0
        outer_iters = 2
        q = 9
    else:
        control_dt = args.control_dt
        outer_iters = args.outer_iters
        q = args.q

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n============================================================"
    )
    print(
        "2025 JMB ALZHEIMER CMO BENCHMARK"
    )
    print(
        "5 states, 2 simultaneous treatment controls"
    )
    print(
        "============================================================"
    )

    print(
        f"control_dt              = {control_dt} s"
    )
    print(
        f"outer iterations        = {outer_iters}"
    )
    print(
        f"joint global grid       = {q} x {q} = {q*q}"
    )

    print(
        f"running-cost quadrature = RK4-integrated, max internal step "
        f"{MAX_INTERNAL_STEP} s"
    )
    print(
        "candidate objective     = same numerical objective as final report"
    )

    # --------------------------------------------------------
    # Baseline validation against Table 5
    # --------------------------------------------------------

    baseline = baseline_validation(
        control_dt
    )

    print(
        "\nPublished-baseline implementation check:"
    )

    print(
        f"no-control J, code      = {baseline['J_zero']:.10f}"
    )
    print(
        f"no-control J, paper     = {PAPER_J_NO_CONTROL:.10f}"
    )

    print(
        f"max-control J, code     = {baseline['J_max']:.10f}"
    )
    print(
        f"max-control J, paper    = {PAPER_J_MAX_CONTROL:.10f}"
    )

    print(
        f"max-control doses, code = {baseline['dose_max']}"
    )
    print(
        f"max-control dose, paper = {PAPER_DOSE_MAX}"
    )

    solvers = {
        "joint": lambda: (
            solve_joint(
                control_dt,
                outer_iters=outer_iters,
                q=q,
            )
        ),

        "joint_refine": lambda: (
            solve_joint_refined(
                control_dt,
                outer_iters=outer_iters,
                global_q=q,
                refine_q=11,
                refine_levels=2,
            )
        ),

        "joint_powell": lambda: (
            solve_joint_powell(
                control_dt,
                outer_iters=outer_iters,
                coarse_q=5,
            )
        ),
    }

    if args.method == "all":
        selected = [
            "joint",
            "joint_refine",
            "joint_powell",
        ]
    else:
        selected = [
            args.method
        ]

    results = []
    rows = []

    for key in selected:
        print(
            f"\nRunning: {key}"
        )

        result = (
            solvers[key]()
        )

        results.append(
            result
        )

        row = comparison_row(
            result
        )

        rows.append(
            row
        )

        print_result(
            row
        )

        np.savez(
            out_dir
            / f"{key}.npz",
            times=result["times"],
            best_A=result["best_A"],
            best_X=result["best_X"],
            best_interval_costs=result["best_interval_costs"],
            best_J_discrete=result["best_J_discrete"],
            best_J_report=result["best_J_report"],
            best_outer=result["best_outer"],
            dose_v=result["dose_v"],
            dose_u=result["dose_u"],
            final_A=result["final_A"],
            final_X=result["final_X"],
            final_interval_costs=result["final_interval_costs"],
            final_J_discrete=result["final_J_discrete"],
            Lambda=result["Lambda"],
            history=result["history"],
        )

        save_trajectory(
            result,
            out_dir
            / f"{key}_trajectory.csv",
        )

    save_summary(
        rows,
        out_dir
        / "summary.csv",
    )

    make_plots(
        results,
        out_dir,
    )

    print(
        "\n============================================================"
    )
    print("DONE")
    print(
        "============================================================"
    )
    print(
        f"Results saved to: "
        f"{out_dir.resolve()}"
    )




# ============================================================
# 19. MATCHED PPO / SAC COMPARISON FOR MULTIDIMENSIONAL EXAMPLE 2
# ============================================================
#
# This section adds standard continuous-control RL baselines to the same CMO
# problem.  All numerical methods use the SAME:
#   * physical dynamics and parameters,
#   * T = 1000 s,
#   * piecewise-constant policy grid (default control_dt = 5 s),
#   * RK4 state integration with internal substeps <= 0.5 s,
#   * control bounds [0, 0.03]^2,
#   * original CMO objective, including terminal cost.
#
# The published Bulai et al. paper reports objective values and treatment
# doses, and states that its simulations used MATLAB and a forward-backward
# sweep method.  It does not report wall-clock computation time or hardware,
# so the published column is left blank for timing.  HPDPO/PPO/SAC timings
# below are measured on the same machine during this run.

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
        "  pip install numpy scipy pandas matplotlib gymnasium stable-baselines3 torch\n"
    ) from exc


# -------------------------- comparison settings --------------------------
COMPARISON_CONTROL_DT = 5.0
COMPARISON_OUTER_ITERS = 12
COMPARISON_GLOBAL_Q = 21
COMPARISON_LOCAL_Q = 11
COMPARISON_LOCAL_LEVELS = 2

TOTAL_TIMESTEPS = 250_000
RL_SEED = 0
REWARD_SCALE = 1000.0
DEVICE = "cpu"

# State concentrations are around 1e-3 or smaller.  Multiplying them by 1000
# only rescales the neural-network input; it does not alter the physical model
# or the objective.
STATE_OBS_SCALE = 1000.0

# PPO settings follow the earlier paper baselines, adapted to the 200-step
# episode length of this CMO example.
PPO_N_STEPS = 4000
PPO_BATCH_SIZE = 200

# SAC settings follow the earlier paper baselines.
SAC_BUFFER_SIZE = 250_000
SAC_LEARNING_STARTS = 2000
SAC_BATCH_SIZE = 256

# Keep timing more interpretable and consistent on CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

COMPARISON_OUTPUT_DIR = Path("alzheimer_cmo_hpdpo_ppo_sac")
SAVE_MODELS = True
SAVE_COMPARISON_PLOTS = True


# -------------------------- RL environment -------------------------------
class CMOEnv(gym.Env):
    """Gymnasium environment for the 5-state, 2-control CMO benchmark."""

    metadata = {"render_modes": []}

    def __init__(self, control_dt=COMPARISON_CONTROL_DT):
        super().__init__()
        self.control_dt = float(control_dt)
        self.M, self.times = make_grid(self.control_dt)

        # Stable-Baselines3 chooses a normalized two-dimensional action.
        # It is mapped linearly to the physical treatment rates [0, 0.03]^2.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        # Observation = 1000*x plus normalized time k/M.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM + 1,),
            dtype=np.float32,
        )

        self.x = None
        self.k = None
        self.total_original_cost = None

    @staticmethod
    def normalized_to_physical(action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        # [-1,1] -> [0,0.03]
        return 0.5 * (a + 1.0) * (U_HI - U_LO) + U_LO

    def _obs(self):
        return np.r_[
            STATE_OBS_SCALE * self.x,
            self.k / self.M,
        ].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = X0.copy()
        self.k = 0
        self.total_original_cost = 0.0
        return self._obs(), {}

    def step(self, action):
        physical_action = self.normalized_to_physical(action)

        x_next, interval_cost = interval_step_and_cost(
            self.x,
            physical_action,
            self.control_dt,
        )

        self.x = x_next
        self.k += 1
        terminated = self.k >= self.M
        truncated = False

        # Exact decomposition of the SAME objective used for HPDPO/reporting:
        #     J = sum_k C_k + Phi(x_M).
        # Terminal cost is added on the final environment transition.
        step_objective_cost = float(interval_cost)
        if terminated:
            step_objective_cost += terminal_cost(self.x)

        self.total_original_cost += step_objective_cost
        reward = -REWARD_SCALE * step_objective_cost

        info = {
            "physical_action": physical_action.copy(),
            "interval_cost": float(interval_cost),
            "original_step_objective_cost": float(step_objective_cost),
            "objective_if_terminal": (
                float(self.total_original_cost) if terminated else None
            ),
        }

        return self._obs(), float(reward), terminated, truncated, info


# -------------------------- utilities ------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_rl_policy(model, control_dt=COMPARISON_CONTROL_DT):
    env = CMOEnv(control_dt=control_dt)
    obs, _ = env.reset(seed=RL_SEED)

    M = env.M
    Actions = np.empty((M, ACTION_DIM), dtype=float)
    X = np.empty((M + 1, STATE_DIM), dtype=float)
    X[0] = X0.copy()

    for k in range(M):
        action, _ = model.predict(obs, deterministic=True)
        physical = env.normalized_to_physical(action)
        Actions[k] = physical

        obs, _, terminated, truncated, _ = env.step(action)
        X[k + 1] = env.x.copy()

        if terminated or truncated:
            if k != M - 1:
                raise RuntimeError(
                    "RL episode terminated before the expected horizon."
                )
            break

    # Recompute the paper objective independently from the collected actions
    # using the exact same RK4-integrated objective function as HPDPO.
    J = objective_report(Actions, control_dt)
    d = dose(Actions, control_dt)

    return {
        "Actions": Actions,
        "X": X,
        "J": float(J),
        "dose_v": float(d[0]),
        "dose_w": float(d[1]),
    }


def train_ppo(control_dt=COMPARISON_CONTROL_DT):
    set_all_seeds(RL_SEED)
    env = CMOEnv(control_dt=control_dt)

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    tic = time.perf_counter()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
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

    result = evaluate_rl_policy(model, control_dt)
    result["seconds"] = float(seconds)
    return model, result


def train_sac(control_dt=COMPARISON_CONTROL_DT):
    set_all_seeds(RL_SEED)
    env = CMOEnv(control_dt=control_dt)

    policy_kwargs = dict(net_arch=[64, 64])

    tic = time.perf_counter()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=SAC_LEARNING_STARTS,
        batch_size=SAC_BATCH_SIZE,
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

    result = evaluate_rl_policy(model, control_dt)
    result["seconds"] = float(seconds)
    return model, result


# -------------------------- metrics / tables -----------------------------
def metrics_from_values(method, J, dose_v_value, dose_w_value, seconds):
    return {
        "Method": method,
        "Objective J": float(J),
        "Absolute objective difference": abs(float(J) - PAPER_J_OPT),
        "Relative objective difference (%)": (
            100.0 * abs(float(J) - PAPER_J_OPT) / abs(PAPER_J_OPT)
        ),
        "Dose v": float(dose_v_value),
        "Absolute dose-v difference": abs(float(dose_v_value) - PAPER_DOSE_V_OPT),
        "Relative dose-v difference (%)": (
            100.0 * abs(float(dose_v_value) - PAPER_DOSE_V_OPT)
            / abs(PAPER_DOSE_V_OPT)
        ),
        "Dose w": float(dose_w_value),
        "Absolute dose-w difference": abs(float(dose_w_value) - PAPER_DOSE_U_OPT),
        "Relative dose-w difference (%)": (
            100.0 * abs(float(dose_w_value) - PAPER_DOSE_U_OPT)
            / abs(PAPER_DOSE_U_OPT)
        ),
        "Computation time (s)": float(seconds),
    }


def publication_table_dataframe(hpdpo_result, ppo_result, sac_result):
    # Published CMO paper: objective and doses reported; runtime not reported.
    rows = [
        {
            "Method": "Published CMO",
            "Objective J": PAPER_J_OPT,
            "Relative objective difference (%)": 0.0,
            "Dose v": PAPER_DOSE_V_OPT,
            "Dose w": PAPER_DOSE_U_OPT,
            "Computation time (s)": np.nan,
        },
        {
            "Method": "HPDPO",
            "Objective J": hpdpo_result["best_J_report"],
            "Relative objective difference (%)": (
                100.0 * abs(hpdpo_result["best_J_report"] - PAPER_J_OPT)
                / PAPER_J_OPT
            ),
            "Dose v": hpdpo_result["dose_v"],
            "Dose w": hpdpo_result["dose_u"],
            "Computation time (s)": hpdpo_result["seconds"],
        },
        {
            "Method": "PPO",
            "Objective J": ppo_result["J"],
            "Relative objective difference (%)": (
                100.0 * abs(ppo_result["J"] - PAPER_J_OPT) / PAPER_J_OPT
            ),
            "Dose v": ppo_result["dose_v"],
            "Dose w": ppo_result["dose_w"],
            "Computation time (s)": ppo_result["seconds"],
        },
        {
            "Method": "SAC",
            "Objective J": sac_result["J"],
            "Relative objective difference (%)": (
                100.0 * abs(sac_result["J"] - PAPER_J_OPT) / PAPER_J_OPT
            ),
            "Dose v": sac_result["dose_v"],
            "Dose w": sac_result["dose_w"],
            "Computation time (s)": sac_result["seconds"],
        },
    ]
    return pd.DataFrame(rows)


def raw_metrics_dataframe(hpdpo_result, ppo_result, sac_result):
    rows = [
        metrics_from_values(
            "HPDPO",
            hpdpo_result["best_J_report"],
            hpdpo_result["dose_v"],
            hpdpo_result["dose_u"],
            hpdpo_result["seconds"],
        ),
        metrics_from_values(
            "PPO",
            ppo_result["J"],
            ppo_result["dose_v"],
            ppo_result["dose_w"],
            ppo_result["seconds"],
        ),
        metrics_from_values(
            "SAC",
            sac_result["J"],
            sac_result["dose_v"],
            sac_result["dose_w"],
            sac_result["seconds"],
        ),
    ]
    return pd.DataFrame(rows)


def print_comparison_table(df):
    print("\n" + "=" * 110)
    print("MULTIDIMENSIONAL EXAMPLE 2 — PUBLISHED CMO / HPDPO / PPO / SAC")
    print("=" * 110)

    display = df.copy()
    for col in display.columns:
        if col == "Method":
            continue
        if col == "Computation time (s)":
            display[col] = display[col].map(
                lambda z: "--" if pd.isna(z) else f"{z:.2f}"
            )
        elif col == "Relative objective difference (%)":
            display[col] = display[col].map(lambda z: f"{z:.6f}")
        else:
            display[col] = display[col].map(lambda z: f"{z:.10f}")

    print(display.to_string(index=False))
    print("=" * 110)
    print("Published CMO computation time: not reported in Bulai et al. (2025).")


def save_comparison_outputs(
    out_dir,
    hpdpo_result,
    ppo_result,
    sac_result,
    ppo_model,
    sac_model,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pub_df = publication_table_dataframe(hpdpo_result, ppo_result, sac_result)
    raw_df = raw_metrics_dataframe(hpdpo_result, ppo_result, sac_result)

    pub_df.to_csv(out_dir / "comparison_table.csv", index=False)
    raw_df.to_csv(out_dir / "comparison_table_raw.csv", index=False)

    # A LaTeX table with the concise quantities appropriate for the manuscript.
    latex_df = pub_df.copy()
    latex_df["Computation time (s)"] = latex_df["Computation time (s)"].apply(
        lambda z: "---" if pd.isna(z) else f"{z:.2f}"
    )
    latex_df["Objective J"] = latex_df["Objective J"].map(lambda z: f"{z:.10f}")
    latex_df["Relative objective difference (%)"] = latex_df[
        "Relative objective difference (%)"
    ].map(lambda z: f"{z:.6f}")
    latex_df["Dose v"] = latex_df["Dose v"].map(lambda z: f"{z:.6f}")
    latex_df["Dose w"] = latex_df["Dose w"].map(lambda z: f"{z:.6f}")

    with open(out_dir / "comparison_table.tex", "w", encoding="utf-8") as fobj:
        fobj.write(
            latex_df.to_latex(
                index=False,
                escape=False,
                column_format="lccccc",
            )
        )

    np.savez(
        out_dir / "comparison_results.npz",
        hpdpo_A=hpdpo_result["best_A"],
        hpdpo_X=hpdpo_result["best_X"],
        ppo_A=ppo_result["Actions"],
        ppo_X=ppo_result["X"],
        sac_A=sac_result["Actions"],
        sac_X=sac_result["X"],
        paper_J=PAPER_J_OPT,
        paper_dose_v=PAPER_DOSE_V_OPT,
        paper_dose_w=PAPER_DOSE_U_OPT,
    )

    # Save trajectories as CSV for plotting or manuscript inspection.
    M, times = make_grid(COMPARISON_CONTROL_DT)
    traj = pd.DataFrame({
        "time": times,
        "M_HPDPO": hpdpo_result["best_X"][:, 0],
        "U2_HPDPO": hpdpo_result["best_X"][:, 1],
        "U3_HPDPO": hpdpo_result["best_X"][:, 2],
        "U4_HPDPO": hpdpo_result["best_X"][:, 3],
        "O_HPDPO": hpdpo_result["best_X"][:, 4],
        "M_PPO": ppo_result["X"][:, 0],
        "U2_PPO": ppo_result["X"][:, 1],
        "U3_PPO": ppo_result["X"][:, 2],
        "U4_PPO": ppo_result["X"][:, 3],
        "O_PPO": ppo_result["X"][:, 4],
        "M_SAC": sac_result["X"][:, 0],
        "U2_SAC": sac_result["X"][:, 1],
        "U3_SAC": sac_result["X"][:, 2],
        "U4_SAC": sac_result["X"][:, 3],
        "O_SAC": sac_result["X"][:, 4],
    })
    traj.to_csv(out_dir / "state_trajectories.csv", index=False)

    ctrl_times = times[:-1]
    ctrl = pd.DataFrame({
        "time": ctrl_times,
        "v_HPDPO": hpdpo_result["best_A"][:, 0],
        "w_HPDPO": hpdpo_result["best_A"][:, 1],
        "v_PPO": ppo_result["Actions"][:, 0],
        "w_PPO": ppo_result["Actions"][:, 1],
        "v_SAC": sac_result["Actions"][:, 0],
        "w_SAC": sac_result["Actions"][:, 1],
    })
    ctrl.to_csv(out_dir / "control_trajectories.csv", index=False)

    if SAVE_MODELS:
        ppo_model.save(str(out_dir / "ppo_model"))
        sac_model.save(str(out_dir / "sac_model"))

    with open(out_dir / "run_metadata.txt", "w", encoding="utf-8") as fobj:
        fobj.write("Multidimensional Example 2: CMO HPDPO/PPO/SAC comparison\n")
        fobj.write(f"Python: {platform.python_version()}\n")
        fobj.write(f"Platform: {platform.platform()}\n")
        fobj.write(f"Processor: {platform.processor()}\n")
        fobj.write(f"NumPy: {np.__version__}\n")
        fobj.write(f"PyTorch: {torch.__version__}\n")
        fobj.write(f"Stable-Baselines3: {sb3.__version__}\n")
        fobj.write(f"Device: {DEVICE}\n")
        fobj.write(f"RL seed: {RL_SEED}\n")
        fobj.write(f"RL transitions per method: {TOTAL_TIMESTEPS}\n")
        fobj.write(f"Reward scale: {REWARD_SCALE}\n")
        fobj.write(f"State observation scale: {STATE_OBS_SCALE}\n")
        fobj.write(f"Control dt: {COMPARISON_CONTROL_DT}\n")
        fobj.write(f"Max internal RK4 step: {MAX_INTERNAL_STEP}\n")
        fobj.write(
            "Published CMO runtime/hardware: not reported in Bulai et al. (2025).\n"
        )

    if SAVE_COMPARISON_PLOTS:
        try:
            import matplotlib.pyplot as plt

            # Control comparison.
            plt.figure(figsize=(9, 5))
            plt.plot(ctrl_times, hpdpo_result["best_A"][:, 0], label="HPDPO v")
            plt.plot(ctrl_times, ppo_result["Actions"][:, 0], label="PPO v")
            plt.plot(ctrl_times, sac_result["Actions"][:, 0], label="SAC v")
            plt.plot(ctrl_times, hpdpo_result["best_A"][:, 1], "--", label="HPDPO w")
            plt.plot(ctrl_times, ppo_result["Actions"][:, 1], "--", label="PPO w")
            plt.plot(ctrl_times, sac_result["Actions"][:, 1], "--", label="SAC w")
            plt.xlabel("Time (s)")
            plt.ylabel("Treatment rate")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(out_dir / "control_comparison.png", dpi=200)
            plt.close()

            # M and O comparison (the two states entering the objective directly).
            plt.figure(figsize=(9, 5))
            plt.plot(times, hpdpo_result["best_X"][:, 0], label="HPDPO M")
            plt.plot(times, ppo_result["X"][:, 0], label="PPO M")
            plt.plot(times, sac_result["X"][:, 0], label="SAC M")
            plt.plot(times, hpdpo_result["best_X"][:, 4], "--", label="HPDPO O")
            plt.plot(times, ppo_result["X"][:, 4], "--", label="PPO O")
            plt.plot(times, sac_result["X"][:, 4], "--", label="SAC O")
            plt.xlabel("Time (s)")
            plt.ylabel("Concentration")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(out_dir / "state_comparison_M_O.png", dpi=200)
            plt.close()
        except Exception as exc:
            print(f"Plotting skipped: {exc}")

    return pub_df, raw_df


# -------------------------- complete experiment ---------------------------
def comparison_main():
    print("\n" + "=" * 78)
    print("MULTIDIMENSIONAL EXAMPLE 2: CMO — HPDPO / PPO / SAC")
    print("=" * 78)
    print(f"T                         = {TF} s")
    print(f"control_dt                = {COMPARISON_CONTROL_DT} s")
    print(f"policy intervals          = {int(TF / COMPARISON_CONTROL_DT)}")
    print(f"max internal RK4 step     = {MAX_INTERNAL_STEP} s")
    print(f"v,w bounds                = [0, 0.03]")
    print(f"RL transitions/method     = {TOTAL_TIMESTEPS}")
    print(f"RL seed                   = {RL_SEED}")
    print("Published CMO wall time   = not reported")
    print("=" * 78)

    # Confirm the two easy published baseline values before optimization.
    baseline = baseline_validation(COMPARISON_CONTROL_DT)
    print("\nPublished-baseline implementation check:")
    print(f"no-treatment J, code      = {baseline['J_zero']:.10f}")
    print(f"no-treatment J, paper     = {PAPER_J_NO_CONTROL:.10f}")
    print(f"maximum-treatment J, code = {baseline['J_max']:.10f}")
    print(f"maximum-treatment J,paper = {PAPER_J_MAX_CONTROL:.10f}")

    print("\nRunning HPDPO on CPU ...")
    hpdpo_result = solve_joint_refined(
        COMPARISON_CONTROL_DT,
        outer_iters=COMPARISON_OUTER_ITERS,
        global_q=COMPARISON_GLOBAL_Q,
        refine_q=COMPARISON_LOCAL_Q,
        refine_levels=COMPARISON_LOCAL_LEVELS,
    )

    print("\nRunning PPO on CPU ...")
    ppo_model, ppo_result = train_ppo(COMPARISON_CONTROL_DT)

    print("\nRunning SAC on CPU ...")
    sac_model, sac_result = train_sac(COMPARISON_CONTROL_DT)

    pub_df, raw_df = save_comparison_outputs(
        COMPARISON_OUTPUT_DIR,
        hpdpo_result,
        ppo_result,
        sac_result,
        ppo_model,
        sac_model,
    )

    print_comparison_table(pub_df)

    out = COMPARISON_OUTPUT_DIR
    print(f"\nSaved table: {out / 'comparison_table.csv'}")
    print(f"Saved LaTeX: {out / 'comparison_table.tex'}")
    print(f"Saved raw metrics: {out / 'comparison_table_raw.csv'}")
    print(f"Saved trajectories: {out / 'comparison_results.npz'}")
    print(f"Saved run metadata: {out / 'run_metadata.txt'}")


if __name__ == "__main__":
    comparison_main()
