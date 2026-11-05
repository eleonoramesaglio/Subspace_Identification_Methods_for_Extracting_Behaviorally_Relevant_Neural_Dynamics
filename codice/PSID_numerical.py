"""
PSID numerical simulations.

Shows:
  1. Eigenvalue recovery (single illustrative model)
  2. Convergence of eigenvalue identification error vs number of training samples
  3. Behavior decoding R2 across random models (fixed large N)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from scipy.linalg import block_diag
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import PSID
from PSID.evaluation import evalPrediction


# ═══════════════════════════════════════════════════════════════════
# Data generation  (from the PSID paper, Supplementary Note 8)
# ═══════════════════════════════════════════════════════════════════

def _sample_eigenvalues(nx, rng):
    """
    Sample nx eigenvalues uniformly from the unit disk, ensuring complex conjugate pairs for non-real eigenvalues.
    """
    units = []
    count = 0
    while count < nx:
        remaining = nx - count
        if remaining == 1:
            r     = rng.uniform(0, 1) ** 0.5
            theta = rng.uniform(0, 2 * np.pi)
            angle = 0.0 if np.cos(theta) >= 0 else np.pi
            units.append([complex(r * np.cos(angle), 0.0)])
            count += 1
        else:
            r     = rng.uniform(0, 1) ** 0.5
            theta = rng.uniform(0, 2 * np.pi)
            lam   = r * np.exp(1j * theta)
            if np.isclose(np.imag(lam), 0.0):
                units.append([lam.real + 0j])
                count += 1
            else:
                units.append([lam, np.conj(lam)])
                count += 2
    eigs = [e for unit in units for e in unit]
    return eigs, units


def _select_behaviorally_relevant(units, n1, rng):
    """
    Randomly select n1 eigenvalues as behaviorally relevant, ensuring they are grouped by unit (real or complex conjugate pairs).
    """
    units = list(units)
    rng.shuffle(units)
    relevant, irrelevant = [], []
    count = 0
    for unit in units:
        if count < n1:
            relevant.extend(unit)
            count += len(unit)
        else:
            irrelevant.extend(unit)
    return relevant + irrelevant, len(relevant)


def _eigs_to_real_block_diag(eigs):
    """
    Convert a list of eigenvalues to a real block diagonal matrix, with 1x1 blocks for real eigenvalues and 2x2 blocks for complex conjugate pairs.
    """
    used, blocks = [False] * len(eigs), []
    for i, lam in enumerate(eigs):
        if used[i]:
            continue
        if np.isreal(lam) and np.imag(lam) == 0.0:
            blocks.append(np.array([[np.real(lam)]]))
            used[i] = True
        else:
            for j in range(i + 1, len(eigs)):
                if not used[j] and np.isclose(eigs[j], np.conj(lam)):
                    a, b = np.real(lam), np.imag(lam)
                    blocks.append(np.array([[a, -b], [b, a]]))
                    used[i] = used[j] = True
                    break
    return block_diag(*blocks)


def _make_noise_covariance(nx, ny, rng):
    """
    Generate positive definite covariance matrices Q, R and cross-covariance S with random scaling.
    """
    Omega = rng.standard_normal((nx + ny, nx + ny))
    L     = Omega @ Omega.T
    a1, a2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
    scale = np.concatenate([10**a1 * np.ones(nx), 10**a2 * np.ones(ny)])
    D     = np.diag(scale)
    cov   = D @ L @ D
    return cov[:nx, :nx], cov[nx:, nx:], cov[:nx, nx:]


def _make_eps_model(nz, rng):
    """
    Create an epsilon model with random eigenvalues and covariance matrices.
    """
    nx_eps    = int(rng.integers(1, 11))
    _, units  = _sample_eigenvalues(nx_eps, rng)
    eigs      = [e for unit in units for e in unit]
    A_eps     = _eigs_to_real_block_diag(eigs)
    nx_actual = A_eps.shape[0]
    C_eps     = rng.standard_normal((nz, nx_actual))
    Q, R, S   = _make_noise_covariance(nx_actual, nz, rng)
    return dict(A=A_eps, C=C_eps, Q=Q, R=R, S=S)


def _simulate_autonomous(model, N, rng):
    """
    Simulate an autonomous system defined by the given model parameters for N time steps.
    """
    A, C  = model['A'], model['C']
    nx    = A.shape[0]
    nout  = C.shape[0]
    cov   = np.block([[model['Q'], model['S']], [model['S'].T, model['R']]])
    cov   = (cov + cov.T) / 2
    noise = rng.multivariate_normal(np.zeros(nx + nout), cov, size=N)
    w, v  = noise[:, :nx], noise[:, nx:]
    x     = np.zeros(nx)
    out   = np.zeros((N, nout))
    for k in range(N):
        out[k] = C @ x + v[k]
        x = A @ x + w[k]
    return out


def generate_psid_simulation(N=20_000, seed=None, i=5):
    """
    Generate a random PSID simulation dataset with specified parameters and random seed.

    Returns:
    - y: (N, ny) array of neural activity
    - z: (N, nz) array of behavior
    - nx: total latent state dimension
    - n1: behaviorally relevant latent state dimension
    - i: PSID horizon parameter (should be >= n1/nz and nx/ny for good identification)
    - true_params: dictionary of true model parameters for evaluation and plotting
    """
    rng = np.random.default_rng(seed)

    ny = int(rng.integers(5, 11))
    nz = int(rng.integers(5, 11))
    nx = int(rng.integers(1, 11))
    n1 = int(rng.integers(1, nx + 1))

    eigs, units     = _sample_eigenvalues(nx, rng)
    eigs, n1_actual = _select_behaviorally_relevant(units, n1, rng)
    n1 = n1_actual

    A_rel  = _eigs_to_real_block_diag(eigs[:n1])
    A_irr  = _eigs_to_real_block_diag(eigs[n1:]) if n1 < len(eigs) else np.zeros((0, 0))
    s1, s2 = A_rel.shape[0], A_irr.shape[0]
    nx, n1 = s1 + s2, s1

    A_21 = rng.standard_normal((s2, s1)) if s2 > 0 else np.zeros((s2, s1))
    A = np.zeros((nx, nx))
    A[:s1, :s1] = A_rel
    A[s1:, s1:] = A_irr
    A[s1:, :s1] = A_21

    Cy         = rng.standard_normal((ny, nx))
    Cz         = np.zeros((nz, nx))
    Cz[:, :n1] = rng.standard_normal((nz, n1))

    Q, R, S = _make_noise_covariance(nx, ny, rng)

    joint_cov = np.block([[Q, S], [S.T, R]])
    joint_cov = (joint_cov + joint_cov.T) / 2
    noise     = rng.multivariate_normal(np.zeros(nx + ny), joint_cov, size=N)
    w, v      = noise[:, :nx], noise[:, nx:]
    x_seq     = np.zeros((N, nx))
    y_seq     = np.zeros((N, ny))
    x = np.zeros(nx)
    for k in range(N):
        y_seq[k] = Cy @ x + v[k]
        x_seq[k] = x
        x = A @ x + w[k]

    eps_model = _make_eps_model(nz, rng)
    eps_seq   = _simulate_autonomous(eps_model, N, rng)
    Cz_scaled = Cz.copy()
    for d in range(nz):
        signal_std = np.std(x_seq @ Cz[d])
        eps_std    = np.std(eps_seq[:, d])
        if signal_std > 1e-12 and eps_std > 1e-12:
            target_snr = np.exp(rng.uniform(np.log(1), np.log(100)))
            Cz_scaled[d] *= target_snr / (signal_std / eps_std)

    z_seq = x_seq @ Cz_scaled.T + eps_seq

    y = y_seq - y_seq.mean(axis=0)
    z = z_seq - z_seq.mean(axis=0)

    i = max(i, int(np.ceil(n1 / nz)), int(np.ceil(nx / ny)))

    true_params = dict(A=A, Cy=Cy, Cz=Cz_scaled, Q=Q, R=R, S=S,
                       ny=ny, nz=nz, nx=nx, n1=n1)
    return y, z, nx, n1, i, true_params


# ═══════════════════════════════════════════════════════════════════
# Identification error
# ═══════════════════════════════════════════════════════════════════

def compute_eigenvalue_error(true_params, idSys):
    """
    Compute normalized mean absolute error between true and identified eigenvalues, matching them by closest distance in the complex plane.
    """
    n1          = true_params['n1']
    A_true      = true_params['A']
    true_eigs   = np.linalg.eigvals(A_true[:n1, :n1])
    id_eigs     = np.linalg.eigvals(idSys.A[:n1, :n1])
    cost        = np.abs(true_eigs[:, None] - id_eigs[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    true_matched = true_eigs[row_ind]
    id_matched   = id_eigs[col_ind]
    denom = np.mean(np.abs(true_matched))
    return (np.mean(np.abs(true_matched - id_matched)) / denom
            if denom > 1e-12 else np.nan)



# ════════════════════════════════════════════════════════════════════════════════════
# Plot 1 — eigenvalue recovery (single illustrative model or grid for multiple models)
# ════════════════════════════════════════════════════════════════════════════════════

def _draw_eigenvalue_errors(ax, true_eigs, id_eigs, color):
    """
    Draw lines connecting true to identified eigenvalues (error visualization), matching them by closest distance in the complex plane.
    """
    cost = np.abs(true_eigs[:, None] - id_eigs[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    for t, d in zip(true_eigs[row_ind], id_eigs[col_ind]):
        ax.plot([np.real(t), np.real(d)],
                [np.imag(t), np.imag(d)],
                color=color, linewidth=1.2, alpha=0.7)

def plot_eigenvalues(true_params, idSys):
    """
    Plot true and identified eigenvalues in the complex plane, with error lines connecting true to identified relevant eigenvalues.
    """
    n1 = true_params['n1']
    nx = true_params['nx']
    A  = true_params['A']

    trueEigsRelevant   = np.linalg.eigvals(A[:n1, :n1])
    trueEigsIrrelevant = np.linalg.eigvals(A[n1:, n1:]) if nx > n1 else np.array([])
    all_idEigs         = np.linalg.eigvals(idSys.A)
    idEigsRelevant     = np.linalg.eigvals(idSys.A[:n1, :n1])
    idEigsIrrelevant   = np.linalg.eigvals(idSys.A[n1:, n1:]) if nx > n1 else np.array([])

    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2)
    axs[1].remove()
    ax  = axs[0]
    ax.axis("equal")
    ax.add_patch(patches.Circle((0, 0), radius=1, fill=False, color="black", alpha=0.2))
    ax.plot([-1, 1, 0, 0, 0], [0, 0, 0, -1, 1], color="black", alpha=0.2)

    if len(trueEigsIrrelevant):
        ax.scatter(np.real(trueEigsIrrelevant), np.imag(trueEigsIrrelevant),
                   marker="o", edgecolors="#dd0000", facecolors="none",
                   label="Behaviorally irrelevant")
    ax.scatter(np.real(trueEigsRelevant), np.imag(trueEigsRelevant),
               marker="o", edgecolors="#00dd00", facecolors="none",
               label="Behaviorally relevant")

    # Error lines: match true relevant to closest identified (full set)
    # _draw_eigenvalue_errors(ax, trueEigsRelevant, all_idEigs, "#9eef9e")
    _draw_eigenvalue_errors(ax, trueEigsRelevant, idEigsRelevant, "#9eef9e")

    ax.scatter(np.real(idEigsRelevant), np.imag(idEigsRelevant),
               marker="x", facecolors="#00cc00",
               label="PSID identified (stage 1)")
    if len(idEigsIrrelevant):
        ax.scatter(np.real(idEigsIrrelevant), np.imag(idEigsIrrelevant),
                   marker="x", facecolors="#cc0000",
                   label="PSID identified (stage 2)")

    ax.set_title("True and identified eigenvalues")
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_eigenvalues_grid(results, nrows=2, ncols=3):
    """
    Plot a grid of eigenvalue recovery plots for a subset of models.
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, res in enumerate(results[:nrows * ncols]):
        ax    = axes[idx]
        true  = res['true_params']
        idSys = res['idSys']
        nx    = res['nx']
        n1    = res['n1']
        err   = res['error']
        A     = true['A']

        ax.set_aspect("equal")
        ax.add_patch(patches.Circle((0, 0), radius=1, fill=False,
                                    color="black", alpha=0.2))
        ax.axhline(0, color="black", alpha=0.2, linewidth=0.8)
        ax.axvline(0, color="black", alpha=0.2, linewidth=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

        trueR = np.linalg.eigvals(A[:n1, :n1])
        trueI = np.linalg.eigvals(A[n1:, n1:]) if nx > n1 else np.array([])
        idR   = np.linalg.eigvals(idSys.A[:n1, :n1])
        idI   = np.linalg.eigvals(idSys.A[n1:, n1:]) if nx > n1 else np.array([])

        if len(trueI):
            ax.scatter(np.real(trueI), np.imag(trueI),
                       marker="o", edgecolors="#dd0000", facecolors="none")
        ax.scatter(np.real(trueR), np.imag(trueR),
                   marker="o", edgecolors="#00aa00", facecolors="none")

        _draw_eigenvalue_errors(ax, trueR, idR, "#9eef9e")

        ax.scatter(np.real(idR), np.imag(idR),
                   marker="x", color="#00cc00")
        if len(idI):
            ax.scatter(np.real(idI), np.imag(idI),
                       marker="x", color="#cc0000")

        ax.set_title(
            f"Model {res['seed'] + 1}",
            fontsize=10
        )

    legend_elements = [
    plt.Line2D([0], [0], marker="o", color='w', markeredgecolor="#00aa00",
               markerfacecolor="none", markersize=8, linestyle="None",
               label="Behaviorally relevant"),
    plt.Line2D([0], [0], marker="o", color='w', markeredgecolor="#dd0000",
               markerfacecolor="none", markersize=8, linestyle="None",
               label="Behaviorally irrelevant"),
    plt.Line2D([0], [0], marker="x", color="#00cc00", linestyle="None",
               markersize=8, label="PSID identified (stage 1)"),
    plt.Line2D([0], [0], marker="x", color="#cc0000", linestyle="None",
               markersize=8, label="PSID identified (stage 2)"),
    ]

    
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, 0.02))
    plt.suptitle("Eigenvalue recovery — PSID", fontsize=13)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# Plot 2 — convergence of eigenvalue error vs training samples
# ═══════════════════════════════════════════════════════════════════

def plot_convergence(N_values, mean_errors, sem_errors):
    """
    Plot convergence of eigenvalue identification error vs number of training samples, with error bars.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(N_values, mean_errors, color="#5cb85c", marker="o", label="PSID")
    ax.fill_between(N_values,
                    np.array(mean_errors) - np.array(sem_errors),
                    np.array(mean_errors) + np.array(sem_errors),
                    alpha=0.3, color="#5cb85c")
    ax.set_xscale("log")

    # Linear y-axis with ticks every 0.1 between 0 and 1
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Explicit ticks for each tested N value on x-axis
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values], rotation=45, ha='right', fontsize=9)

    ax.grid(alpha=0.3, which="major", linestyle="--")
    ax.minorticks_off()

    ax.set_xlabel("Number of training samples", fontsize=11)
    ax.set_ylabel("Normalized eigenvalue error", fontsize=11)
    ax.set_title("Eigenvalue identification error vs training samples — PSID", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# Plot 3 — behavior decoding R2 across models
# ═══════════════════════════════════════════════════════════════════

def plot_decoding_r2(r2_full):
    """
    Plot behavior decoding R2 across random models as a bar plot, with mean R2 indicated by a dashed line.
    """
    x = np.arange(len(r2_full))
    fig, ax = plt.subplots(figsize=(max(6, len(x) * 0.8), 4))
    ax.bar(x, r2_full, color="#5cb85c", alpha=0.8, label="PSID")
    ax.axhline(np.mean(r2_full), color="black", linestyle="--", alpha=0.6,
               label=f"Mean R2={np.mean(r2_full):.2f}")
    ax.set_xlabel("Model index", fontsize=11)
    ax.set_ylabel("Behavior decoding R2", fontsize=11)
    ax.set_title("Behavior decoding across random models — PSID", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i+1}" for i in x])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    N_MODELS   = 20
    N_LARGE    = 50_000                                     # for plots 1 and 3
    N_VALUES   = [500, 1000, 2000, 5000, 10000, 50000]      # for plot 2
    i_horizon  = 5
    '''

    # ── Plot 1: single illustrative model at large N ──────────────────────
    print("── Plot 1: eigenvalue recovery ──")
    y, z, nx, n1, i, true = generate_psid_simulation(N=N_LARGE, seed=0, i=i_horizon)
    half = N_LARGE // 2
    idSys_illustrative = PSID.PSID(y[:half], z[:half], nx=nx, n1=n1, i=i)
    plot_eigenvalues(true, idSys_illustrative)
    '''

    # ── Plot 2: convergence across N_MODELS models and N_VALUES samples ───
    print("\n── Plot 2: convergence ──")
    mean_errors, sem_errors = [], []

    for N in N_VALUES:
        print(f"   N={N}")
        errors = []
        for seed in range(N_MODELS):
            # Generate with large N, then take first N samples for training
            y, z, nx, n1, i, true = generate_psid_simulation(
                N=N_LARGE, seed=seed, i=i_horizon
            )
            yTrain, zTrain = y[:N], z[:N]
            yTest,  zTest  = y[N_LARGE//2:], z[N_LARGE//2:]
            try:
                idSys = PSID.PSID(yTrain, zTrain, nx=nx, n1=n1, i=i)
                err   = compute_eigenvalue_error(true, idSys)
                if not np.isnan(err):
                    errors.append(err)
            except Exception:
                pass   # skip models where PSID fails at very small N

        mean_errors.append(np.mean(errors))
        sem_errors.append(np.std(errors) / np.sqrt(len(errors)))
        print(f"   mean error={mean_errors[-1]:.3f}, n={len(errors)}")

    plot_convergence(N_VALUES, mean_errors, sem_errors)

    # ── Plot 1 grid and Plot 3:
    # ── Eigenvalue recovery grid and behavior decoding R2 across models ──────────
    print("\n── Eigenvalue recovery and behavior decoding ──")
    print(f"\n{'Model':<10} {'nx':<6} {'n1':<6} {'E_lambda':<12} {'R2':<8}")
    print("-" * 44)

    r2_all = []
    results_exp1 = []

    for seed in range(N_MODELS):
        y, z, nx, n1, i, true = generate_psid_simulation(
            N=N_LARGE, seed=seed, i=i_horizon
        )
        half          = N_LARGE // 2
        yTrain, yTest = y[:half], y[half:]
        zTrain, zTest = z[:half], z[half:]
        idSys       = PSID.PSID(yTrain, zTrain, nx=nx, n1=n1, i=i)
        zPred, _, _ = idSys.predict(yTest)
        r2          = np.mean(evalPrediction(zTest, zPred, "R2"))
        err         = compute_eigenvalue_error(true, idSys)

        print(f"M{seed+1:<9} {nx:<6} {n1:<6} {err:<12.4f} {r2:<8.3f}")

        r2_all.append(r2)
        results_exp1.append(dict(seed=seed, nx=nx, n1=n1,
                                true_params=true, idSys=idSys, error=err))
        
        errors_all = [res['error'] for res in results_exp1]

    # Remove any nan values before computing correlation
    valid = [(e, r) for e, r in zip(errors_all, r2_all) if not np.isnan(e)]
    errors_clean = [v[0] for v in valid]
    r2_clean     = [v[1] for v in valid]

    corr, pval = pearsonr(errors_clean, r2_clean)
    print(f"\nCorrelation between eigenvalue error and R2:")
    print(f"  Pearson r = {corr:.3f}, p = {pval:.3e}, n = {len(valid)}")
        
    plot_eigenvalues_grid(results_exp1, nrows=5, ncols=4)
    plot_decoding_r2(r2_all)