"""
IPSID numerical simulations.

Shows:
  1. Eigenvalue recovery (single illustrative model)
  2. Convergence of eigenvalue identification error vs number of training samples
  3. Behavior decoding R2 across random models (fixed large N)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.linalg import block_diag
from scipy.stats import pearsonr
import PSID
from PSID.evaluation import evalPrediction
from scipy.optimize import linear_sum_assignment

from PSID_IPSID_comparison import plot_eigenvalues_grid_comparison


# ═══════════════════════════════════════════════════════════════════
# Data generation  (from the IPSID paper, S1 Appendix - Methods)
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
    Convert a list of eigenvalues to a real block diagonal matrix, with 1x1 blocks for real eigenvalues and 2x2 blocks for complex conjugate pairs."""
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


def _make_input_model(nu, rng):
    """
    Autonomous model to generate structured input u_k.
    """
    nx_u      = int(rng.integers(1, 5))
    _, units  = _sample_eigenvalues(nx_u, rng)
    eigs      = [e for unit in units for e in unit]
    A_u       = _eigs_to_real_block_diag(eigs)
    nx_actual = A_u.shape[0]
    C_u       = rng.standard_normal((nu, nx_actual))
    Q, R, S   = _make_noise_covariance(nx_actual, nu, rng)
    return dict(A=A_u, C=C_u, Q=Q, R=R, S=S)


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


def generate_ipsid_simulation(N=20_000, seed=None, i=5):
    """
    Generate a random IPSID simulation dataset with specified parameters and random seed.

    Returns:
    - y: (N, ny) array of neural activity
    - z: (N, nz) array of behavior
    - u: (N, nu) array of inputs
    - nx: total latent state dimension
    - n1: behaviorally relevant latent state dimension
    - i: PSID horizon parameter (should be >= n1/nz and nx/ny for good identification)
    - true_params: dictionary of true model parameters for evaluation and plotting
    - input_model: the autonomous model used to generate the input sequence (for eigenvalue comparison)
    """
    rng = np.random.default_rng(seed)

    ny = int(rng.integers(5, 11))
    nz = int(rng.integers(5, 11))
    nu = int(rng.integers(1, 5))
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
    B          = rng.standard_normal((nx, nu))
    Dy         = rng.standard_normal((ny, nu))

    Q, R, S = _make_noise_covariance(nx, ny, rng)

    input_model = _make_input_model(nu, rng)
    u_seq       = _simulate_autonomous(input_model, N, rng)

    joint_cov = np.block([[Q, S], [S.T, R]])
    joint_cov = (joint_cov + joint_cov.T) / 2
    noise     = rng.multivariate_normal(np.zeros(nx + ny), joint_cov, size=N)
    w, v      = noise[:, :nx], noise[:, nx:]
    x_seq     = np.zeros((N, nx))
    y_seq     = np.zeros((N, ny))
    x = np.zeros(nx)
    for k in range(N):
        y_seq[k] = Cy @ x + Dy @ u_seq[k] + v[k]
        x_seq[k] = x
        x = A @ x + B @ u_seq[k] + w[k]

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
    u = u_seq - u_seq.mean(axis=0)

    i = max(i, int(np.ceil(n1 / nz)), int(np.ceil(nx / ny)))

    true_params = dict(A=A, B=B, Cy=Cy, Dy=Dy, Cz=Cz_scaled,
                       Q=Q, R=R, S=S, ny=ny, nz=nz, nu=nu, nx=nx, n1=n1)
    return y, z, u, nx, n1, i, true_params, input_model


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


# ═══════════════════════════════════════════════════════════════════
# Plot 1 — eigenvalue recovery (single illustrative model)
# ═══════════════════════════════════════════════════════════════════


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
        

def plot_eigenvalues_ipsid(true_params, idSys_ipsid, input_model):
    """
    Plot true and identified eigenvalues in the complex plane, with error lines connecting true to identified relevant eigenvalues.
    """
    n1 = true_params['n1']
    nx = true_params['nx']
    A  = true_params['A']

    trueEigsRelevant   = np.linalg.eigvals(A[:n1, :n1])
    trueEigsIrrelevant = np.linalg.eigvals(A[n1:, n1:]) if nx > n1 else np.array([])
    inputEigs          = np.linalg.eigvals(input_model['A'])
    all_idEigs         = np.linalg.eigvals(idSys_ipsid.A)
    idEigsRelevant     = np.linalg.eigvals(idSys_ipsid.A[:n1, :n1])
    idEigsIrrelevant   = np.linalg.eigvals(idSys_ipsid.A[n1:, n1:]) if nx > n1 else np.array([])

    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2)
    axs[1].remove()
    ax  = axs[0]
    ax.axis("equal")
    ax.add_patch(patches.Circle((0, 0), radius=1, fill=False, color="black", alpha=0.2))
    ax.plot([-1, 1, 0, 0, 0], [0, 0, 0, -1, 1], color="black", alpha=0.2)

    ax.scatter(np.real(inputEigs), np.imag(inputEigs),
               marker="o", edgecolors="#0000ff", facecolors="none",
               label="Input eigenvalues")
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
               marker="x", color="#00cc00",
               label="IPSID identified (stage 1)")
    if len(idEigsIrrelevant):
        ax.scatter(np.real(idEigsIrrelevant), np.imag(idEigsIrrelevant),
                   marker="x", color="#cc0000",
                   label="IPSID identified (stage 2)")

    ax.set_title("True and identified eigenvalues — IPSID")
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_eigenvalues_grid(results, nrows=2, ncols=3, title="Eigenvalue recovery"):
    """
    Plot a grid of eigenvalue recovery plots for a subset of models.
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, res in enumerate(results[:nrows * ncols]):
        ax         = axes[idx]
        true       = res['true_params']
        idSys      = res['idSys']
        nx         = res['nx']
        n1         = res['n1']
        err        = res['error']
        A          = true['A']
        input_eigs = res.get('input_eigs', None)  # None for PSID, set for IPSID

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

        if input_eigs is not None:
            ax.scatter(np.real(input_eigs), np.imag(input_eigs),
                       marker="o", edgecolors="#0000ff", facecolors="none")
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

        ax.set_title(f"Model {res['seed'] + 1}", fontsize=10)

    has_input = any(res.get('input_eigs') is not None for res in results)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color='w', markeredgecolor="#00aa00",
                   markerfacecolor="none", markersize=8, linestyle="None",
                   label="Behaviorally relevant"),
        plt.Line2D([0], [0], marker="o", color='w', markeredgecolor="#dd0000",
                   markerfacecolor="none", markersize=8, linestyle="None",
                   label="Behaviorally irrelevant"),
        plt.Line2D([0], [0], marker="x", color="#00cc00", linestyle="None",
                   markersize=8, label="Identified (stage 1)"),
        plt.Line2D([0], [0], marker="x", color="#cc0000", linestyle="None",
                   markersize=8, label="Identified (stage 2)"),
    ]

    if has_input:
        legend_elements.insert(2,
            plt.Line2D([0], [0], marker="o", color='w', markeredgecolor="#0000ff",
                       markerfacecolor="none", markersize=8, linestyle="None",
                       label="Input eigenvalues"),
        )

    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.02))
    plt.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.16 if has_input else 0.12, 1, 0.95])
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# Plot 2 — convergence of eigenvalue error vs training samples
# ═══════════════════════════════════════════════════════════════════

def plot_convergence_ipsid(N_values, mean_errors, sem_errors):
    """
    Plot convergence of eigenvalue identification error vs number of training samples, with error bars.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(N_values, mean_errors, color="#5c8fe0", marker="o", label="IPSID")
    ax.fill_between(N_values,
                    np.array(mean_errors) - np.array(sem_errors),
                    np.array(mean_errors) + np.array(sem_errors),
                    alpha=0.3, color="#5c8fe0")

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values], rotation=45, ha='right', fontsize=9)
    ax.grid(alpha=0.3, which="major", linestyle="--")
    ax.minorticks_off()
    ax.set_xlabel("Number of training samples", fontsize=11)
    ax.set_ylabel("Normalized eigenvalue error", fontsize=11)
    ax.set_title("Eigenvalue identification error vs training samples — IPSID", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# Plot 3 — behavior decoding R2 across models
# ═══════════════════════════════════════════════════════════════════

def plot_decoding_r2_ipsid(r2_all):
    """
    Plot behavior decoding R2 across random models as a bar plot, with mean R2 indicated by a dashed line.
    """
    x = np.arange(len(r2_all))

    fig, ax = plt.subplots(figsize=(max(6, len(x) * 0.8), 4))
    ax.bar(x, r2_all, color="#5c8fe0", alpha=0.8, label="IPSID")
    ax.axhline(np.mean(r2_all), color="black", linestyle="--", alpha=0.6,
               label=f"Mean R2={np.mean(r2_all):.2f}")
    ax.set_xlabel("Model index", fontsize=11)
    ax.set_ylabel("Behavior decoding R2", fontsize=11)
    ax.set_title("Behavior decoding across random models — IPSID", fontsize=12)
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

    N_MODELS  = 20
    N_LARGE   = 50_000
    N_VALUES  = [500, 1000, 2000, 5000, 10000, 50000]
    i_horizon = 5

    '''
    # ── Plot 1: single illustrative model ────────────────────────────────
    print("── Plot 1: eigenvalue recovery ──")
    y, z, u, nx, n1, i, true, input_model = generate_ipsid_simulation(
        N=N_LARGE, seed=11, i=i_horizon
    )
    half        = N_LARGE // 2
    idSys_ipsid = PSID.IPSID(y[:half], z[:half], u[:half], nx=nx, n1=n1, i=i)
    plot_eigenvalues_ipsid(true, idSys_ipsid, input_model)

    '''
    # ── Plot 2: convergence ───────────────────────────────────────────────
    print("\n── Plot 2: convergence ──")
    mean_errors, sem_errors = [], []

    for N in N_VALUES:
        print(f"   N={N}")
        errors = []
        for seed in range(N_MODELS):
            y, z, u, nx, n1, i, true, input_model = generate_ipsid_simulation(
                N=N_LARGE, seed=seed, i=i_horizon
            )
            yTrain, zTrain, uTrain = y[:N], z[:N], u[:N]
            try:
                idSys = PSID.IPSID(yTrain, zTrain, uTrain, nx=nx, n1=n1, i=i)
                err   = compute_eigenvalue_error(true, idSys)
                if not np.isnan(err):
                    errors.append(err)
            except Exception:
                pass

        mean_errors.append(np.mean(errors))
        sem_errors.append(np.std(errors) / np.sqrt(len(errors)))
        print(f"   mean error={mean_errors[-1]:.3f}, n={len(errors)}")

    plot_convergence_ipsid(N_VALUES, mean_errors, sem_errors)
    
    
    # ── Plot 1 grid and Plot 3:
    # ── Eigenvalue recovery grid and behavior decoding R2 across models ──────────
    print("\n── Eigenvalue recovery and behavior decoding ──")
    print(f"\n{'Model':<10} {'nx':<6} {'n1':<6} {'E_lambda':<12} {'R2':<8}")
    print("-" * 44)

    r2_all        = []
    results_ipsid = []

    for seed in range(N_MODELS):
        y, z, u, nx, n1, i, true, input_model = generate_ipsid_simulation(
            N=N_LARGE, seed=seed, i=i_horizon
        )
        half          = N_LARGE // 2
        yTrain, yTest = y[:half], y[half:]
        zTrain, zTest = z[:half], z[half:]
        uTrain, uTest = u[:half], u[half:]

        idSys       = PSID.IPSID(yTrain, zTrain, uTrain, nx=nx, n1=n1, i=i)
        zPred, _, _ = idSys.predict(yTest, uTest)
        r2          = np.mean(evalPrediction(zTest, zPred, "R2"))
        err         = compute_eigenvalue_error(true, idSys)

        print(f"M{seed+1:<9} {nx:<6} {n1:<6} {err:<12.4f} {r2:<8.3f}")

        r2_all.append(r2)
        input_eigs = np.linalg.eigvals(input_model['A'])
        results_ipsid.append(dict(seed=seed, nx=nx, n1=n1,
                               true_params=true, idSys=idSys, error=err,
                               input_eigs=np.linalg.eigvals(input_model['A'])))
        
        errors_all = [res['error'] for res in results_ipsid]

    plot_eigenvalues_grid(results_ipsid)
    plot_decoding_r2_ipsid(r2_all)

    # Remove any nan values before computing correlation
    valid = [(e, r) for e, r in zip(errors_all, r2_all) if not np.isnan(e)]
    errors_clean = [v[0] for v in valid]
    r2_clean     = [v[1] for v in valid]

    corr, pval = pearsonr(errors_clean, r2_clean)
    print(f"\nCorrelation between eigenvalue error and R2:")
    print(f"  Pearson r = {corr:.3f}, p = {pval:.3e}, n = {len(valid)}")
        
    plot_eigenvalues_grid(results_ipsid, nrows=2, ncols=3)
    # plot_decoding_r2(r2_all)
        
    '''
    ##### New

    # ── Pregenerate all models ────────────────────────────────────────────
    print("\n── Pregenerating models ──")
    all_models = []
    for seed in range(N_MODELS):
        y, z, u, nx, n1, i, true, input_model = generate_ipsid_simulation(
            N=N_LARGE, seed=seed, i=i_horizon
        )
        all_models.append(dict(y=y, z=z, u=u, nx=nx, n1=n1, i=i,
                            true=true, input_model=input_model,
                            seed=seed))

    # ── Plot 3: behavior decoding R2 + eigenvalue grid comparison ────────
    print("\n── Plot 3: behavior decoding ──")
    r2_psid_all, r2_ipsid_all = [], []
    results_comparison = []

    for model in all_models:
        y, z, u, nx, n1, i, true, input_model = (
            model['y'], model['z'], model['u'], model['nx'],
            model['n1'], model['i'], model['true'], model['input_model']
        )
        half          = N_LARGE // 2
        yTrain, yTest = y[:half], y[half:]
        zTrain, zTest = z[:half], z[half:]
        uTrain, uTest = u[:half], u[half:]

        idSys_psid  = PSID.PSID( yTrain, zTrain,        nx=nx, n1=n1, i=i)
        idSys_ipsid = PSID.IPSID(yTrain, zTrain, uTrain, nx=nx, n1=n1, i=i)

        zPred_psid,  _, _ = idSys_psid.predict(yTest)
        zPred_ipsid, _, _ = idSys_ipsid.predict(yTest, uTest)

        r2_psid  = np.mean(evalPrediction(zTest, zPred_psid,  "R2"))
        r2_ipsid = np.mean(evalPrediction(zTest, zPred_ipsid, "R2"))

        r2_psid_all.append(r2_psid)
        r2_ipsid_all.append(r2_ipsid)

        err_psid  = compute_eigenvalue_error(true, idSys_psid)
        err_ipsid = compute_eigenvalue_error(true, idSys_ipsid)

        print(f"   Model {model['true']['nx']}: "
              f"PSID R2={r2_psid:.3f} E={err_psid:.3f}, "
              f"IPSID R2={r2_ipsid:.3f} E={err_ipsid:.3f}")

        results_comparison.append(dict(
            seed=model['seed'], nx=nx, n1=n1,
            true_params=true,
            idSys_psid=idSys_psid, idSys_ipsid=idSys_ipsid,
            error_psid=err_psid,   error_ipsid=err_ipsid,
            input_eigs=np.linalg.eigvals(input_model['A'])
        ))

    plot_eigenvalues_grid_comparison(results_comparison)
    '''