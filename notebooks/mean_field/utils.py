"""
Mean-Field Transformer Dynamics Utilities
==========================================

Core simulation and visualization functions for exploring the mean-field dynamics
of self-attention, based on Rigollet et al. "The Mean-Field Dynamics of Transformers"
(arXiv:2512.01868v3).

This module provides:
- Particle dynamics on the unit sphere (SA/USA equations)
- ODE integration for simulating attention dynamics
- Energy functional computation
- Visualization utilities for 3D trajectories and analysis plots
- Interactive widget helpers

Theory Reference:
- Tokens are particles on the unit sphere S^{d-1}
- Self-Attention (SA): ẋᵢ = P⊥_{xᵢ} [ (1/Zᵢ) Σⱼ exp(β⟨xᵢ,xⱼ⟩) xⱼ ]
- Unnormalized SA (USA): ẋᵢ = P⊥_{xᵢ} [ (1/n) Σⱼ exp(β⟨xᵢ,xⱼ⟩) xⱼ ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.special import softmax


# =============================================================================
# Core Dynamics Functions
# =============================================================================

def project_to_sphere(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalize vectors to lie on the unit sphere.

    Args:
        x: Array of shape (n, d) or (d,) - vectors to normalize

    Returns:
        Normalized vectors with unit L2 norm
    """
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / norm if norm > 1e-10 else x
    else:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return x / norms


def orthogonal_projection(x: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Project vector v onto the tangent space of the sphere at point x.

    P⊥_x(v) = v - ⟨x, v⟩x

    This keeps the dynamics on the sphere by removing the radial component.

    Args:
        x: Point on sphere, shape (d,) or (n, d)
        v: Vector to project, same shape as x

    Returns:
        Projected vector in tangent space at x
    """
    if x.ndim == 1:
        return v - np.dot(x, v) * x
    else:
        # For batch: (n, d) @ (n, d) -> (n,) then broadcast
        dot_products = np.sum(x * v, axis=1, keepdims=True)
        return v - dot_products * x


def attention_weights(
    X: NDArray[np.floating],
    beta: float,
    normalized: bool = True
) -> NDArray[np.floating]:
    """
    Compute attention weights between all pairs of particles.

    A_ij = exp(β⟨xᵢ, xⱼ⟩) / Z_i  (normalized)
    A_ij = exp(β⟨xᵢ, xⱼ⟩) / n     (unnormalized)

    Args:
        X: Particle positions, shape (n, d)
        beta: Temperature parameter (inverse temperature)
        normalized: If True, use softmax normalization (SA). If False, use 1/n (USA).

    Returns:
        Attention weight matrix, shape (n, n)
    """
    n = X.shape[0]
    # Compute pairwise inner products
    similarities = X @ X.T  # Shape (n, n)

    if normalized:
        # Self-Attention: softmax over rows
        return softmax(beta * similarities, axis=1)
    else:
        # Unnormalized Self-Attention: exp / n
        return np.exp(beta * similarities) / n


def sa_velocity(X: NDArray[np.floating], beta: float) -> NDArray[np.floating]:
    """
    Compute the Self-Attention (SA) velocity field.

    ẋᵢ = P⊥_{xᵢ} [ Σⱼ A_ij xⱼ ]

    where A_ij = softmax(β⟨xᵢ, xⱼ⟩)

    Args:
        X: Particle positions on sphere, shape (n, d)
        beta: Temperature parameter

    Returns:
        Velocity vectors, shape (n, d)
    """
    A = attention_weights(X, beta, normalized=True)
    # Weighted average of all particles
    weighted_avg = A @ X  # Shape (n, d)
    # Project onto tangent space at each particle
    return orthogonal_projection(X, weighted_avg)


def usa_velocity(X: NDArray[np.floating], beta: float) -> NDArray[np.floating]:
    """
    Compute the Unnormalized Self-Attention (USA) velocity field.

    ẋᵢ = P⊥_{xᵢ} [ (1/n) Σⱼ exp(β⟨xᵢ, xⱼ⟩) xⱼ ]

    Args:
        X: Particle positions on sphere, shape (n, d)
        beta: Temperature parameter

    Returns:
        Velocity vectors, shape (n, d)
    """
    A = attention_weights(X, beta, normalized=False)
    weighted_avg = A @ X
    return orthogonal_projection(X, weighted_avg)


def kuramoto_velocity(theta: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the Kuramoto model velocity (β=0 limit of USA on the circle).

    θ̇ᵢ = -(1/n) Σⱼ sin(θᵢ - θⱼ)

    This is the classical synchronized oscillator model.

    Args:
        theta: Angles on the circle, shape (n,)

    Returns:
        Angular velocities, shape (n,)
    """
    n = len(theta)
    # Compute all pairwise phase differences
    diff = theta[:, np.newaxis] - theta[np.newaxis, :]  # Shape (n, n)
    return -np.mean(np.sin(diff), axis=1)


def kuramoto_velocity_general(
    theta: NDArray[np.floating],
    beta: float
) -> NDArray[np.floating]:
    """
    Compute the generalized Kuramoto velocity for arbitrary β.

    θ̇ᵢ = -(1/n) Σⱼ exp(β cos(θᵢ - θⱼ)) sin(θᵢ - θⱼ)

    For β=0, this reduces to the standard Kuramoto model.

    Args:
        theta: Angles on the circle, shape (n,)
        beta: Temperature parameter

    Returns:
        Angular velocities, shape (n,)
    """
    n = len(theta)
    diff = theta[:, np.newaxis] - theta[np.newaxis, :]
    weights = np.exp(beta * np.cos(diff))
    return -np.mean(weights * np.sin(diff), axis=1)


# =============================================================================
# ODE Simulation
# =============================================================================

def simulate_dynamics(
    X0: NDArray[np.floating],
    velocity_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    t_span: tuple[float, float],
    n_steps: int = 100,
    method: str = 'RK45'
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Simulate particle dynamics on the sphere using ODE integration.

    Args:
        X0: Initial positions, shape (n, d)
        velocity_fn: Function that computes velocity given positions
        t_span: (t_start, t_end) time interval
        n_steps: Number of time points to return
        method: ODE solver method ('RK45', 'RK23', 'DOP853', etc.)

    Returns:
        times: Time points, shape (n_steps,)
        trajectory: Particle positions over time, shape (n_steps, n, d)
    """
    n, d = X0.shape

    def flat_velocity(t: float, x_flat: NDArray[np.floating]) -> NDArray[np.floating]:
        """Wrapper that handles flattening/unflattening for scipy."""
        X = x_flat.reshape(n, d)
        # Ensure we stay on the sphere (numerical drift correction)
        X = project_to_sphere(X)
        V = velocity_fn(X)
        return V.flatten()

    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    solution = solve_ivp(
        flat_velocity,
        t_span,
        X0.flatten(),
        t_eval=t_eval,
        method=method,
        dense_output=False
    )

    # Reshape trajectory and project back to sphere
    trajectory = solution.y.T.reshape(n_steps, n, d)
    for i in range(n_steps):
        trajectory[i] = project_to_sphere(trajectory[i])

    return solution.t, trajectory


def simulate_sa(
    X0: NDArray[np.floating],
    beta: float,
    t_span: tuple[float, float],
    n_steps: int = 100
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Convenience function to simulate Self-Attention dynamics.

    Args:
        X0: Initial positions on sphere, shape (n, d)
        beta: Temperature parameter
        t_span: Time interval (t_start, t_end)
        n_steps: Number of output time points

    Returns:
        times, trajectory
    """
    return simulate_dynamics(
        X0,
        lambda X: sa_velocity(X, beta),
        t_span,
        n_steps
    )


def simulate_usa(
    X0: NDArray[np.floating],
    beta: float,
    t_span: tuple[float, float],
    n_steps: int = 100
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Convenience function to simulate Unnormalized Self-Attention dynamics.
    """
    return simulate_dynamics(
        X0,
        lambda X: usa_velocity(X, beta),
        t_span,
        n_steps
    )


def simulate_kuramoto(
    theta0: NDArray[np.floating],
    t_span: tuple[float, float],
    n_steps: int = 100,
    beta: float = 0.0
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Simulate Kuramoto oscillator dynamics on the circle.

    Args:
        theta0: Initial angles, shape (n,)
        t_span: Time interval
        n_steps: Number of output points
        beta: Temperature (0 for classical Kuramoto)

    Returns:
        times: shape (n_steps,)
        trajectory: angles over time, shape (n_steps, n)
    """
    n = len(theta0)

    if beta == 0:
        vel_fn = lambda theta: kuramoto_velocity(theta)
    else:
        vel_fn = lambda theta: kuramoto_velocity_general(theta, beta)

    def flat_velocity(t: float, theta: NDArray[np.floating]) -> NDArray[np.floating]:
        return vel_fn(theta)

    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    solution = solve_ivp(
        flat_velocity,
        t_span,
        theta0,
        t_eval=t_eval,
        method='RK45'
    )

    return solution.t, solution.y.T


# =============================================================================
# Energy and Analysis Functions
# =============================================================================

def compute_energy(X: NDArray[np.floating], beta: float) -> float:
    """
    Compute the energy functional for the SA/USA dynamics.

    E_β(X) = (1/2β) Σᵢⱼ exp(β⟨xᵢ, xⱼ⟩)

    The SA/USA dynamics are gradient flows of this energy.

    Args:
        X: Particle positions, shape (n, d)
        beta: Temperature parameter (must be > 0)

    Returns:
        Energy value (scalar)
    """
    if beta <= 0:
        raise ValueError("beta must be positive for energy computation")

    similarities = X @ X.T
    return (1 / (2 * beta)) * np.sum(np.exp(beta * similarities))


def pairwise_similarity_dist(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute distribution of pairwise inner products (cosine similarities).

    Returns only off-diagonal elements (i ≠ j).

    Args:
        X: Particle positions on sphere, shape (n, d)

    Returns:
        Array of pairwise similarities, shape (n*(n-1)/2,)
    """
    similarities = X @ X.T
    # Extract upper triangle (excluding diagonal)
    return similarities[np.triu_indices(len(X), k=1)]


def compute_cluster_metric(X: NDArray[np.floating]) -> float:
    """
    Compute a clustering metric: average pairwise similarity.

    When all particles are clustered, this approaches 1.

    Args:
        X: Particle positions on sphere, shape (n, d)

    Returns:
        Mean pairwise similarity (0 to 1)
    """
    sims = pairwise_similarity_dist(X)
    return float(np.mean(sims))


def detect_clusters(
    X: NDArray[np.floating],
    threshold: float = 0.95
) -> list[list[int]]:
    """
    Detect clusters of particles based on similarity threshold.

    Two particles are in the same cluster if ⟨xᵢ, xⱼ⟩ > threshold.

    Args:
        X: Particle positions, shape (n, d)
        threshold: Similarity threshold for clustering

    Returns:
        List of clusters, each cluster is a list of particle indices
    """
    n = X.shape[0]
    similarities = X @ X.T

    # Union-find for connected components
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    # Connect particles with high similarity
    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] > threshold:
                union(i, j)

    # Group by cluster
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    return list(clusters.values())


# =============================================================================
# Equiangular Model (1D reduction)
# =============================================================================

def equiangular_ode_rhs(
    rho: float,
    t: float,
    beta: float,
    n: int,
    norm_type: Literal['pre_ln', 'post_ln']
) -> float:
    """
    Right-hand side of the equiangular model ODE.

    In the equiangular model, all pairwise correlations are equal:
    ⟨xᵢ, xⱼ⟩ = ρ for all i ≠ j

    This reduces the n-particle system to a 1D ODE for ρ(t).

    Args:
        rho: Current correlation value (-1/(n-1) < ρ < 1)
        t: Time (unused, but required by ODE solver)
        beta: Temperature parameter
        n: Number of particles
        norm_type: 'pre_ln' or 'post_ln' normalization

    Returns:
        dρ/dt
    """
    if rho >= 1:
        return 0.0  # Already converged

    # Common quantities
    exp_beta = np.exp(beta)
    exp_beta_rho = np.exp(beta * rho)

    if norm_type == 'post_ln':
        # Post-LN: exponential contraction 1 - ρ ~ exp(-2t)
        Z = exp_beta + (n - 1) * exp_beta_rho
        A_self = exp_beta / Z
        A_other = exp_beta_rho / Z

        # ρ̇ for post-LN (equation 24 in paper, adapted)
        numerator = (n - 1) * A_other * (1 - rho)
        drho = numerator * (1 + rho) / (1 + (n - 2) * rho + (n - 1) * rho)

    elif norm_type == 'pre_ln':
        # Pre-LN: polynomial contraction 1 - ρ ~ 1/t²
        # Simpler form from Section 6.2
        avg_weight = (exp_beta + (n - 1) * exp_beta_rho) / n
        drho = (1 - rho) * (exp_beta_rho - rho * avg_weight) / avg_weight

    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    return drho


def simulate_equiangular(
    rho0: float,
    beta: float,
    n: int,
    t_span: tuple[float, float],
    n_steps: int = 100,
    norm_type: Literal['pre_ln', 'post_ln'] = 'post_ln'
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Simulate the 1D equiangular model.

    Args:
        rho0: Initial correlation
        beta: Temperature
        n: Number of particles
        t_span: Time interval
        n_steps: Output points
        norm_type: Normalization scheme

    Returns:
        times, rho_values
    """
    def rhs(t: float, rho: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array([equiangular_ode_rhs(rho[0], t, beta, n, norm_type)])

    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    solution = solve_ivp(
        rhs,
        t_span,
        np.array([rho0]),
        t_eval=t_eval,
        method='RK45'
    )

    return solution.t, solution.y[0]


# =============================================================================
# Initialization Utilities
# =============================================================================

def random_sphere_points(
    n: int,
    d: int,
    seed: int | None = None
) -> NDArray[np.floating]:
    """
    Generate n random points uniformly distributed on S^{d-1}.

    Args:
        n: Number of points
        d: Dimension of ambient space
        seed: Random seed for reproducibility

    Returns:
        Points on unit sphere, shape (n, d)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return project_to_sphere(X)


def clustered_initialization(
    n_clusters: int,
    n_per_cluster: int,
    d: int,
    spread: float = 0.1,
    seed: int | None = None
) -> NDArray[np.floating]:
    """
    Initialize particles in clusters around random centers.

    Args:
        n_clusters: Number of cluster centers
        n_per_cluster: Particles per cluster
        d: Dimension
        spread: Standard deviation of cluster spread (smaller = tighter)
        seed: Random seed

    Returns:
        Particle positions, shape (n_clusters * n_per_cluster, d)
    """
    rng = np.random.default_rng(seed)

    # Random cluster centers
    centers = random_sphere_points(n_clusters, d, seed)

    # Generate points around each center
    points = []
    for i in range(n_clusters):
        # Add noise and reproject to sphere
        noise = rng.standard_normal((n_per_cluster, d)) * spread
        cluster_points = centers[i] + noise
        cluster_points = project_to_sphere(cluster_points)
        points.append(cluster_points)

    return np.vstack(points)


def hemisphere_initialization(
    n: int,
    d: int,
    seed: int | None = None
) -> NDArray[np.floating]:
    """
    Initialize particles in the positive hemisphere (x_1 > 0).

    Useful for demonstrating fast convergence (Theorem 3).

    Args:
        n: Number of particles
        d: Dimension
        seed: Random seed

    Returns:
        Points in positive hemisphere, shape (n, d)
    """
    X = random_sphere_points(n, d, seed)
    # Flip particles to positive hemisphere
    X[X[:, 0] < 0] *= -1
    return X


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_sphere_3d(
    X: NDArray[np.floating],
    trajectories: NDArray[np.floating] | None = None,
    title: str = "Particles on Sphere",
    show_sphere: bool = True,
    colorscale: str = 'Viridis'
) -> 'plotly.graph_objects.Figure':
    """
    Create 3D plotly visualization of particles on the unit sphere.

    Args:
        X: Current particle positions, shape (n, 3)
        trajectories: Optional trajectory history, shape (n_steps, n, 3)
        title: Plot title
        show_sphere: Whether to show wireframe sphere
        colorscale: Color scale for particles

    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go

    if X.shape[1] != 3:
        raise ValueError("plot_sphere_3d requires 3D points (d=3)")

    n = X.shape[0]

    fig = go.Figure()

    # Add wireframe sphere
    if show_sphere:
        phi = np.linspace(0, 2 * np.pi, 30)
        theta = np.linspace(0, np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)

        x_sphere = np.sin(theta) * np.cos(phi)
        y_sphere = np.sin(theta) * np.sin(phi)
        z_sphere = np.cos(theta)

        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            showscale=False,
            name='Sphere'
        ))

    # Add trajectories if provided
    if trajectories is not None:
        colors = np.linspace(0, 1, n)
        for i in range(n):
            fig.add_trace(go.Scatter3d(
                x=trajectories[:, i, 0],
                y=trajectories[:, i, 1],
                z=trajectories[:, i, 2],
                mode='lines',
                line=dict(width=2, color=f'rgba({int(colors[i]*255)}, {int((1-colors[i])*255)}, 150, 0.6)'),
                name=f'Particle {i} trajectory',
                showlegend=False
            ))

    # Add current particle positions
    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=np.arange(n),
            colorscale=colorscale,
            opacity=1.0
        ),
        name='Particles'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            xaxis=dict(range=[-1.2, 1.2]),
            yaxis=dict(range=[-1.2, 1.2]),
            zaxis=dict(range=[-1.2, 1.2])
        ),
        width=700,
        height=700
    )

    return fig


def plot_similarity_evolution(
    trajectory: NDArray[np.floating],
    times: NDArray[np.floating],
    title: str = "Pairwise Similarity Evolution"
) -> 'matplotlib.figure.Figure':
    """
    Plot the evolution of pairwise similarities over time.

    Args:
        trajectory: Particle positions over time, shape (n_steps, n, d)
        times: Time points, shape (n_steps,)
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n_steps = len(times)

    # Compute statistics at each time step
    means = []
    stds = []
    mins = []
    maxs = []

    for i in range(n_steps):
        sims = pairwise_similarity_dist(trajectory[i])
        means.append(np.mean(sims))
        stds.append(np.std(sims))
        mins.append(np.min(sims))
        maxs.append(np.max(sims))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(times, mins, maxs, alpha=0.2, label='Min-Max range')
    ax.fill_between(
        times,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.4,
        label='Mean ± std'
    )
    ax.plot(times, means, 'b-', linewidth=2, label='Mean similarity')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Full clustering')

    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Pairwise Inner Product', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)

    plt.tight_layout()
    return fig


def plot_energy_staircase(
    trajectory: NDArray[np.floating],
    times: NDArray[np.floating],
    beta: float,
    title: str = "Energy Staircase"
) -> 'matplotlib.figure.Figure':
    """
    Plot the energy evolution showing metastable plateaus.

    Args:
        trajectory: Particle positions over time, shape (n_steps, n, d)
        times: Time points
        beta: Temperature parameter
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    energies = [compute_energy(trajectory[i], beta) for i in range(len(times))]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, energies, 'b-', linewidth=2)
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel(f'Energy E_β (β={beta})', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_similarity_histogram(
    X: NDArray[np.floating],
    title: str = "Pairwise Similarity Distribution",
    bins: int = 50
) -> 'matplotlib.figure.Figure':
    """
    Plot histogram of pairwise similarities.

    Args:
        X: Particle positions, shape (n, d)
        title: Plot title
        bins: Number of histogram bins

    Returns:
        Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    sims = pairwise_similarity_dist(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=bins, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(sims), color='r', linestyle='--', label=f'Mean: {np.mean(sims):.3f}')
    ax.set_xlabel('Inner Product ⟨xᵢ, xⱼ⟩', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)

    plt.tight_layout()
    return fig


# =============================================================================
# Interactive Widget Helpers
# =============================================================================

@dataclass
class SimulationResult:
    """Container for simulation results."""
    times: NDArray[np.floating]
    trajectory: NDArray[np.floating]
    beta: float
    n_particles: int
    dimension: int
    dynamics_type: str  # 'SA' or 'USA'

    @property
    def final_positions(self) -> NDArray[np.floating]:
        """Get final particle positions."""
        return self.trajectory[-1]

    @property
    def final_clustering(self) -> float:
        """Get final clustering metric."""
        return compute_cluster_metric(self.final_positions)


def run_interactive_simulation(
    n: int = 10,
    d: int = 3,
    beta: float = 1.0,
    t_end: float = 10.0,
    n_steps: int = 100,
    dynamics_type: Literal['SA', 'USA'] = 'SA',
    seed: int | None = 42
) -> SimulationResult:
    """
    Run a simulation with configurable parameters.

    Designed to be called from ipywidgets interactive elements.

    Args:
        n: Number of particles
        d: Dimension of space
        beta: Temperature parameter
        t_end: End time
        n_steps: Number of output steps
        dynamics_type: 'SA' or 'USA'
        seed: Random seed

    Returns:
        SimulationResult with full trajectory
    """
    X0 = random_sphere_points(n, d, seed)

    if dynamics_type == 'SA':
        times, trajectory = simulate_sa(X0, beta, (0, t_end), n_steps)
    else:
        times, trajectory = simulate_usa(X0, beta, (0, t_end), n_steps)

    return SimulationResult(
        times=times,
        trajectory=trajectory,
        beta=beta,
        n_particles=n,
        dimension=d,
        dynamics_type=dynamics_type
    )


# =============================================================================
# Attention Mechanism (for Notebook 0)
# =============================================================================

def single_head_attention(
    Q: NDArray[np.floating],
    K: NDArray[np.floating],
    V: NDArray[np.floating],
    beta: float = 1.0
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute single-head attention output.

    Attention(Q, K, V) = softmax(β * Q @ K^T) @ V

    Args:
        Q: Query matrix, shape (n, d_k)
        K: Key matrix, shape (n, d_k)
        V: Value matrix, shape (n, d_v)
        beta: Temperature parameter (inverse of typical sqrt(d_k))

    Returns:
        output: Attention output, shape (n, d_v)
        weights: Attention weights, shape (n, n)
    """
    # Compute attention scores
    scores = beta * (Q @ K.T)  # Shape (n, n)

    # Apply softmax
    weights = softmax(scores, axis=1)

    # Weighted sum of values
    output = weights @ V

    return output, weights


def layer_normalization(
    X: NDArray[np.floating],
    eps: float = 1e-6
) -> NDArray[np.floating]:
    """
    Apply layer normalization (projects to approximate unit sphere).

    LayerNorm(x) = (x - mean(x)) / std(x)

    Note: This is a simplified version. In practice, LN also has
    learnable scale (γ) and bias (β) parameters.

    Args:
        X: Input, shape (n, d)
        eps: Small constant for numerical stability

    Returns:
        Normalized output, shape (n, d)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return (X - mean) / (std + eps)


if __name__ == "__main__":
    # Quick test of core functionality
    print("Testing mean-field transformer utilities...")

    # Test basic simulation
    X0 = random_sphere_points(10, 3, seed=42)
    times, trajectory = simulate_sa(X0, beta=2.0, t_span=(0, 5), n_steps=50)

    print(f"Initial clustering metric: {compute_cluster_metric(X0):.4f}")
    print(f"Final clustering metric: {compute_cluster_metric(trajectory[-1]):.4f}")
    print(f"Number of clusters (threshold=0.95): {len(detect_clusters(trajectory[-1], 0.95))}")

    # Test equiangular model
    t_eq, rho_eq = simulate_equiangular(0.5, beta=2.0, n=10, t_span=(0, 5))
    print(f"Equiangular model: ρ(0)={0.5:.2f} → ρ(5)={rho_eq[-1]:.4f}")

    print("All tests passed!")
