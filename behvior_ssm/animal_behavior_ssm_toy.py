
"""
animal_behavior_ssm_toy.py
==========================

Toy project illustrating the end‑to‑end workflow for **unsupervised behaviour
segmentation** and **state‑space modelling** (MC → LDS → SLDS) on simple
**animal‑pose (skeleton) time‑series**.

What you get
------------
* Synthetic skeleton generator (4 joints, 2‑D)
* Unsupervised behaviour discovery with tiny K‑means
* Markov‑chain estimation of transitions
* LDS & SLDS simulation conditioned on discovered states
* Matplotlib visualisations for quick sanity‑checks

Run
---
::

    python animal_behavior_ssm_toy.py

It pops up four figures and prints intermediate evaluation numbers.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Synthetic skeleton generator
# ---------------------------------------------------------------------
def generate_skeleton(
    T: int = 1_000,
    n_joints: int = 4,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    T
        Number of time‑steps.
    n_joints
        #Joints (each joint has x, y).
    seed
        RNG seed.

    Returns
    -------
    skeleton : (T, n_joints, 2) array
    true_states : (T,) int array in {0:rest, 1:walk, 2:run}
    """
    rng = np.random.default_rng(seed)

    # Hidden Markov chain for *ground‑truth* behaviour
    P = np.array(
        [[0.94, 0.05, 0.01],    # rest  → {rest, walk, run}
         [0.05, 0.90, 0.05],    # walk  → …
         [0.02, 0.08, 0.90]]    # run   → …
    )
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(3, p=P[states[t - 1]])

    # Per‑state forward velocity (x‑axis) and jitter scale
    v_map    = np.array([0.000, 0.015, 0.040])
    jitter   = np.array([0.002, 0.003, 0.004])

    # Base posture (triangle‑ish animal)
    base = np.array([
        [0.0,  0.0],   # nose
        [-0.1, 0.05],  # left ear
        [-0.1,-0.05],  # right ear
        [-0.25, 0.0],  # tail root
    ]) * 50  # make dimensions ~pixels

    skeleton = np.empty((T, n_joints, 2), float)
    skeleton[0] = base
    for t in range(1, T):
        v = v_map[states[t]]
        dv = np.array([v, 0.0])
        # translate previous pose, add tiny Gaussian jitter
        skeleton[t] = skeleton[t - 1] + dv + rng.normal(scale=jitter[states[t]], size=(n_joints, 2))
    return skeleton, states


# ---------------------------------------------------------------------
# 2. Tiny NumPy K‑means (for zero‑dependency clustering)
# ---------------------------------------------------------------------
def kmeans_np(x: np.ndarray, k: int, n_iter: int = 50, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Extremely small K‑means; *x* is (N, D).  Returns labels & centroids.
    """
    rng = np.random.default_rng(seed)
    N, D = x.shape
    centroids = x[rng.choice(N, size=k, replace=False)]
    for _ in range(n_iter):
        # assign
        dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels = dists.argmin(1)
        # update
        for j in range(k):
            pts = x[labels == j]
            if len(pts):
                centroids[j] = pts.mean(0)
    return labels, centroids


def unsupervised_segmentation(skeleton: np.ndarray) -> np.ndarray:
    """
    Uses **speed of nose joint** as 1‑D feature + K‑means (k=3).
    """
    nose = skeleton[:, 0, :]          # (T, 2)
    speed = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    speed = np.concatenate([[0.0], speed])  # pad to length T

    labels, cents = kmeans_np(speed[:, None], k=3, seed=0)
    # Map clusters to {rest, walk, run} via ascending centroid speed
    ordering = np.argsort(cents[:, 0])
    mapped = np.array([ordering[label] for label in labels], int)
    return mapped


# ---------------------------------------------------------------------
# 3. Transition matrix estimation
# ---------------------------------------------------------------------
def estimate_markov(labels: np.ndarray, n_states: int = 3) -> np.ndarray:
    counts = np.zeros((n_states, n_states))
    for a, b in zip(labels[:-1], labels[1:]):
        counts[a, b] += 1
    # add Dirichlet(1) prior for numerical safety
    P_hat = (counts + 1) / (counts.sum(1, keepdims=True) + n_states)
    return P_hat


# ---------------------------------------------------------------------
# 4. LDS dynamics per state (position / velocity in x‑direction)
# ---------------------------------------------------------------------
def simulate_lds(
    labels: np.ndarray,
    P: np.ndarray,
    T_future: int = 600,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = {         # per‑state LDS transition for [x, vx]
        0: np.array([[1.0, 1.0],
                     [0.0, 0.8]]),   # rest-ish slow decay
        1: np.array([[1.0, 1.0],
                     [0.0, 0.9]]),   # walk
        2: np.array([[1.0, 1.0],
                     [0.0, 0.95]]),  # run
    }
    C = np.array([1.0, 0.0])

    z = np.zeros((T_future, 2))
    x = np.zeros(T_future)
    z[0] = np.array([0.0, 0.1])
    x[0] = C @ z[0]
    states = np.empty(T_future, int)
    states[0] = labels[-1]         # start from last discovered state

    for t in range(1, T_future):
        states[t] = rng.choice(3, p=P[states[t - 1]])
        z[t] = A[states[t]] @ z[t - 1] + rng.normal(scale=0.05, size=2)
        x[t] = C @ z[t] + rng.normal(scale=0.05)
    return x, states


# ---------------------------------------------------------------------
# 5. Plot helpers
# ---------------------------------------------------------------------
def plot_pose_sequence(skel: np.ndarray, labels: np.ndarray, step: int = 25):
    """
    Scatter nose trajectory with colour = behaviour label.  Every *step* frame.
    """
    nose = skel[::step, 0, :]
    plt.figure(figsize=(6, 3))
    plt.title("Nose trajectory (sub‑sampled)")
    plt.scatter(nose[:, 0], nose[:, 1], c=labels[::step], s=8, cmap="viridis")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()


def plot_speed(speed: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(8, 2.5))
    plt.title("Speed & discovered behaviour")
    plt.plot(speed, lw=0.8, label="speed")
    plt.scatter(np.arange(len(speed)), speed, c=labels, s=8, cmap="viridis")
    plt.ylabel("pixel / frame"); plt.xlabel("time")
    plt.tight_layout()


# ---------------------------------------------------------------------
# 6. Main demo
# ---------------------------------------------------------------------
def main():
    print("Generating synthetic skeleton …")
    skeleton, true_states = generate_skeleton(T=1_200, seed=0)

    print("Running unsupervised segmentation …")
    disc_labels = unsupervised_segmentation(skeleton)
    acc = (disc_labels == true_states).mean()
    print(f"  → Adjusted Rand sim ≈ {acc:.2%} (for toy data)")

    # estimate transition
    P_hat = estimate_markov(disc_labels)
    print("Estimated transition matrix (rows sum to 1):")
    print(np.round(P_hat, 3))

    # future simulation
    print("Simulating future LDS dynamics conditioned on states …")
    x_future, states_future = simulate_lds(disc_labels, P_hat)

    # --- plotting ----------------------------------------------------
    nose = skeleton[:, 0, :]
    speed = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    speed = np.concatenate([[0.0], speed])

    plot_speed(speed, disc_labels)
    plot_pose_sequence(skeleton, disc_labels)
    plt.figure(figsize=(8, 2.5))
    plt.title("Simulated 1‑D observation from LDS")
    plt.plot(x_future, lw=0.8); plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
