# core/similarity_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .lfd_descriptor import LFDModelDescriptor
from .depth_buffer_descriptor import DepthModelDescriptor


ModelDescriptor = Union[LFDModelDescriptor, DepthModelDescriptor]


@dataclass(frozen=True)
class SimilarityResult:
    """
    Result of comparing two model descriptors.
    """
    distance: float
    best_rotation_id: int
    # For debugging / analysis:
    best_permutation: Optional[np.ndarray] = None  # (N,) mapping i -> perm[i]


class SimilarityEngine:
    """
    Unified SimilarityEngine supporting BOTH:
      - LFD (silhouette-based, preset LFD_10): rotation handled by cyclic shifts on ring indices
      - DepthBuffer (depth-based, preset DEPTH_42): rotation handled by discrete 3D rotations + direction remapping

    It computes:
      distance(A, B) = min_over_rotations  aggregate_i  dist(viewA[i], viewB[perm_rot[i]])

    Notes:
    - This engine assumes you already computed per-view features for each model.
    - It does not load meshes or render anything.
    """

    def __init__(
        self,
        metric: Literal["l2", "l1", "cosine"] = "l2",
        aggregation: Literal["mean", "sum"] = "mean",
        depth_rotation_set: Literal["yaw12", "yaw24", "grid12", "grid24"] = "grid12",
        # NEW: optional feature normalization inside the engine (safe default = False)
        l2_normalize: bool = False,
    ) -> None:
        """
        metric:
          - "l2": Euclidean distance between feature vectors
          - "l1": Manhattan distance
          - "cosine": cosine distance (1 - cosine similarity)

        aggregation:
          - "mean": average across views
          - "sum":  sum across views

        depth_rotation_set:
          Defines how many 3D rotations we test for DEPTH_42.
          Options (practical, finite, deterministic):
            - "yaw12": yaw-only rotations, 12 steps (fast, weaker invariance)
            - "yaw24": yaw-only rotations, 24 steps
            - "grid12": small yaw/pitch grid (~12 rotations)
            - "grid24": larger yaw/pitch grid (~24 rotations)

        l2_normalize:
          If True, L2-normalize EACH per-view feature vector before distance computation.
          (Useful if you want to enforce it even if descriptors were stored unnormalized.)
        """
        self.metric = metric
        self.aggregation = aggregation
        self.depth_rotation_set = depth_rotation_set
        self.l2_normalize = bool(l2_normalize)

        # Cache for depth permutations:
        # key: (hash(directions_bytes), rotation_set) -> list of permutations
        self._depth_perm_cache: Dict[Tuple[int, str], List[np.ndarray]] = {}

    # ---------------------------
    # Public API
    # ---------------------------
    def compare(self, a: ModelDescriptor, b: ModelDescriptor) -> SimilarityResult:
        """
        Compare two model descriptors and return the rotation-invariant distance.
        """
        preset_a = getattr(a, "meta").preset
        preset_b = getattr(b, "meta").preset
        if preset_a != preset_b:
            raise ValueError(f"Cannot compare different presets: {preset_a} vs {preset_b}")

        if preset_a == "LFD_10":
            return self._compare_lfd(a, b)  # type: ignore[arg-type]
        elif preset_a == "DEPTH_42":
            return self._compare_depth(a, b)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown preset: {preset_a}")

    # ---------------------------
    # LFD comparison
    # ---------------------------
    def _compare_lfd(self, a: LFDModelDescriptor, b: LFDModelDescriptor) -> SimilarityResult:
        A = np.asarray(a.features, dtype=np.float32)
        B = np.asarray(b.features, dtype=np.float32)

        if A.shape != B.shape:
            raise ValueError(f"LFD shapes differ: {A.shape} vs {B.shape}")

        if self.l2_normalize:
            A = self._l2_normalize_rows(A)
            B = self._l2_normalize_rows(B)

        ring_start = int(a.meta.ring_start)
        ring_size = int(a.meta.ring_size)
        n_views = A.shape[0]

        if ring_start + ring_size > n_views:
            raise ValueError("Invalid ring_start/ring_size for LFD descriptor.")

        best_d = float("inf")
        best_k = 0
        best_perm = None

        for k in range(ring_size):
            perm = self._lfd_ring_shift_permutation(n_views, ring_start, ring_size, k)
            d = self._aggregate_distance(A, B, perm)

            if d < best_d:
                best_d = d
                best_k = k
                best_perm = perm

        return SimilarityResult(distance=float(best_d), best_rotation_id=int(best_k), best_permutation=best_perm)

    @staticmethod
    def _lfd_ring_shift_permutation(n_views: int, ring_start: int, ring_size: int, shift: int) -> np.ndarray:
        """
        Build a permutation perm of length n_views such that:
          - indices outside ring map to themselves
          - ring indices map by cyclic shift

        perm[i] gives which index in B corresponds to view i in A.
        """
        perm = np.arange(n_views, dtype=np.int32)
        ring_idx = np.arange(ring_start, ring_start + ring_size, dtype=np.int32)

        shifted = ring_idx[(np.arange(ring_size) + shift) % ring_size]
        perm[ring_idx] = shifted
        return perm

    # ---------------------------
    # Depth comparison
    # ---------------------------
    def _compare_depth(self, a: DepthModelDescriptor, b: DepthModelDescriptor) -> SimilarityResult:
        A = np.asarray(a.features, dtype=np.float32)
        B = np.asarray(b.features, dtype=np.float32)

        if A.shape != B.shape:
            raise ValueError(f"DEPTH shapes differ: {A.shape} vs {B.shape}")

        if self.l2_normalize:
            A = self._l2_normalize_rows(A)
            B = self._l2_normalize_rows(B)

        directions = np.asarray(a.meta.directions, dtype=np.float32)
        if directions.shape[0] != A.shape[0] or directions.shape[1] != 3:
            raise ValueError("Depth metadata directions must have shape (N,3) aligned with features.")

        perms = self._get_depth_permutations(directions, self.depth_rotation_set)

        best_d = float("inf")
        best_r = 0
        best_perm = None

        for rid, perm in enumerate(perms):
            d = self._aggregate_distance(A, B, perm)
            if d < best_d:
                best_d = d
                best_r = rid
                best_perm = perm

        return SimilarityResult(distance=float(best_d), best_rotation_id=int(best_r), best_permutation=best_perm)

    def _get_depth_permutations(self, directions: np.ndarray, rotation_set: str) -> List[np.ndarray]:
        """
        Build (and cache) the list of permutations induced by a chosen set of 3D rotations.

        directions: (N,3) unit vectors, fixed ordering from ViewGenerator.
        Each rotation R produces:
          d'_i = R d_i
          perm[i] = argmax_j dot(d'_i, d_j)
        """
        key = (hash(directions.tobytes()), rotation_set)
        if key in self._depth_perm_cache:
            return self._depth_perm_cache[key]

        # Ensure unit
        D = directions.astype(np.float32)
        D = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)

        rotations = self._make_rotation_matrices(rotation_set)

        perms: List[np.ndarray] = []
        for R in rotations:
            Dp = (R @ D.T).T  # (N,3)
            sim = Dp @ D.T
            perm = np.argmax(sim, axis=1).astype(np.int32)
            perms.append(perm)

        self._depth_perm_cache[key] = perms
        return perms

    # ---------------------------
    # Rotation sets (Depth)
    # ---------------------------
    def _make_rotation_matrices(self, rotation_set: str) -> List[np.ndarray]:
        """
        Return a deterministic, finite list of 3x3 rotation matrices.
        """
        if rotation_set == "yaw12":
            yaws = np.linspace(0, 360, 12, endpoint=False)
            return [self._Ry(np.deg2rad(y)) for y in yaws]

        if rotation_set == "yaw24":
            yaws = np.linspace(0, 360, 24, endpoint=False)
            return [self._Ry(np.deg2rad(y)) for y in yaws]

        if rotation_set == "grid12":
            mats: List[np.ndarray] = []
            for y in np.linspace(0, 360, 6, endpoint=False):
                mats.append(self._Ry(np.deg2rad(y)) @ self._Rx(0.0))
            for y in np.linspace(0, 360, 3, endpoint=False):
                mats.append(self._Ry(np.deg2rad(y)) @ self._Rx(np.deg2rad(30)))
            for y in np.linspace(0, 360, 3, endpoint=False):
                mats.append(self._Ry(np.deg2rad(y)) @ self._Rx(np.deg2rad(-30)))
            return mats

        if rotation_set == "grid24":
            mats: List[np.ndarray] = []
            for pitch in [0.0, 30.0, -30.0]:
                for y in np.linspace(0, 360, 8, endpoint=False):
                    mats.append(self._Ry(np.deg2rad(y)) @ self._Rx(np.deg2rad(pitch)))
            return mats

        raise ValueError(f"Unknown depth_rotation_set: {rotation_set}")

    @staticmethod
    def _Rx(a: float) -> np.ndarray:
        ca, sa = float(np.cos(a)), float(np.sin(a))
        return np.array(
            [[1, 0, 0],
             [0, ca, -sa],
             [0, sa, ca]],
            dtype=np.float32,
        )

    @staticmethod
    def _Ry(a: float) -> np.ndarray:
        ca, sa = float(np.cos(a)), float(np.sin(a))
        return np.array(
            [[ca, 0, sa],
             [0,  1, 0],
             [-sa, 0, ca]],
            dtype=np.float32,
        )

    @staticmethod
    def _Rz(a: float) -> np.ndarray:
        ca, sa = float(np.cos(a)), float(np.sin(a))
        return np.array(
            [[ca, -sa, 0],
             [sa,  ca, 0],
             [0,    0, 1]],
            dtype=np.float32,
        )

    # ---------------------------
    # Distance utilities
    # ---------------------------
    def _aggregate_distance(self, A: np.ndarray, B: np.ndarray, perm: np.ndarray) -> float:
        """
        Aggregate per-view distances under a given permutation.
        A: (N,d), B: (N,d), perm: (N,) mapping A index -> B index
        """
        Bp = B[perm]
        dv = self._view_distance(A, Bp)  # (N,)

        if self.aggregation == "mean":
            return float(dv.mean())
        elif self.aggregation == "sum":
            return float(dv.sum())
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _view_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute distance between corresponding rows of X and Y.
        X, Y: (N,d) -> returns (N,)
        """
        if self.metric == "l2":
            diff = X - Y
            return np.sqrt(np.sum(diff * diff, axis=1) + 1e-12).astype(np.float32)

        if self.metric == "l1":
            return np.sum(np.abs(X - Y), axis=1).astype(np.float32)

        if self.metric == "cosine":
            xn = np.linalg.norm(X, axis=1) + 1e-12
            yn = np.linalg.norm(Y, axis=1) + 1e-12
            sim = np.sum(X * Y, axis=1) / (xn * yn)
            return (1.0 - sim).astype(np.float32)

        raise ValueError(f"Unknown metric: {self.metric}")

    # ---------------------------
    # NEW: normalization utility
    # ---------------------------
    @staticmethod
    def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n = np.clip(n, eps, None)
        return (X / n).astype(np.float32)
