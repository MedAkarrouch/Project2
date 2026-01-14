from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """
    Minimal camera pose representation used by Renderer later.

    direction: unit 3D vector (anchor on the unit sphere)
    eye:       camera position in world coordinates (radius * direction)
    target:    usually (0,0,0)
    up:        camera up vector (unit-ish, orthogonalized later by renderer if needed)
    """
    direction: np.ndarray  # (3,) float32
    eye: np.ndarray        # (3,) float32
    target: np.ndarray     # (3,) float32
    up: np.ndarray         # (3,) float32


class ViewGenerator:
    """
    Generates ordered camera viewpoints around the normalized mesh.

    Presets:
      - 'LFD_10'    : structured viewpoints for LFD-style silhouettes
      - 'DEPTH_42'  : 42 approximately uniform viewpoints on the sphere (Fibonacci)
      - 'grid12'    : deterministic 12-view grid on the sphere (yaw x pitch)
      - 'grid24'    : deterministic 24-view grid on the sphere (yaw x pitch)
      - 'yaw12'     : 12 views on a horizontal ring (yaw-only)
      - 'yaw24'     : 24 views on a horizontal ring (yaw-only)

    IMPORTANT:
    - This class does NOT render images and does NOT compute similarity.
    - It only defines a deterministic, reusable ordering of camera poses.
    """

    def __init__(self, seed: int = 0) -> None:
        """
        seed: used only for deterministic generation if a method has any randomness.
              Fibonacci sphere is deterministic even without a seed, but we keep it for future extensibility.
        """
        self.seed = seed

    def generate(
        self,
        preset: Literal["LFD_10", "DEPTH_42", "grid12", "grid24", "yaw12", "yaw24"] = "LFD_10",
        radius: float = 2.5,
        target: Optional[np.ndarray] = None,
    ) -> List[CameraPose]:
        """
        Returns an ordered list of CameraPose.

        radius: distance of camera from origin (mesh is normalized so radius can be fixed)
        target: look-at point (default origin)
        """
        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            target = np.asarray(target, dtype=np.float32).reshape(3)

        # --- choose directions preset
        if preset == "LFD_10":
            directions, _ring_meta = self._lfd10_directions()
        elif preset == "DEPTH_42":
            directions = self._fibonacci_sphere_directions(n=42)
        elif preset == "yaw12":
            directions = self._yaw_ring_directions(n=12)
        elif preset == "yaw24":
            directions = self._yaw_ring_directions(n=24)
        elif preset == "grid12":
            directions = self._grid_directions(n_views=12)
        elif preset == "grid24":
            directions = self._grid_directions(n_views=24)
        else:
            raise ValueError(f"Unknown preset: {preset}")

        poses: List[CameraPose] = []
        for d in directions:
            d = self._normalize(d)

            eye = (radius * d).astype(np.float32)

            # Choose an "up" vector that avoids degeneracy if camera direction is near world-up.
            up = self._choose_up(d)

            poses.append(
                CameraPose(
                    direction=d.astype(np.float32),
                    eye=eye,
                    target=target.copy(),
                    up=up.astype(np.float32),
                )
            )

        self._validate_poses(poses)
        return poses

    # ---------------------------
    # Preset: LFD_10
    # ---------------------------
    def _lfd10_directions(self) -> Tuple[List[np.ndarray], dict]:
        """
        Create 10 structured viewpoints for an LFD-style setup.

        Ordering (IMPORTANT):
          index 0   : top
          index 1   : bottom
          index 2-9 : ring views in increasing azimuth (0, 45, 90, ..., 315 degrees)
        """
        top = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        bottom = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        ring: List[np.ndarray] = []
        for k in range(8):
            az = 2.0 * np.pi * (k / 8.0)
            x = np.cos(az)
            z = np.sin(az)
            ring.append(np.array([x, 0.0, z], dtype=np.float32))

        directions: List[np.ndarray] = [top, bottom] + ring

        ring_meta = {
            "type": "ring",
            "ring_start": 2,
            "ring_size": 8,
            "notes": "For yaw-rotation invariance, SimilarityEngine can test cyclic shifts over indices [2..9].",
        }
        return directions, ring_meta

    # ---------------------------
    # Preset: DEPTH_42 (Fibonacci sphere)
    # ---------------------------
    def _fibonacci_sphere_directions(self, n: int) -> List[np.ndarray]:
        """
        Generate n approximately uniform directions on the unit sphere using a Fibonacci spiral.
        Deterministic; ordering is i=0..n-1.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        directions: List[np.ndarray] = []
        for i in range(n):
            y = 1.0 - 2.0 * (i + 0.5) / n
            r = np.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * i
            x = np.cos(theta) * r
            z = np.sin(theta) * r
            directions.append(np.array([x, y, z], dtype=np.float32))

        return directions

    # ---------------------------
    # Preset: yaw-only rings
    # ---------------------------
    def _yaw_ring_directions(self, n: int) -> List[np.ndarray]:
        """
        n views around the object on the horizontal ring (Y = 0), evenly spaced in azimuth.
        Ordering: increasing azimuth.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        directions: List[np.ndarray] = []
        for k in range(n):
            az = 2.0 * np.pi * (k / float(n))
            x = np.cos(az)
            z = np.sin(az)
            directions.append(np.array([x, 0.0, z], dtype=np.float32))
        return directions

    # ---------------------------
    # Preset: grid views (yaw x pitch)
    # ---------------------------
    def _grid_directions(self, n_views: int) -> List[np.ndarray]:
        """
        Deterministic yaw x pitch grid on the sphere (no randomness).

        We intentionally avoid the exact poles to reduce "up vector" degeneracy.
        For n_views=12:  yaw=6, pitch=2  -> 6*2 = 12
        For n_views=24:  yaw=8, pitch=3  -> 8*3 = 24

        Ordering:
          pitch loop outer (from higher to lower), yaw inner (0..2pi).
          This gives stable deterministic ordering across runs.
        """
        if n_views == 12:
            n_yaw, pitch_degs = 6, [35.0, -35.0]
        elif n_views == 24:
            n_yaw, pitch_degs = 8, [45.0, 0.0, -45.0]
        else:
            raise ValueError("grid_directions supports only n_views=12 or 24.")

        directions: List[np.ndarray] = []
        for p_deg in pitch_degs:
            p = np.deg2rad(p_deg)
            cp, sp = float(np.cos(p)), float(np.sin(p))

            for k in range(n_yaw):
                yaw = 2.0 * np.pi * (k / float(n_yaw))
                cy, sy = float(np.cos(yaw)), float(np.sin(yaw))

                # Spherical to Cartesian (yaw around Y axis, pitch around XZ plane)
                x = cp * cy
                y = sp
                z = cp * sy
                directions.append(np.array([x, y, z], dtype=np.float32))

        return directions

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(v))
        if norm < eps:
            raise ValueError("Cannot normalize near-zero vector.")
        return v / norm

    @staticmethod
    def _choose_up(direction: np.ndarray) -> np.ndarray:
        """
        Choose a stable up vector given a viewing direction.

        If direction is close to world-up (0,1,0), we use world-forward (0,0,1) as up,
        otherwise we use world-up (0,1,0).
        """
        d = np.asarray(direction, dtype=np.float32).reshape(3)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        world_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if abs(float(np.dot(d, world_up))) > 0.95:
            return world_forward
        return world_up

    @staticmethod
    def _validate_poses(poses: List[CameraPose]) -> None:
        if len(poses) == 0:
            raise ValueError("No poses generated.")

        for i, p in enumerate(poses):
            if p.direction.shape != (3,):
                raise ValueError(f"Pose {i}: direction shape must be (3,)")
            if p.eye.shape != (3,):
                raise ValueError(f"Pose {i}: eye shape must be (3,)")
            if p.target.shape != (3,):
                raise ValueError(f"Pose {i}: target shape must be (3,)")
            if p.up.shape != (3,):
                raise ValueError(f"Pose {i}: up shape must be (3,)")

            dn = float(np.linalg.norm(p.direction))
            if not (0.999 <= dn <= 1.001):
                raise ValueError(f"Pose {i}: direction is not unit length (norm={dn})")

            en = float(np.linalg.norm(p.eye))
            if en <= 0:
                raise ValueError(f"Pose {i}: eye has zero length.")
            eye_dir = p.eye / en
            if float(np.dot(eye_dir, p.direction)) < 0.999:
                raise ValueError(f"Pose {i}: eye is not aligned with direction.")
