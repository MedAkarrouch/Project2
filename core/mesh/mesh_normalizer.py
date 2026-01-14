from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..mesh.mesh_loader import Mesh


class MeshNormalizer:
    """
    Normalizes a mesh for retrieval:
      - Center vertices at origin
      - Scale vertices to fit in unit sphere (max distance = 1)

    Implemented as staticmethod so BOTH work:
      - MeshNormalizer.normalize(mesh)
      - MeshNormalizer().normalize(mesh)
    """

    @staticmethod
    def normalize(mesh: Mesh, *, eps: float = 1e-12) -> Mesh:
        if mesh is None:
            raise ValueError("MeshNormalizer.normalize(): mesh is None")

        v = mesh.vertices
        f = mesh.faces

        if not isinstance(v, np.ndarray) or v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"Invalid vertices array shape: {getattr(v, 'shape', None)}")
        if v.shape[0] == 0:
            raise ValueError("Mesh has no vertices")

        # Ensure dtype
        v = v.astype(np.float32, copy=False)

        # Replace non-finite values (defensive)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # 1) Centering
        center = v.mean(axis=0, dtype=np.float64).astype(np.float32)
        v_centered = v - center

        # 2) Scaling to unit sphere
        radii = np.linalg.norm(v_centered, axis=1)
        max_r = float(np.max(radii)) if radii.size else 0.0

        if max_r < eps:
            # Degenerate mesh (all points identical or extremely close)
            # Keep centered vertices (will be ~0)
            v_norm = v_centered
        else:
            v_norm = v_centered / max_r

        # Faces: keep as-is (but ensure int32 ndarray)
        if not isinstance(f, np.ndarray):
            f = np.asarray(f, dtype=np.int32)
        else:
            f = f.astype(np.int32, copy=False)

        return Mesh(vertices=v_norm.astype(np.float32, copy=False), faces=f)
