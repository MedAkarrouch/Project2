"""
TODO:
1. Class design & modularity : 
- Since all methods are static, convert the class MeshLoader to a module-level function.
- Add configuration class for loading option(skip validation, custom triangulation,..)

2. Performance : 
- Add support for parallel processing.
- Use memory mapping for large files.
- Implement streaming parsing for memory efficieny
"""







from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """Minimal mesh representation for geometry-based retrieval."""
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray     # (M, 3) int32, 0-based indices


class MeshLoader:
    """
    Loads a Wavefront OBJ mesh and returns ONLY:
      - vertices (v)
      - faces (f) -> triangulated, 0-based indices

    Notes:
    - Ignores vt/vn and materials.
    - Supports faces in formats:
        f v1 v2 v3
        f v1/vt1 v2/vt2 v3/vt3
        f v1//vn1 v2//vn2 v3//vn3
        f v1/vt1/vn1 ...
    - Supports polygons (n-gons) by fan triangulation.
    - Supports negative indices per OBJ spec.
    """

    @staticmethod
    def load(obj_path: Union[str, Path]) -> Mesh:
        obj_path = Path(obj_path)
        if not obj_path.exists():
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")

        vertices: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []

        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Vertex
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x, y, z))
                    continue

                # Face
                if line.startswith("f "):
                    # Example tokens: ["f", "12/7/5", "14/9/5", "16/11/5", ...]
                    tokens = line.split()[1:]
                    if len(tokens) < 3:
                        continue

                    # Parse each vertex reference into a vertex index (ignore vt/vn)
                    vidx = [MeshLoader._parse_vertex_index(tok, n_verts=len(vertices)) for tok in tokens]

                    # Triangulate polygon using fan triangulation:
                    # (v0, v1, v2), (v0, v2, v3), ...
                    v0 = vidx[0]
                    for i in range(1, len(vidx) - 1):
                        faces.append((v0, vidx[i], vidx[i + 1]))
                    continue

                # Ignore everything else: vt, vn, usemtl, mtllib, g, o, s, ...
                continue

        if len(vertices) == 0:
            raise ValueError(f"No vertices found in OBJ: {obj_path}")
        if len(faces) == 0:
            raise ValueError(f"No faces found in OBJ: {obj_path} (maybe it's a point cloud?)")

        v = np.asarray(vertices, dtype=np.float32)
        f_idx = np.asarray(faces, dtype=np.int32)

        # Basic sanity checks
        if f_idx.min() < 0 or f_idx.max() >= v.shape[0]:
            raise ValueError("Face indices out of bounds after parsing. OBJ may be malformed.")

        return Mesh(vertices=v, faces=f_idx)

    @staticmethod
    def _parse_vertex_index(token: str, n_verts: int) -> int:
        """
        Parse a face token to obtain the vertex index only.
        - token examples: "3", "3/2", "3//7", "3/2/7"
        - OBJ indices are 1-based; negative indices are relative to end.
        Returns 0-based index.
        """
        v_str = token.split("/")[0]
        if not v_str:
            raise ValueError(f"Invalid face token (missing vertex index): '{token}'")

        vi = int(v_str)
        if vi > 0:
            # 1-based -> 0-based
            idx = vi - 1
        else:
            # Negative indices: -1 refers to last defined vertex
            idx = n_verts + vi  # vi is negative
        return idx
