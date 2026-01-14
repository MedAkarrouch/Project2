# core/depth_buffer_descriptor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from skimage.transform import resize

from ..mesh.mesh_loader import Mesh
from ..mesh.renderer import Renderer
from ..mesh.view_generator import ViewGenerator


@dataclass(frozen=True)
class DepthMetadata:
    """
    Metadata needed by SimilarityEngine (so it doesn't guess).
    """
    preset: Literal["DEPTH_42"]
    image_size: int
    # Ordered view directions (anchors) used to render all depth views
    directions: np.ndarray  # (42, 3) float32
    # Whether depth was linearized when rendered (should match renderer settings)
    linearized_depth: bool
    # Rotation set used later inside SimilarityEngine (grid12/grid24/yaw12/yaw24)
    rotation_set: str
    # Whether per-view features were L2-normalized
    l2_normalized: bool


@dataclass(frozen=True)
class DepthModelDescriptor:
    """
    Final descriptor for one 3D model using depth-buffer views.
    features: (42, feature_dim)
    meta:     DepthMetadata (preset, ordering, directions, etc.)
    """
    features: np.ndarray
    meta: DepthMetadata


class DepthBufferDescriptor:
    """
    Depth-buffer descriptor using depth maps rendered from multiple viewpoints.

    Pipeline:
      Mesh
        → ViewGenerator (DEPTH_42 viewpoints)
        → Renderer (depth)
        → Preprocessing (resize, foreground mask, normalization)
        → Per-view feature vector
        → (optional) per-view L2-normalization
        → Model descriptor (per-view features + metadata)

    Feature design:
      - Depth histogram over foreground pixels
      - Gradient-magnitude histogram (captures surface variation)
      - Scalar depth stats (mean/std/min/max/percentiles)
    """

    def __init__(
        self,
        renderer: Renderer,
        image_size: int = 128,
        depth_hist_bins: int = 32,
        grad_hist_bins: int = 16,
        clip_percentile: float = 99.0,
        view_radius: float = 2.5,
        linearized_depth: bool = True,
        rotation_set: str = "grid12",
        l2_normalize: bool = False,
        l2_eps: float = 1e-12,
    ) -> None:
        self.renderer = renderer
        self.image_size = int(image_size)
        self.depth_hist_bins = int(depth_hist_bins)
        self.grad_hist_bins = int(grad_hist_bins)
        self.clip_percentile = float(clip_percentile)
        self.view_radius = float(view_radius)
        self.linearized_depth = bool(linearized_depth)
        self.rotation_set = str(rotation_set)
        self.l2_normalize = bool(l2_normalize)
        self.l2_eps = float(l2_eps)

        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.depth_hist_bins <= 0:
            raise ValueError("depth_hist_bins must be positive.")
        if self.grad_hist_bins <= 0:
            raise ValueError("grad_hist_bins must be positive.")
        if not (0.0 < self.clip_percentile <= 100.0):
            raise ValueError("clip_percentile must be in (0, 100].")
        if self.l2_eps <= 0:
            raise ValueError("l2_eps must be positive.")

        # rotation_set is used in SimilarityEngine, not in ViewGenerator.
        # Keep it as metadata for reproducibility, but DO NOT pass it to ViewGenerator.
        self.view_generator = ViewGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, mesh: Mesh) -> DepthModelDescriptor:
        """
        Compute DepthBuffer descriptor for one mesh.

        Returns:
            DepthModelDescriptor with:
              - features: (42, feature_dim)
              - meta: preset, directions, depth linearization flag, rotation_set, l2 flag
        """
        # Always render the fixed DEPTH_42 viewpoints (42 directions).
        poses = self.view_generator.generate(preset="DEPTH_42", radius=self.view_radius)
        directions = np.asarray(np.stack([p.direction for p in poses], axis=0), dtype=np.float32)

        depth_maps = self.renderer.render_views(
            mesh,
            poses,
            mode="depth",
            linearize_depth=self.linearized_depth,
        )

        feature_list: List[np.ndarray] = []
        for dimg in depth_maps:
            dimg_p, fg = self._preprocess_depth(dimg)

            # If a view is empty (rare but possible), return zeros for that view
            if int(fg.sum()) == 0:
                feature_list.append(self._empty_feature_vector())
                continue

            feat = self._depth_features(dimg_p, fg)

            # Optional per-view L2 normalization
            if self.l2_normalize:
                feat = self._l2_normalize_vec(feat)

            feature_list.append(feat)

        features = np.asarray(np.vstack(feature_list), dtype=np.float32)

        meta = DepthMetadata(
            preset="DEPTH_42",
            image_size=self.image_size,
            directions=directions,
            linearized_depth=self.linearized_depth,
            rotation_set=self.rotation_set,
            l2_normalized=self.l2_normalize,
        )
        return DepthModelDescriptor(features=features, meta=meta)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess_depth(self, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess depth map:
          - ensure float32
          - resize to (image_size, image_size)
          - foreground mask: depth > 0
          - clip outliers
          - normalize foreground to [0, 1] (background stays 0)

        Returns:
          depth_norm: float32 (H,W) in [0,1] (background 0)
          fg_mask:    bool (H,W)
        """
        depth = np.asarray(depth, dtype=np.float32)

        # Resize (keep range)
        if depth.shape != (self.image_size, self.image_size):
            depth = np.asarray(
                resize(
                    depth,
                    (self.image_size, self.image_size),
                    order=1,  # bilinear
                    preserve_range=True,
                    anti_aliasing=True,
                ),
                dtype=np.float32
            )

        fg = depth > 0.0
        if int(fg.sum()) == 0:
            return depth * 0.0, fg

        vals = depth[fg]

        # Clip extreme depths
        hi = float(np.percentile(vals, self.clip_percentile))
        if hi <= 0.0:
            return depth * 0.0, fg

        depth_clipped = depth.copy()
        depth_clipped[fg] = np.clip(depth_clipped[fg], 0.0, hi)

        # Normalize to [0,1] over foreground
        vmin = float(depth_clipped[fg].min())
        vmax = float(depth_clipped[fg].max())
        if vmax - vmin < 1e-8:
            depth_norm = np.zeros_like(depth_clipped, dtype=np.float32)
            depth_norm[fg] = 1.0
            return depth_norm, fg

        depth_norm = np.zeros_like(depth_clipped, dtype=np.float32)
        depth_norm[fg] = (depth_clipped[fg] - vmin) / (vmax - vmin)
        return depth_norm, fg

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _depth_features(self, depth01: np.ndarray, fg: np.ndarray) -> np.ndarray:
        """
        Per-view feature vector:
          1) depth histogram (foreground only)
          2) gradient magnitude histogram (foreground only)
          3) scalar depth stats (foreground only)
        """
        vals = depth01[fg]

        # (1) Depth histogram
        depth_hist, _ = np.histogram(
            vals, bins=self.depth_hist_bins, range=(0.0, 1.0), density=False
        )
        depth_hist = depth_hist.astype(np.float32)
        depth_hist /= max(float(depth_hist.sum()), 1.0)

        # (2) Gradient magnitude histogram
        gy, gx = np.gradient(depth01)
        gmag = np.sqrt(gx * gx + gy * gy)
        gvals = gmag[fg]
        gmax = float(np.percentile(gvals, 99.0)) if gvals.size > 0 else 1.0
        gmax = max(gmax, 1e-6)
        gvals_c = np.clip(gvals / gmax, 0.0, 1.0)

        grad_hist, _ = np.histogram(
            gvals_c, bins=self.grad_hist_bins, range=(0.0, 1.0), density=False
        )
        grad_hist = grad_hist.astype(np.float32)
        grad_hist /= max(float(grad_hist.sum()), 1.0)

        # (3) Scalar stats
        mean = float(vals.mean())
        std = float(vals.std())
        vmin = float(vals.min())
        vmax = float(vals.max())
        p10 = float(np.percentile(vals, 10.0))
        p25 = float(np.percentile(vals, 25.0))
        p50 = float(np.percentile(vals, 50.0))
        p75 = float(np.percentile(vals, 75.0))
        p90 = float(np.percentile(vals, 90.0))

        stats = np.array([mean, std, vmin, vmax, p10, p25, p50, p75, p90], dtype=np.float32)
        return np.asarray(np.concatenate([depth_hist, grad_hist, stats], axis=0), dtype=np.float32)

    def _empty_feature_vector(self) -> np.ndarray:
        dim = self.depth_hist_bins + self.grad_hist_bins + 9
        return np.zeros((dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _l2_normalize_vec(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n <= self.l2_eps:
            return v
        return np.asarray(v / n, dtype=np.float32)
