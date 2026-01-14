# core/lfd_descriptor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from scipy.fft import fft
from skimage.measure import find_contours
from skimage.transform import resize

import mahotas  # <-- stable Zernike implementation

from ..mesh.mesh_loader import Mesh
from ..mesh.renderer import Renderer
from ..mesh.view_generator import CameraPose, ViewGenerator


@dataclass(frozen=True)
class LFDMetadata:
    """
    Metadata needed by SimilarityEngine (so it doesn't guess).
    """
    preset: Literal["LFD_10"]
    image_size: int
    ring_start: int
    ring_size: int
    # Useful for debugging / later extensions:
    directions: np.ndarray  # (num_views, 3) float32


@dataclass(frozen=True)
class LFDModelDescriptor:
    """
    Final descriptor for one 3D model using LFD.
    features: (num_views, feature_dim)
    meta:     LFDMetadata (preset, ordering, ring info, directions)
    """
    features: np.ndarray
    meta: LFDMetadata


class LFDDescriptor:
    """
    Light Field Descriptor (LFD) using silhouette-based 2D descriptors.

    Pipeline:
      Mesh
        → ViewGenerator (LFD_10)
        → Renderer (silhouette)
        → Preprocessing (binarize, crop, resize)
        → Zernike moments (mahotas) + Fourier boundary descriptor
        → Model descriptor (per-view features + metadata)
    """

    def __init__(
        self,
        renderer: Renderer,
        image_size: int = 128,
        zernike_radius: Optional[int] = None,
        zernike_degree: int = 8,
        fourier_coeffs: int = 10,
        view_radius: float = 2.5,
    ) -> None:
        """
        image_size:
            Target silhouette resolution (default 128x128).

        zernike_radius:
            Radius for Zernike unit disk. If None, it defaults to image_size//2.

        zernike_degree:
            Maximum Zernike degree (controls Zernike feature length).
            NOTE: In mahotas, "degree" is the maximum order.

        fourier_coeffs:
            Number of low-frequency Fourier coefficients kept from the boundary
            (DC removed). Magnitudes are used.

        view_radius:
            Camera radius passed to ViewGenerator (kept here for reproducibility).
        """
        self.renderer = renderer
        self.image_size = int(image_size)
        self.zernike_radius = int(zernike_radius) if zernike_radius is not None else int(self.image_size // 2)
        self.zernike_degree = int(zernike_degree)
        self.fourier_coeffs = int(fourier_coeffs)
        self.view_radius = float(view_radius)

        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.zernike_radius <= 0:
            raise ValueError("zernike_radius must be positive.")
        if self.zernike_degree < 0:
            raise ValueError("zernike_degree must be >= 0.")
        if self.fourier_coeffs <= 0:
            raise ValueError("fourier_coeffs must be positive.")

        self.view_generator = ViewGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, mesh: Mesh) -> LFDModelDescriptor:
        """
        Compute LFD descriptor for one mesh.

        Returns:
            LFDModelDescriptor with:
              - features: (10, feature_dim)
              - meta: preset, ring indexing info, directions
        """
        poses = self.view_generator.generate(preset="LFD_10", radius=self.view_radius)

        # Extract directions (ordered) for metadata
        directions = np.stack([p.direction for p in poses], axis=0).astype(np.float32)

        # LFD_10 ordering conventions from ViewGenerator:
        # index 0: top
        # index 1: bottom
        # index 2..9: ring (8 views) in increasing azimuth
        ring_start, ring_size = 2, 8

        silhouettes = self.renderer.render_views(mesh, poses, mode="silhouette")

        feature_list: List[np.ndarray] = []
        for mask in silhouettes:
            mask_p = self._preprocess_mask(mask)
            zernike = self._zernike_features(mask_p)
            fourier = self._fourier_boundary_features(mask_p)
            feature = np.concatenate([zernike, fourier], axis=0)
            feature_list.append(feature)

        features = np.vstack(feature_list).astype(np.float32)

        meta = LFDMetadata(
            preset="LFD_10",
            image_size=self.image_size,
            ring_start=ring_start,
            ring_size=ring_size,
            directions=directions,
        )

        return LFDModelDescriptor(features=features, meta=meta)

    # ------------------------------------------------------------------
    # Silhouette preprocessing
    # ------------------------------------------------------------------
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess silhouette:
          - binarize
          - crop tight bounding box
          - resize to fixed resolution

        Returns:
          uint8 image of shape (image_size, image_size) with values 0/1.
        """
        mask = (np.asarray(mask) > 0).astype(np.uint8)

        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            raise ValueError("Empty silhouette encountered (no foreground pixels).")

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = mask[y0:y1, x0:x1]

        resized = resize(
            cropped,
            (self.image_size, self.image_size),
            order=0,                 # nearest neighbor to preserve binary edges
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

        resized = (resized > 0).astype(np.uint8)
        return resized

    # ------------------------------------------------------------------
    # Zernike moments (mahotas)
    # ------------------------------------------------------------------
    def _zernike_features(self, mask01: np.ndarray) -> np.ndarray:
        """
        Compute Zernike moments on the silhouette using mahotas.

        mahotas.features.zernike_moments(image, radius, degree) expects:
          - a 2D array (binary or grayscale)
          - radius in pixels
          - degree (max order)

        We use a centered disk implicitly (mahotas uses the image center).
        We also enforce radius <= min(h,w)//2 - 1 for safety.
        """
        img = (mask01 > 0).astype(np.uint8)

        h, w = img.shape
        max_r = max(1, min(h, w) // 2 - 1)
        radius = min(int(self.zernike_radius), int(max_r))

        # mahotas returns a 1D vector of Zernike moments (real-valued magnitudes)
        z = mahotas.features.zernike_moments(img, radius, self.zernike_degree)
        return np.asarray(z, dtype=np.float32)

    # ------------------------------------------------------------------
    # Fourier boundary descriptor
    # ------------------------------------------------------------------
    def _fourier_boundary_features(self, mask01: np.ndarray) -> np.ndarray:
        """
        Compute Fourier descriptors from the largest silhouette contour.

        Steps:
          - find contours at level 0.5
          - choose the longest contour
          - represent as complex sequence x + i y
          - FFT
          - drop DC component
          - keep first `fourier_coeffs` magnitudes
        """
        contours = find_contours(mask01, level=0.5)
        if not contours:
            raise ValueError("No contour found in silhouette.")

        contour = max(contours, key=lambda c: c.shape[0])

        # contour points are (row=y, col=x). Convert to complex x + i y.
        complex_contour = contour[:, 1] + 1j * contour[:, 0]

        coeffs = fft(complex_contour)

        # Remove DC and keep low frequencies
        coeffs = coeffs[1 : self.fourier_coeffs + 1]

        return np.abs(coeffs).astype(np.float32)
