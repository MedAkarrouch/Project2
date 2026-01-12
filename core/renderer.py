# core/renderer.py
from __future__ import annotations

from typing import List, Literal, Optional, Tuple
import os

import numpy as np

# Must be set before importing OpenGL
os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import glfw  # replaces GLUT
from OpenGL.GL import *
from OpenGL.GLU import *

from .mesh_loader import Mesh
from .view_generator import CameraPose


class Renderer:
    """
    Off-screen renderer using PyOpenGL + GLFW (hidden window) + FBO.

    Responsibilities:
      - Render silhouette mask (binary image) from a mesh and camera pose.
      - Render depth-buffer (depth map) from a mesh and camera pose.

    Notes:
      - No GLUT dependency (GLFW is far more reliable on Windows).
      - We render into an FBO and read back pixels via glReadPixels.
      - Depth: background is set to 0; foreground is depth (linearized or raw depth buffer).
    """

    _glfw_initialized: bool = False
    _glfw_window_count: int = 0

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        fov_y_deg: float = 60.0,
        near: float = 0.1,
        far: float = 10.0,
        projection: Literal["perspective", "orthographic"] = "perspective",
        ortho_scale: float = 1.2,
        # NEW: better depth precision (important when linearizing depth)
        depth_bits: Literal[16, 24, 32] = 24,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fov_y_deg = float(fov_y_deg)
        self.near = float(near)
        self.far = float(far)
        self.projection = projection
        self.ortho_scale = float(ortho_scale)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be positive.")
        if not (0.0 < self.near < self.far):
            raise ValueError("Require 0 < near < far.")

        if depth_bits not in (16, 24, 32):
            raise ValueError("depth_bits must be one of {16,24,32}.")
        self.depth_bits = int(depth_bits)

        # Context + GL resources
        self._window: Optional[object] = None
        self._fbo: Optional[int] = None
        self._color_tex: Optional[int] = None
        self._depth_rb: Optional[int] = None

        # Mesh GPU cache
        self._cached_mesh_key: Optional[Tuple[int, int]] = None
        self._vbo: Optional[int] = None
        self._ebo: Optional[int] = None
        self._index_count: int = 0

        self._init_glfw_context()
        self._init_fbo()

        # Basic OpenGL state
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)

        # NEW: make depth writes deterministic
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)

    # -------------------------
    # Public API
    # -------------------------
    def render_silhouette(self, mesh: Mesh, pose: CameraPose) -> np.ndarray:
        """Render a silhouette mask (H x W), uint8 with values {0,255}."""
        self._ensure_context_current()
        self._ensure_mesh_uploaded(mesh)
        self._bind_fbo()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._set_camera(pose)

        glColor3f(1.0, 1.0, 1.0)
        self._draw_mesh()

        rgb = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(rgb, dtype=np.uint8).reshape(self.height, self.width, 3)
        img = np.flipud(img)

        mask = (img[:, :, 0] > 0) | (img[:, :, 1] > 0) | (img[:, :, 2] > 0)
        out = (mask.astype(np.uint8) * 255)

        self._unbind_fbo()
        return out

    def render_depth(self, mesh: Mesh, pose: CameraPose, linearize: bool = True) -> np.ndarray:
        """
        Render a depth map (H x W), float32.
        Background pixels become 0.
        Foreground pixels are depth values (linearized if requested).
        """
        self._ensure_context_current()
        self._ensure_mesh_uploaded(mesh)
        self._bind_fbo()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._set_camera(pose)

        glColor3f(1.0, 1.0, 1.0)
        self._draw_mesh()

        depth = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT)
        z = np.frombuffer(depth, dtype=np.float32).reshape(self.height, self.width)
        z = np.flipud(z)

        # NEW: more robust background detection
        bg = z >= 1.0 - 1e-7

        if linearize:
            n = self.near
            f = self.far
            # Convert depth buffer value [0,1] -> NDC [-1,1]
            z_ndc = 2.0 * z - 1.0
            denom = (f + n - z_ndc * (f - n))
            denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
            linear = (2.0 * n * f) / denom
            depth_img = linear.astype(np.float32)
        else:
            depth_img = z.astype(np.float32)

        depth_img[bg] = 0.0

        self._unbind_fbo()
        return depth_img

    def render_views(
        self,
        mesh: Mesh,
        poses: List[CameraPose],
        mode: Literal["silhouette", "depth"] = "silhouette",
        linearize_depth: bool = True,
    ) -> List[np.ndarray]:
        outputs: List[np.ndarray] = []
        for p in poses:
            if mode == "silhouette":
                outputs.append(self.render_silhouette(mesh, p))
            elif mode == "depth":
                outputs.append(self.render_depth(mesh, p, linearize=linearize_depth))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return outputs

    def close(self) -> None:
        """Free GL resources and destroy the GLFW window."""
        try:
            self._ensure_context_current()
            self._delete_mesh_buffers()
            self._delete_fbo()
        finally:
            if self._window is not None:
                try:
                    glfw.destroy_window(self._window)
                except Exception:
                    pass
                self._window = None
                Renderer._glfw_window_count = max(0, Renderer._glfw_window_count - 1)

            if Renderer._glfw_initialized and Renderer._glfw_window_count == 0:
                try:
                    glfw.terminate()
                except Exception:
                    pass
                Renderer._glfw_initialized = False

    # -------------------------
    # GLFW context init
    # -------------------------
    def _init_glfw_context(self) -> None:
        if not Renderer._glfw_initialized:
            if not glfw.init():
                raise RuntimeError(
                    "glfw.init() failed. On Windows this usually means missing OpenGL support or a bad install."
                )
            Renderer._glfw_initialized = True

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        # Compatibility context for fixed pipeline (gluLookAt, glMatrixMode, etc.)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        win = glfw.create_window(self.width, self.height, "OffscreenRenderer", None, None)
        if not win:
            raise RuntimeError("glfw.create_window() failed (could not create OpenGL context).")

        self._window = win
        Renderer._glfw_window_count += 1

        glfw.make_context_current(self._window)

        try:
            glfw.swap_interval(0)
        except Exception:
            pass

    def _ensure_context_current(self) -> None:
        if self._window is None:
            raise RuntimeError("Renderer window/context is not initialized.")
        glfw.make_context_current(self._window)

    # -------------------------
    # FBO init
    # -------------------------
    def _init_fbo(self) -> None:
        # Color texture
        self._color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._color_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # GL_CLAMP is deprecated; use CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Depth renderbuffer (choose precision)
        self._depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rb)
        if self.depth_bits == 16:
            depth_format = GL_DEPTH_COMPONENT16
        elif self.depth_bits == 24:
            depth_format = GL_DEPTH_COMPONENT24
        else:
            # Not supported on every driver, but often works
            depth_format = GL_DEPTH_COMPONENT32
        glRenderbufferStorage(GL_RENDERBUFFER, depth_format, self.width, self.height)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        # FBO
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._color_tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depth_rb)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete. Status: {hex(status)}")

    def _bind_fbo(self) -> None:
        if self._fbo is None:
            raise RuntimeError("FBO not initialized.")
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.width, self.height)

    @staticmethod
    def _unbind_fbo() -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _delete_fbo(self) -> None:
        if self._fbo is not None:
            glDeleteFramebuffers(1, [self._fbo])
            self._fbo = None
        if self._depth_rb is not None:
            glDeleteRenderbuffers(1, [self._depth_rb])
            self._depth_rb = None
        if self._color_tex is not None:
            glDeleteTextures(1, [self._color_tex])
            self._color_tex = None

    # -------------------------
    # Camera
    # -------------------------
    def _set_camera(self, pose: CameraPose) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = float(self.width) / float(self.height)

        if self.projection == "perspective":
            gluPerspective(self.fov_y_deg, aspect, self.near, self.far)
        elif self.projection == "orthographic":
            s = self.ortho_scale
            glOrtho(-s * aspect, s * aspect, -s, s, self.near, self.far)
        else:
            raise ValueError(f"Unknown projection: {self.projection}")

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        eye = np.asarray(pose.eye, dtype=np.float32).reshape(3)
        target = np.asarray(pose.target, dtype=np.float32).reshape(3)
        up = np.asarray(pose.up, dtype=np.float32).reshape(3)

        gluLookAt(
            float(eye[0]), float(eye[1]), float(eye[2]),
            float(target[0]), float(target[1]), float(target[2]),
            float(up[0]), float(up[1]), float(up[2]),
        )

    # -------------------------
    # Mesh upload + draw
    # -------------------------
    def _ensure_mesh_uploaded(self, mesh: Mesh) -> None:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.int32)

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("mesh.vertices must have shape (N,3)")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("mesh.faces must have shape (M,3) (triangles)")

        key = (v.__array_interface__["data"][0], f.__array_interface__["data"][0])
        if self._cached_mesh_key == key and self._vbo is not None and self._ebo is not None:
            return

        self._delete_mesh_buffers()

        self._vbo = glGenBuffers(1)
        self._ebo = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v, GL_STATIC_DRAW)

        indices = f.astype(np.uint32).ravel()
        self._index_count = int(indices.size)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        self._cached_mesh_key = key

    def _draw_mesh(self) -> None:
        if self._vbo is None or self._ebo is None:
            raise RuntimeError("Mesh buffers not uploaded.")

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, None)

        glDisableClientState(GL_VERTEX_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glFlush()

    def _delete_mesh_buffers(self) -> None:
        if self._vbo is not None:
            glDeleteBuffers(1, [self._vbo])
            self._vbo = None
        if self._ebo is not None:
            glDeleteBuffers(1, [self._ebo])
            self._ebo = None
        self._cached_mesh_key = None
        self._index_count = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
