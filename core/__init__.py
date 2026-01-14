# core/__init__.py
from .descriptors import (
    DepthBufferDescriptor, DepthModelDescriptor, DepthMetadata,
    LFDDescriptor, LFDModelDescriptor, LFDMetadata
)
from .mesh import (
    Mesh, MeshLoader, MeshNormalizer, Renderer, ViewGenerator, CameraPose
)
from .similarity import SimilarityEngine, SimilarityResult

__all__ = [
    # Descriptors
    "DepthBufferDescriptor", "DepthModelDescriptor", "DepthMetadata",
    "LFDDescriptor", "LFDModelDescriptor", "LFDMetadata",
    # Mesh operations
    "Mesh", "MeshLoader", "MeshNormalizer", "Renderer", "ViewGenerator", "CameraPose",
    # Similarity
    "SimilarityEngine", "SimilarityResult",
]