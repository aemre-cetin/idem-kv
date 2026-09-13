"""
idempotent-kv: Universal Zero-Copy In-Place Memory Compactor for Transformer & SSM/Mamba.

Protected under U.S. Patent Application Nos. 64/148,668, 64/149,520 ("Patent Pending").
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
"""

__version__ = "0.3.0"
__author__ = "Dr. A. Emre ÇETİN"
__patent__ = "U.S. Patent Application No. 64/148,668, 64/149,520, 64/152,256 (Patent Pending)"

from .kernel import compact_kv_cache_inplace
from .compactor import InplaceKVCompactor, AutoContextEngine, ContextType
from .tarski_state import TarskiStateCompactor
from .integrations import VLLMInplaceCompactionHook, SGLangInplaceCompactionHook
from .perm_kv import PermRankKVCompactor, compute_multihead_borda_ranks
from .saliency import (
    PermSaliencyScorer,
    ChebyshevSpectralSaliencyScorer,
    HybridPermSpectralScorer,
)
from .young_compactor import YoungFactorizedCompactor
from .compactor_2d import TwoDimensionalIdempotentCompactor
from .hysteresis import HysteresisTokenStabilizer

__all__ = [
    "compact_kv_cache_inplace",
    "InplaceKVCompactor",
    "AutoContextEngine",
    "ContextType",
    "TarskiStateCompactor",
    "VLLMInplaceCompactionHook",
    "SGLangInplaceCompactionHook",
    "PermRankKVCompactor",
    "compute_multihead_borda_ranks",
    "PermSaliencyScorer",
    "ChebyshevSpectralSaliencyScorer",
    "HybridPermSpectralScorer",
    "YoungFactorizedCompactor",
    "TwoDimensionalIdempotentCompactor",
    "HysteresisTokenStabilizer",
    "__version__",
    "__patent__",
]
