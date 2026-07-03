"""Basis choices for projected Langevin sampling.

OrthonormalBasis is the main-paper Nyström/KL eigenbasis. InducingPointBasis is
the non-orthonormal inducing-point variant used in generalized experiments.
"""

from projected_langevin_sampling.basis.inducing_point import InducingPointBasis
from projected_langevin_sampling.basis.orthonormal import OrthonormalBasis

__all__ = [
    "InducingPointBasis",
    "OrthonormalBasis",
]
