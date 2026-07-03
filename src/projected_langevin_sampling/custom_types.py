from typing import Union

from projected_langevin_sampling.conformalise.base import ConformaliseBase
from projected_langevin_sampling.conformalise.pls import ConformalisePLS
from projected_langevin_sampling.gaussian_process.exact_gp import ExactGP
from projected_langevin_sampling.gaussian_process.svgp import SVGP
from projected_langevin_sampling.projected_langevin_sampling import PLS
from projected_langevin_sampling.temper.base import TemperBase
from projected_langevin_sampling.temper.pls import TemperPLS

MODEL_TYPE = Union[ExactGP, SVGP, PLS, TemperBase, ConformaliseBase]
GP_TYPE = Union[ExactGP, SVGP]
PLS_TYPE = Union[PLS, ConformalisePLS, TemperPLS]
