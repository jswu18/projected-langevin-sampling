from typing import Union

from src.conformalise.base import ConformaliseBase
from src.conformalise.pls import ConformalisePLS
from src.gaussian_process.exact_gp import ExactGP
from src.gaussian_process.svgp import SVGP
from src.projected_langevin_sampling.projected_langevin_sampling import PLS
from src.temper.base import TemperBase
from src.temper.pls import TemperPLS

MODEL_TYPE = Union[ExactGP, SVGP, PLS, TemperBase, ConformaliseBase]
GP_TYPE = Union[ExactGP, SVGP]
PLS_TYPE = Union[PLS, ConformalisePLS, TemperPLS]
