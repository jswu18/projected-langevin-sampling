from typing import Union

from experiments.data import Data
from src.conformalise import ConformaliseBase, ConformaliseGP, ConformaliseGradientFlow
from src.gps import svGP
from src.gradient_flows import ProjectedWassersteinGradientFlow
from src.temper import TemperBase, TemperGP, TemperGradientFlow


def construct_tempered_model(
    model: Union[svGP, ProjectedWassersteinGradientFlow],
    data: Data,
) -> TemperBase:
    if isinstance(model, svGP):
        return TemperGP(
            gp=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    elif isinstance(model, ProjectedWassersteinGradientFlow):
        return TemperGradientFlow(
            gradient_flow=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def construct_conformalised_model(
    model: Union[svGP, ProjectedWassersteinGradientFlow],
    data: Data,
) -> ConformaliseBase:
    if isinstance(model, svGP):
        return ConformaliseGP(
            gp=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    elif isinstance(model, ProjectedWassersteinGradientFlow):
        return ConformaliseGradientFlow(
            gradient_flow=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    else:
        raise ValueError(f"Model type {type(model)} not supported")
