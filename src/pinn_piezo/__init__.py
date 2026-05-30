"""PINN piezoelectricity simulation package.

This package contains the source code associated with the paper on
Physics-Informed Neural Networks applied to a 2D piezoelectric beam.
Two formulations are provided:

* ``pinn_piezo.indirect``: trains the indirect (voltage-driven) PINN.
* ``pinn_piezo.direct``:   trains the direct (force-driven) PINN.

Shared utilities (geometry, materials, plotting and evaluation) live at
the top level of the package.
"""

from . import config
from . import materials
from . import geometry
from . import plotting
from . import evaluation

__all__ = ["config", "materials", "geometry", "plotting", "evaluation"]
