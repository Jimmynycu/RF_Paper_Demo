"""
Paper 2: Inverse Design of Frequency Selective Surface Using 
         Physics-Informed Neural Networks (PINN)

Implementation of the PINN framework for FSS inverse design.

Key Components:
- ShapeNetwork: FCNN that generates diaphragm patterns g(x,y)
- ModeMatching: Mode matching physics solver
- PINNLoss: Physics-informed loss function
- FSSDesigner: Main training/inference class
"""

from .shape_network import ShapeNetwork
from .mode_matching import ModeMatchingSolver
from .pinn_loss import PINNLoss
from .fss_designer import FSSDesigner

__all__ = [
    'ShapeNetwork',
    'ModeMatchingSolver', 
    'PINNLoss',
    'FSSDesigner'
]
