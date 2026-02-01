"""
Paper 1: Chu-Limit-Guided Decomposition-Based Multiobjective Large-Scale 
         Optimization for Generative Broadband Electrically Small Antenna Design

Implementation of the MOEA/D optimization framework guided by the Chu limit
for designing electrically small antennas (ESAs).

Key Components:
- ChuLimitCalculator: Computes theoretical bandwidth limits
- ESAConfiguration: 3D antenna structure representation
- MOEAD: Multi-objective evolutionary algorithm
- AntennaEvaluator: Simulates antenna performance (simplified EM model)
"""

from .chu_limit import ChuLimitCalculator
from .esa_configuration import ESAConfiguration  
from .moead import MOEAD
from .evaluator import AntennaEvaluator

__all__ = [
    'ChuLimitCalculator',
    'ESAConfiguration', 
    'MOEAD',
    'AntennaEvaluator'
]
