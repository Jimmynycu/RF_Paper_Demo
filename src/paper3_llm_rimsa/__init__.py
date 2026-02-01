"""
Paper 3: LLM-RIMSA - Large Language Models driven Reconfigurable 
         Intelligent Metasurface Antenna Systems

Implementation of the LLM-based framework for RIMSA beamforming control.

Key Components:
- RIMSAModel: Reconfigurable Intelligent Metasurface Antenna model
- ChannelModel: Wireless channel with LoS and NLoS components
- LLMBackbone: GPT-2 based transformer for RIMSA control
- Trainer: Training and inference pipeline
"""

from .rimsa_model import RIMSASystem, RIMSAConfig
from .channel_model import ChannelGenerator, RicianChannel
from .llm_backbone import LLMRIMSAModel, LLMConfig
from .trainer import LLMRIMSATrainer, TrainingConfig

__all__ = [
    'RIMSASystem',
    'RIMSAConfig',
    'ChannelGenerator',
    'RicianChannel',
    'LLMRIMSAModel',
    'LLMConfig',
    'LLMRIMSATrainer',
    'TrainingConfig'
]
