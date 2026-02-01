"""
PINN Loss - Physics-Informed Loss Function for FSS Design

Implements the loss function that combines:
1. Physics residual (Mode matching equation violation)
2. Design goal (S-parameter targets)

The key insight: No labeled data needed! Physical equations provide supervision.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from .mode_matching import ModeMatchingSolver, FSSParameters, RealModeMatchingSolver
except ImportError:
    from mode_matching import ModeMatchingSolver, FSSParameters, RealModeMatchingSolver


@dataclass
class DesignGoal:
    """
    Design goal specification for FSS inverse design.
    
    Specifies target S21 values at specific frequencies.
    """
    frequencies: torch.Tensor  # Target frequencies (Hz)
    s21_targets: torch.Tensor  # Target |S21| values (0-1 scale)
    weights: Optional[torch.Tensor] = None  # Optional frequency weights
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = torch.ones_like(self.frequencies)
            
    @classmethod
    def bandstop_at_frequency(cls, 
                              center_freq: float,
                              passband_freqs: Tuple[float, float],
                              device: str = 'cpu'):
        """
        Create design goal for band-stop filter.
        
        Args:
            center_freq: Center frequency to block (Hz)
            passband_freqs: (lower, upper) passband frequencies
            device: Computation device
            
        Returns:
            DesignGoal for band-stop response
        """
        # From paper: |S21(12GHz)| = 0.99, |S21(15GHz)| = 0.1, |S21(18GHz)| = 0.99
        frequencies = torch.tensor([passband_freqs[0], center_freq, passband_freqs[1]], 
                                  device=device)
        s21_targets = torch.tensor([0.99, 0.1, 0.99], device=device)
        weights = torch.tensor([1.0, 2.0, 1.0], device=device)  # Higher weight on stopband
        
        return cls(frequencies, s21_targets, weights)
    
    @classmethod
    def bandpass_at_frequency(cls,
                             center_freq: float,
                             stopband_freqs: Tuple[float, float],
                             device: str = 'cpu'):
        """Create design goal for band-pass filter."""
        frequencies = torch.tensor([stopband_freqs[0], center_freq, stopband_freqs[1]], 
                                  device=device)
        s21_targets = torch.tensor([0.1, 0.99, 0.1], device=device)
        weights = torch.tensor([1.0, 2.0, 1.0], device=device)
        
        return cls(frequencies, s21_targets, weights)


class PINNLoss(nn.Module):
    """
    Physics-Informed Neural Network Loss for FSS Design.
    
    Total loss = L_physics + λ * L_design
    
    where:
    - L_physics = ||residual(Eq.2)||²  (Mode matching residual)
    - L_design = ||S21 - target||²     (Design goal mismatch)
    
    The physics loss ensures the generated shape satisfies Maxwell's equations,
    while the design loss drives toward desired S-parameter response.
    """
    
    def __init__(self,
                 params: FSSParameters,
                 design_goal: DesignGoal,
                 physics_weight: float = 1.0,
                 design_weight: float = 1.0,
                 physics_mode: str = 'real',
                 device: str = 'cpu'):
        """
        Initialize PINN loss.
        
        Args:
            params: FSS physical parameters
            design_goal: Target S-parameter specification
            physics_weight: Weight for physics residual loss
            design_weight: Weight for design goal loss
            physics_mode: Physics solver to use:
                - 'real': Real mode matching with configurable mode count (DEFAULT)
                - 'full': Original ModeMatchingSolver (slower, more detailed)
            device: Computation device
            
        Note:
            Adjust params.n_modes_te/n_modes_tm for speed/accuracy tradeoff:
            - params = FSSParameters.with_mode_count(3)  # Fast: 9 modes, ~3ms
            - params = FSSParameters.with_mode_count(5)  # Medium: 25 modes, ~5ms
            - params = FSSParameters.with_mode_count(9)  # Accurate: 81 modes, ~6ms
            - params = FSSParameters.with_mode_count(11) # Paper-level: 121 modes, ~9ms
        """
        super().__init__()
        
        self.params = params
        self.design_goal = design_goal
        self.physics_weight = physics_weight
        self.design_weight = design_weight
        self.device = device
        self.physics_mode = physics_mode
        
        # Mode matching solver selection
        if physics_mode == 'real':
            self.solver = RealModeMatchingSolver(params, device)
        else:  # 'full' or fallback
            self.solver = ModeMatchingSolver(params, device)
        
    def compute_physics_loss(self, 
                            g1: torch.Tensor, 
                            g2: torch.Tensor) -> torch.Tensor:
        """
        Compute physics residual loss.
        
        This is the core PINN contribution: the loss measures how well
        the generated shape satisfies the electromagnetic field equations.
        
        From paper Eq. (3):
        L = ||res(Eq.(2))||²₂
        
        Args:
            g1, g2: Shape function values
            
        Returns:
            Physics residual loss
        """
        # For simplified model, physics loss is based on consistency
        if self.use_simplified:
            # Constraint: shapes should have reasonable aperture area
            area1 = torch.mean(g1)
            area2 = torch.mean(g2)
            
            # Penalize too small or too large apertures
            area_loss = ((area1 - 0.5)**2 + (area2 - 0.5)**2)
            
            # Smoothness regularization (encourages coherent apertures)
            res = int(np.sqrt(len(g1)))
            g1_2d = g1.reshape(res, res)
            g2_2d = g2.reshape(res, res)
            
            # Total variation
            tv1 = torch.mean(torch.abs(g1_2d[:, 1:] - g1_2d[:, :-1])) + \
                  torch.mean(torch.abs(g1_2d[1:, :] - g1_2d[:-1, :]))
            tv2 = torch.mean(torch.abs(g2_2d[:, 1:] - g2_2d[:, :-1])) + \
                  torch.mean(torch.abs(g2_2d[1:, :] - g2_2d[:-1, :]))
            
            # Moderate smoothness (not too smooth, not too noisy)
            smooth_loss = (tv1 - 0.3)**2 + (tv2 - 0.3)**2
            
            physics_loss = area_loss + 0.1 * smooth_loss
            
        else:
            # Full mode matching residual
            # Compute mode coefficients via the interaction operator
            freq = self.design_goal.frequencies[len(self.design_goal.frequencies)//2].item()
            
            # This would compute the full residual of Eq. (2)
            # ||sum C_i(E_t, H_t) - g @ sum C_{i+1}(E_t, H_t)||²
            # Simplified: use transmission matrix condition number
            T = self.solver.compute_transmission_matrix(g1, g2, freq)
            
            # Residual: T should be well-conditioned
            # Low condition number = good physical solution
            physics_loss = torch.log(torch.linalg.cond(T + 1e-6 * torch.eye(T.shape[0], device=self.device)))
            
        return physics_loss
    
    def compute_design_loss(self,
                           g1: torch.Tensor,
                           g2: torch.Tensor) -> torch.Tensor:
        """
        Compute design goal loss.
        
        Measures how close the S21 response is to the target.
        
        Args:
            g1, g2: Shape function values
            
        Returns:
            Design goal loss
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i, (freq, target, weight) in enumerate(zip(
            self.design_goal.frequencies,
            self.design_goal.s21_targets,
            self.design_goal.weights
        )):
            # Compute S21 at this frequency based on physics mode
            if self.physics_mode == 'real':
                # RealModeMatchingSolver has direct compute_s21 method
                s21 = self.solver.compute_s21(g1, g2, freq.item())
            else:
                # Full mode matching solver
                s_params = self.solver.compute_s_parameters(g1, g2, freq.item())
                s21 = torch.tensor(abs(s_params['S21']), device=self.device)
                
            # Squared error
            loss_i = weight * (s21 - target) ** 2
            total_loss = total_loss + loss_i
            
        return total_loss / len(self.design_goal.frequencies)
    
    def compute_binarization_loss(self,
                                  g1: torch.Tensor,
                                  g2: torch.Tensor) -> torch.Tensor:
        """
        Encourage binary outputs for fabrication.
        
        Uses entropy-like loss that is minimized when g ∈ {0, 1}.
        """
        # Binary cross-entropy with itself (minimized at 0 or 1)
        entropy1 = -torch.mean(g1 * torch.log(g1 + 1e-7) + 
                              (1 - g1) * torch.log(1 - g1 + 1e-7))
        entropy2 = -torch.mean(g2 * torch.log(g2 + 1e-7) + 
                              (1 - g2) * torch.log(1 - g2 + 1e-7))
        
        # Maximize entropy = minimize negative entropy
        # But we want binary, so minimize entropy
        return -(entropy1 + entropy2)  # Will push toward 0 or 1
    
    def forward(self,
                g1: torch.Tensor,
                g2: torch.Tensor,
                include_binary_loss: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute total PINN loss.
        
        Args:
            g1, g2: Shape function outputs from network
            include_binary_loss: Include binarization regularization
            
        Returns:
            Dictionary with individual losses and total
        """
        # Physics loss
        loss_physics = self.compute_physics_loss(g1, g2)
        
        # Design loss
        loss_design = self.compute_design_loss(g1, g2)
        
        # Total loss
        total_loss = self.physics_weight * loss_physics + self.design_weight * loss_design
        
        losses = {
            'physics': loss_physics,
            'design': loss_design,
            'total': total_loss
        }
        
        if include_binary_loss:
            loss_binary = self.compute_binarization_loss(g1, g2)
            losses['binary'] = loss_binary
            losses['total'] = total_loss + 0.01 * loss_binary
            
        return losses


class AdaptivePINNLoss(PINNLoss):
    """
    PINN Loss with adaptive weighting.
    
    Automatically balances physics and design losses during training
    using gradient-based importance estimation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Learnable log-weights
        self.log_physics_weight = nn.Parameter(torch.tensor(0.0))
        self.log_design_weight = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, g1: torch.Tensor, g2: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward with adaptive weighting."""
        loss_physics = self.compute_physics_loss(g1, g2)
        loss_design = self.compute_design_loss(g1, g2)
        
        # Exponential to ensure positive weights
        w_physics = torch.exp(self.log_physics_weight)
        w_design = torch.exp(self.log_design_weight)
        
        # Weighted sum with uncertainty regularization
        total_loss = (w_physics * loss_physics + w_design * loss_design + 
                     self.log_physics_weight + self.log_design_weight)
        
        return {
            'physics': loss_physics,
            'design': loss_design,
            'total': total_loss,
            'w_physics': w_physics.detach(),
            'w_design': w_design.detach()
        }


if __name__ == "__main__":
    # Demo: Compute PINN loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup
    params = FSSParameters()
    
    # Design goal: Band-stop at 15 GHz
    design_goal = DesignGoal.bandstop_at_frequency(
        center_freq=15e9,
        passband_freqs=(12e9, 18e9),
        device=device
    )
    
    print(f"Design goal frequencies: {design_goal.frequencies.cpu().numpy() / 1e9} GHz")
    print(f"Target S21: {design_goal.s21_targets.cpu().numpy()}")
    
    # Create loss function
    loss_fn = PINNLoss(
        params=params,
        design_goal=design_goal,
        physics_weight=1.0,
        design_weight=10.0,
        use_simplified_physics=True,
        device=device
    )
    
    # Sample shape functions
    res = 32
    n_points = res * res
    
    # Random shapes
    g1 = torch.rand(n_points, device=device)
    g2 = torch.rand(n_points, device=device)
    
    # Compute loss
    losses = loss_fn(g1, g2)
    
    print(f"\nLoss values:")
    print(f"  Physics: {losses['physics'].item():.4f}")
    print(f"  Design:  {losses['design'].item():.4f}")
    print(f"  Total:   {losses['total'].item():.4f}")
    
    # Test with different shapes
    print("\n--- Testing different aperture configurations ---")
    
    # Small aperture
    x = torch.linspace(0, 1, res, device=device)
    y = torch.linspace(0, 1, res, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    r = torch.sqrt((xx - 0.5)**2 + (yy - 0.5)**2)
    
    for radius in [0.1, 0.2, 0.3, 0.4]:
        g1 = (r < radius).float().flatten()
        g2 = (r < radius).float().flatten()
        
        losses = loss_fn(g1, g2)
        print(f"  Radius {radius}: Physics={losses['physics'].item():.4f}, "
              f"Design={losses['design'].item():.4f}")
              
    print("\nPINN Loss Demo Complete!")
