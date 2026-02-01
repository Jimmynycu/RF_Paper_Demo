"""
FSS Designer - Main Training and Inference Class

Implements the complete PINN-based FSS inverse design pipeline:
1. Setup: Create network, loss function, optimizer
2. Training: Optimize shape network to minimize physics + design loss
3. Inference: Generate FSS patterns for given design goals
4. Validation: Verify design with full-wave simulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from tqdm import tqdm
import time

try:
    from .shape_network import ShapeNetwork, create_coordinate_grid
    from .mode_matching import FSSParameters, ModeMatchingSolver, RealModeMatchingSolver
    from .pinn_loss import PINNLoss, DesignGoal
except ImportError:
    from shape_network import ShapeNetwork, create_coordinate_grid
    from mode_matching import FSSParameters, ModeMatchingSolver, RealModeMatchingSolver
    from pinn_loss import PINNLoss, DesignGoal


@dataclass
class TrainingConfig:
    """Configuration for PINN training."""
    max_steps: int = 10000
    learning_rate: float = 1e-3
    grid_resolution: int = 64
    physics_weight: float = 1.0
    design_weight: float = 10.0
    binarization_weight: float = 0.01
    binarization_start_step: int = 5000
    log_interval: int = 1000
    checkpoint_interval: int = 2000
    use_scheduler: bool = True
    scheduler_patience: int = 1000
    early_stop_patience: int = 3000
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass  
class DesignResult:
    """Result from FSS design process."""
    g1_pattern: np.ndarray  # Binary diaphragm 1 pattern
    g2_pattern: np.ndarray  # Binary diaphragm 2 pattern
    g1_continuous: np.ndarray  # Continuous shape before binarization
    g2_continuous: np.ndarray  # Continuous shape before binarization
    training_history: Dict[str, List[float]]
    design_goal: DesignGoal
    achieved_s21: Dict[float, float]  # Frequency -> S21 mapping
    design_error: float  # Error metric
    training_time: float  # Seconds


class FSSDesigner:
    """
    FSS Inverse Design using Physics-Informed Neural Networks.
    
    Main class for training and using the PINN to design FSS structures.
    
    Example usage:
        designer = FSSDesigner(fss_params, design_goal)
        result = designer.train()
        designer.visualize_result(result)
    """
    
    def __init__(self,
                 fss_params: FSSParameters,
                 design_goal: DesignGoal,
                 training_config: Optional[TrainingConfig] = None):
        """
        Initialize FSS designer.
        
        Args:
            fss_params: Physical parameters of FSS structure
            design_goal: Target S-parameter specification
            training_config: Training hyperparameters
        """
        self.fss_params = fss_params
        self.design_goal = design_goal
        self.config = training_config or TrainingConfig()
        self.device = self.config.device
        
        # Create shape network
        self.network = ShapeNetwork(
            hidden_layers=3,
            neurons_per_layer=32,
            activation='tanh'
        ).to(self.device)
        
        # Create coordinate grid
        self.coord_grid = create_coordinate_grid(
            resolution=self.config.grid_resolution,
            device=self.device
        )
        
        # Create loss function
        self.loss_fn = PINNLoss(
            params=fss_params,
            design_goal=design_goal,
            physics_weight=self.config.physics_weight,
            design_weight=self.config.design_weight,
            use_simplified_physics=True,
            device=self.device
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Learning rate scheduler
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.scheduler_patience // self.config.log_interval,
                min_lr=1e-6
            )
        else:
            self.scheduler = None
            
        # Training history
        self.history = {
            'step': [],
            'total_loss': [],
            'physics_loss': [],
            'design_loss': [],
            'learning_rate': []
        }
        
    def train(self, verbose: bool = True) -> DesignResult:
        """
        Train the PINN to design FSS.
        
        Args:
            verbose: Print training progress
            
        Returns:
            DesignResult with trained patterns and metrics
        """
        start_time = time.time()
        
        if verbose:
            print("=" * 60)
            print("PINN Training for FSS Inverse Design")
            print("=" * 60)
            print(f"Device: {self.device}")
            print(f"Grid resolution: {self.config.grid_resolution}")
            print(f"Max steps: {self.config.max_steps}")
            print(f"Target frequencies: {self.design_goal.frequencies.cpu().numpy() / 1e9} GHz")
            print("=" * 60)
            
        best_loss = float('inf')
        best_state = None
        no_improve_count = 0
        
        pbar = tqdm(range(self.config.max_steps), disable=not verbose)
        
        for step in pbar:
            self.optimizer.zero_grad()
            
            # Forward pass
            g1, g2 = self.network(self.coord_grid)
            
            # Compute loss
            include_binary = step >= self.config.binarization_start_step
            losses = self.loss_fn(g1, g2, include_binary_loss=include_binary)
            
            # Add binarization loss if applicable
            if include_binary:
                losses['total'] = losses['total'] + self.config.binarization_weight * losses['binary']
                
            # Backward pass
            losses['total'].backward()
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None and step % self.config.log_interval == 0:
                self.scheduler.step(losses['total'])
                
            # Track best model
            if losses['total'].item() < best_loss:
                best_loss = losses['total'].item()
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Early stopping
            if no_improve_count > self.config.early_stop_patience:
                if verbose:
                    print(f"\nEarly stopping at step {step}")
                break
                
            # Logging
            if step % self.config.log_interval == 0:
                self.history['step'].append(step)
                self.history['total_loss'].append(losses['total'].item())
                self.history['physics_loss'].append(losses['physics'].item())
                self.history['design_loss'].append(losses['design'].item())
                self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'physics': f"{losses['physics'].item():.4f}",
                    'design': f"{losses['design'].item():.4f}"
                })
                
        # Load best model
        if best_state is not None:
            self.network.load_state_dict(best_state)
            
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {training_time:.1f} seconds")
            print(f"Best loss: {best_loss:.6f}")
            
        # Generate final result
        return self._create_result(training_time)
    
    def _create_result(self, training_time: float) -> DesignResult:
        """Create DesignResult from trained network."""
        # Get patterns
        with torch.no_grad():
            g1, g2 = self.network(self.coord_grid)
            
        res = self.config.grid_resolution
        g1_continuous = g1.cpu().numpy().reshape(res, res)
        g2_continuous = g2.cpu().numpy().reshape(res, res)
        
        # Binarize
        g1_binary = (g1_continuous > 0.5).astype(float)
        g2_binary = (g2_continuous > 0.5).astype(float)
        
        # Compute achieved S21
        achieved_s21 = {}
        for freq in self.design_goal.frequencies:
            s21 = self.loss_fn.solver.compute_s21_simplified(
                g1, g2, freq.item()
            )
            achieved_s21[freq.item()] = s21.item()
            
        # Compute error
        errors = []
        for freq, target in zip(self.design_goal.frequencies, self.design_goal.s21_targets):
            errors.append(abs(achieved_s21[freq.item()] - target.item()))
        design_error = np.mean(errors)
        
        return DesignResult(
            g1_pattern=g1_binary,
            g2_pattern=g2_binary,
            g1_continuous=g1_continuous,
            g2_continuous=g2_continuous,
            training_history=self.history,
            design_goal=self.design_goal,
            achieved_s21=achieved_s21,
            design_error=design_error,
            training_time=training_time
        )
    
    def visualize_result(self, result: DesignResult, save_path: Optional[str] = None):
        """
        Visualize the design result.
        
        Args:
            result: DesignResult to visualize
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            
            # Continuous patterns
            im1 = axes[0, 0].imshow(result.g1_continuous, cmap='RdYlBu', 
                                    origin='lower', extent=[0, 1, 0, 1])
            axes[0, 0].set_title('Diaphragm 1 (Continuous)')
            axes[0, 0].set_xlabel('x/a')
            axes[0, 0].set_ylabel('y/a')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(result.g2_continuous, cmap='RdYlBu',
                                    origin='lower', extent=[0, 1, 0, 1])
            axes[0, 1].set_title('Diaphragm 2 (Continuous)')
            axes[0, 1].set_xlabel('x/a')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Binary patterns
            axes[0, 2].imshow(result.g1_pattern, cmap='binary',
                             origin='lower', extent=[0, 1, 0, 1])
            axes[0, 2].set_title('Diaphragm 1 (Binary)')
            axes[0, 2].set_xlabel('x/a')
            
            # Training history
            axes[1, 0].plot(result.training_history['step'], 
                          result.training_history['total_loss'], 'b-', label='Total')
            axes[1, 0].plot(result.training_history['step'],
                          result.training_history['physics_loss'], 'g--', label='Physics')
            axes[1, 0].plot(result.training_history['step'],
                          result.training_history['design_loss'], 'r:', label='Design')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training History')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # S21 comparison
            freqs_ghz = result.design_goal.frequencies.cpu().numpy() / 1e9
            targets = result.design_goal.s21_targets.cpu().numpy()
            achieved = [result.achieved_s21[f * 1e9] for f in freqs_ghz]
            
            x = np.arange(len(freqs_ghz))
            width = 0.35
            
            bars1 = axes[1, 1].bar(x - width/2, targets, width, label='Target', color='blue', alpha=0.7)
            bars2 = axes[1, 1].bar(x + width/2, achieved, width, label='Achieved', color='orange', alpha=0.7)
            
            axes[1, 1].set_xlabel('Frequency (GHz)')
            axes[1, 1].set_ylabel('|S₂₁|')
            axes[1, 1].set_title('S21 Performance')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([f'{f:.0f}' for f in freqs_ghz])
            axes[1, 1].legend()
            axes[1, 1].set_ylim([0, 1.1])
            axes[1, 1].grid(True, axis='y', alpha=0.3)
            
            # Binary pattern 2
            axes[1, 2].imshow(result.g2_pattern, cmap='binary',
                             origin='lower', extent=[0, 1, 0, 1])
            axes[1, 2].set_title('Diaphragm 2 (Binary)')
            axes[1, 2].set_xlabel('x/a')
            
            # Add text summary
            summary = f"Design Error: {result.design_error:.4f}\n"
            summary += f"Training Time: {result.training_time:.1f}s"
            fig.text(0.98, 0.02, summary, ha='right', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Figure saved to {save_path}")
                
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
    
    def save_patterns(self, result: DesignResult, path: str):
        """Save FSS patterns to file for fabrication."""
        np.savez(
            path,
            g1_binary=result.g1_pattern,
            g2_binary=result.g2_pattern,
            g1_continuous=result.g1_continuous,
            g2_continuous=result.g2_continuous,
            unit_cell_size=self.fss_params.unit_cell_size,
            resolution=self.config.grid_resolution
        )
        print(f"Patterns saved to {path}")
        
    def export_to_gerber(self, result: DesignResult, path: str):
        """
        Export patterns to Gerber format for PCB fabrication.
        
        Note: This is a placeholder - real implementation would
        use a Gerber generation library.
        """
        print(f"Would export Gerber files to {path}")
        # In a real implementation:
        # - Convert binary pattern to polygons
        # - Generate proper Gerber format with apertures
        # - Include drill files if needed


def run_fss_design(target_freq: float = 15e9,
                   passband: Tuple[float, float] = (12e9, 18e9),
                   training_steps: int = 5000) -> DesignResult:
    """
    Convenience function to run FSS design.
    
    Args:
        target_freq: Frequency to block (Hz)
        passband: Lower and upper passband frequencies
        training_steps: Number of training steps
        
    Returns:
        DesignResult
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup
    fss_params = FSSParameters(
        unit_cell_size=10e-3,
        dielectric_thickness=2e-3,
        relative_permittivity=3.2
    )
    
    design_goal = DesignGoal.bandstop_at_frequency(
        center_freq=target_freq,
        passband_freqs=passband,
        device=device
    )
    
    config = TrainingConfig(
        max_steps=training_steps,
        learning_rate=1e-3,
        grid_resolution=64,
        device=device
    )
    
    # Create designer and train
    designer = FSSDesigner(fss_params, design_goal, config)
    result = designer.train(verbose=True)
    
    return result, designer


if __name__ == "__main__":
    print("=" * 60)
    print("Paper 2: PINN for FSS Inverse Design")
    print("=" * 60)
    
    # Run design
    result, designer = run_fss_design(
        target_freq=15e9,
        passband=(12e9, 18e9),
        training_steps=3000  # Reduced for demo
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("DESIGN RESULTS")
    print("=" * 60)
    
    print(f"Design error: {result.design_error:.4f}")
    print(f"Training time: {result.training_time:.1f} seconds")
    
    print("\nS21 Performance:")
    for freq, s21 in result.achieved_s21.items():
        target_idx = (result.design_goal.frequencies == freq).nonzero()
        if len(target_idx) > 0:
            target = result.design_goal.s21_targets[target_idx[0]].item()
            print(f"  {freq/1e9:.0f} GHz: Achieved={s21:.3f}, Target={target:.3f}")
            
    print(f"\nDiaphragm 1 aperture ratio: {result.g1_pattern.mean():.2%}")
    print(f"Diaphragm 2 aperture ratio: {result.g2_pattern.mean():.2%}")
    
    # Visualize
    designer.visualize_result(result, save_path='fss_design_result.png')
    
    print("\nFSS Design Complete!")
