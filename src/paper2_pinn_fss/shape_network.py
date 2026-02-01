"""
Shape Network - Neural Network for FSS Diaphragm Generation

Implements a Fully Connected Neural Network (FCNN) that takes (x, y) 
coordinates as input and outputs shape functions g_1(x,y) and g_2(x,y)
that define the metal diaphragm patterns.

Output values:
- 0 = PEC (Perfect Electric Conductor)
- 1 = Vacuum (aperture)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class ShapeNetwork(nn.Module):
    """
    Fully Connected Neural Network for generating FSS diaphragm shapes.
    
    Architecture (from paper):
    - Input: (x, y) coordinates normalized to [0, 1]
    - 3 hidden layers with 32 neurons each
    - Tanh activation
    - Output: 2 values (g1, g2) through sigmoid for [0, 1] range
    """
    
    def __init__(self, 
                 hidden_layers: int = 3,
                 neurons_per_layer: int = 32,
                 activation: str = 'tanh'):
        """
        Initialize shape network.
        
        Args:
            hidden_layers: Number of hidden layers
            neurons_per_layer: Neurons in each hidden layer
            activation: Activation function ('tanh', 'relu', 'gelu')
        """
        super().__init__()
        
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        
        # Build network layers
        layers = []
        
        # Input layer: (x, y) -> hidden
        layers.append(nn.Linear(2, neurons_per_layer))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self._get_activation(activation))
            
        # Output layer: hidden -> (g1, g2)
        layers.append(nn.Linear(neurons_per_layer, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.Tanh())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: generate shape functions from coordinates.
        
        Args:
            xy: Tensor of shape (batch, 2) with (x, y) coordinates
            
        Returns:
            g1, g2: Shape functions for the two diaphragms
        """
        output = self.network(xy)
        
        # Apply sigmoid to constrain to [0, 1]
        g = torch.sigmoid(output)
        
        g1 = g[:, 0]  # Shape function for diaphragm 1
        g2 = g[:, 1]  # Shape function for diaphragm 2
        
        return g1, g2
    
    def get_shape_image(self, 
                        resolution: int = 64,
                        device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate shape images for visualization.
        
        Args:
            resolution: Grid resolution (resolution x resolution)
            device: Device for computation
            
        Returns:
            g1_image, g2_image: 2D arrays of shape values
        """
        # Create coordinate grid
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        xy = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
        
        # Evaluate network
        with torch.no_grad():
            g1, g2 = self.forward(xy)
            
        g1_image = g1.cpu().numpy().reshape(resolution, resolution)
        g2_image = g2.cpu().numpy().reshape(resolution, resolution)
        
        return g1_image, g2_image
    
    def binarize_output(self, 
                        g: torch.Tensor, 
                        threshold: float = 0.5) -> torch.Tensor:
        """
        Convert continuous shape to binary (fabrication-ready).
        
        Args:
            g: Continuous shape values
            threshold: Binarization threshold
            
        Returns:
            Binary tensor (0 or 1)
        """
        return (g > threshold).float()
    
    def smooth_binarize(self, 
                        g: torch.Tensor,
                        temperature: float = 10.0) -> torch.Tensor:
        """
        Differentiable approximation of binarization.
        
        Uses sigmoid with high temperature to approximate step function
        while maintaining gradients.
        
        Args:
            g: Continuous shape values
            temperature: Higher = sharper transition
            
        Returns:
            Approximately binary tensor
        """
        return torch.sigmoid(temperature * (g - 0.5))


class SymmetricShapeNetwork(ShapeNetwork):
    """
    Shape network with built-in symmetry constraints.
    
    Many FSS designs have 4-fold rotational symmetry.
    This network generates patterns in 1/4 of the unit cell
    and applies symmetry to produce the full pattern.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with 4-fold symmetry.
        
        Maps all points to first quadrant, evaluates network,
        then applies to get full symmetric pattern.
        """
        x = xy[:, 0]
        y = xy[:, 1]
        
        # Center coordinates (shift to [-0.5, 0.5])
        x_centered = x - 0.5
        y_centered = y - 0.5
        
        # Map to first quadrant (absolute values)
        x_quad = torch.abs(x_centered) / 0.5  # Normalize to [0, 1]
        y_quad = torch.abs(y_centered) / 0.5
        
        xy_quad = torch.stack([x_quad, y_quad], dim=1)
        
        return super().forward(xy_quad)


class ParametricShapeNetwork(nn.Module):
    """
    Shape network conditioned on design parameters.
    
    Allows generating different FSS designs by varying
    input parameters (e.g., target frequency, bandwidth).
    """
    
    def __init__(self,
                 n_design_params: int = 1,
                 hidden_layers: int = 3,
                 neurons_per_layer: int = 64):
        super().__init__()
        
        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(n_design_params, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # Coordinate encoder
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # Combined network
        layers = []
        layers.append(nn.Linear(64, neurons_per_layer))  # 32 + 32
        layers.append(nn.GELU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(neurons_per_layer, 2))
        
        self.combined_network = nn.Sequential(*layers)
        
    def forward(self, 
                xy: torch.Tensor,
                params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with design parameters.
        
        Args:
            xy: Coordinates (batch, 2)
            params: Design parameters (batch, n_params)
            
        Returns:
            g1, g2: Shape functions
        """
        coord_features = self.coord_encoder(xy)
        param_features = self.param_encoder(params)
        
        combined = torch.cat([coord_features, param_features], dim=1)
        output = self.combined_network(combined)
        
        g = torch.sigmoid(output)
        return g[:, 0], g[:, 1]


def create_coordinate_grid(resolution: int = 64,
                          device: str = 'cpu') -> torch.Tensor:
    """
    Create uniform coordinate grid for the unit cell.
    
    Args:
        resolution: Grid size
        device: Computation device
        
    Returns:
        Tensor of shape (resolution^2, 2)
    """
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return xy


if __name__ == "__main__":
    # Demo: Create and test shape network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create network
    net = ShapeNetwork(hidden_layers=3, neurons_per_layer=32)
    net = net.to(device)
    
    print(f"Network parameters: {sum(p.numel() for p in net.parameters())}")
    
    # Generate sample output
    xy = create_coordinate_grid(resolution=64, device=device)
    g1, g2 = net(xy)
    
    print(f"Input shape: {xy.shape}")
    print(f"Output g1 shape: {g1.shape}")
    print(f"g1 range: [{g1.min():.3f}, {g1.max():.3f}]")
    print(f"g2 range: [{g2.min():.3f}, {g2.max():.3f}]")
    
    # Get images
    g1_img, g2_img = net.get_shape_image(resolution=128, device=device)
    
    # Visualize
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        im1 = axes[0].imshow(g1_img, cmap='binary', origin='lower', extent=[0, 1, 0, 1])
        axes[0].set_title('Diaphragm 1: g₁(x,y)')
        axes[0].set_xlabel('x/a')
        axes[0].set_ylabel('y/a')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(g2_img, cmap='binary', origin='lower', extent=[0, 1, 0, 1])
        axes[1].set_title('Diaphragm 2: g₂(x,y)')  
        axes[1].set_xlabel('x/a')
        axes[1].set_ylabel('y/a')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('fss_shape_network.png', dpi=150)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available")
        
    print("\nShape Network Demo Complete!")
