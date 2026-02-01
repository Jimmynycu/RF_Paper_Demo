"""
Channel Model - Wireless Channel for RIMSA System

Implements the Rician fading channel model with:
- Line-of-Sight (LoS) component based on user position
- Non-Line-of-Sight (NLoS) multipath component
- 3GPP UMi channel characteristics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

try:
    from .rimsa_model import RIMSAConfig
except ImportError:
    from rimsa_model import RIMSAConfig


@dataclass
class ChannelConfig:
    """Configuration for channel model."""
    rician_k: float = 10.0  # Rician K-factor (dB)
    path_loss_exponent: float = 2.5  # Path loss exponent ρ
    reference_distance: float = 1.0  # Reference distance (m)
    noise_power_dbm: float = -90  # Noise power (dBm)
    carrier_frequency: float = 28e9  # Carrier frequency (Hz)
    
    @property
    def rician_k_linear(self) -> float:
        return 10 ** (self.rician_k / 10)
    
    @property
    def noise_power(self) -> float:
        return 10 ** ((self.noise_power_dbm - 30) / 10)  # Convert to linear W


class RicianChannel:
    """
    Rician fading channel model.
    
    From paper Eq. (5):
    h_k = √L_k * (√(K/(K+1)) * h_{k,LoS} + √(1/(K+1)) * h_{k,NLoS})
    
    where:
    - L_k = path loss = 1/d_k^ρ
    - K = Rician factor
    - h_{k,LoS} = deterministic LoS component
    - h_{k,NLoS} = random NLoS component (CN(0,1))
    """
    
    def __init__(self, config: ChannelConfig, rimsa_config: RIMSAConfig):
        self.config = config
        self.rimsa_config = rimsa_config
        
    def compute_path_loss(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute large-scale path loss.
        
        L_k = d_ref^ρ / d_k^ρ
        """
        d_ref = self.config.reference_distance
        rho = self.config.path_loss_exponent
        
        # Avoid division by zero
        distance = torch.clamp(distance, min=0.1)
        
        loss = (d_ref / distance) ** rho
        return loss
    
    def compute_los_component(self,
                             user_position: torch.Tensor,
                             rimsa_position: torch.Tensor,
                             device: str = 'cpu') -> torch.Tensor:
        """
        Compute LoS channel component.
        
        From paper Eq. (7-8):
        h_{k,LoS} = a_R(φ_k, θ_k)
        
        where angles are from user position relative to RIMSA.
        
        Args:
            user_position: User 3D position (3,) or (n_users, 3)
            rimsa_position: RIMSA center position (3,)
            device: Computation device
            
        Returns:
            LoS channel vector (n_elements,) or (n_users, n_elements)
        """
        # Ensure proper dimensions
        if user_position.dim() == 1:
            user_position = user_position.unsqueeze(0)
            
        n_users = user_position.shape[0]
        
        # Relative position
        rel_pos = user_position - rimsa_position.unsqueeze(0)  # (n_users, 3)
        
        # Distance
        distance = torch.norm(rel_pos, dim=1)  # (n_users,)
        
        # Angles from paper Eq. (8)
        # sin(φ)cos(θ) = (y_k - y_R) / d_k
        # sin(θ) = (z_k - z_R) / d_k
        sin_phi_cos_theta = rel_pos[:, 1] / distance
        sin_theta = rel_pos[:, 2] / distance
        
        # Array parameters
        λ = self.rimsa_config.wavelength
        d_R = self.rimsa_config.element_spacing
        N_t = self.rimsa_config.n_total_elements
        N_x = self.rimsa_config.n_elements_x * self.rimsa_config.n_rimsa_x
        
        # Element indices
        n = torch.arange(N_t, device=device).float()
        i1 = n % N_x  # x-index
        i2 = n // N_x  # y-index
        
        # Steering vector for each user
        h_los = torch.zeros(n_users, N_t, dtype=torch.complex64, device=device)
        
        for k in range(n_users):
            phase = 2 * np.pi * d_R / λ * (
                i1 * sin_phi_cos_theta[k] + 
                i2 * sin_theta[k]
            )
            h_los[k] = torch.exp(1j * phase)
            
        return h_los.squeeze(0) if n_users == 1 else h_los
    
    def compute_nlos_component(self,
                              n_users: int,
                              device: str = 'cpu') -> torch.Tensor:
        """
        Generate NLoS (scattered) channel component.
        
        Each component is i.i.d. CN(0, 1).
        """
        N_t = self.rimsa_config.n_total_elements
        
        # Complex Gaussian
        real = torch.randn(n_users, N_t, device=device) / np.sqrt(2)
        imag = torch.randn(n_users, N_t, device=device) / np.sqrt(2)
        
        h_nlos = torch.complex(real, imag)
        return h_nlos
    
    def generate_channel(self,
                        user_positions: torch.Tensor,
                        rimsa_position: Optional[torch.Tensor] = None,
                        device: str = 'cpu') -> torch.Tensor:
        """
        Generate complete Rician channel.
        
        Args:
            user_positions: User positions (n_users, 3)
            rimsa_position: RIMSA position (3,), default at origin
            device: Computation device
            
        Returns:
            Channel matrix H (n_users, n_elements)
        """
        if rimsa_position is None:
            rimsa_position = torch.zeros(3, device=device)
        else:
            rimsa_position = rimsa_position.to(device)
            
        user_positions = user_positions.to(device)
        n_users = user_positions.shape[0]
        
        # Compute distance
        rel_pos = user_positions - rimsa_position.unsqueeze(0)
        distance = torch.norm(rel_pos, dim=1)
        
        # Path loss
        L_k = self.compute_path_loss(distance)
        
        # LoS component
        h_los = self.compute_los_component(user_positions, rimsa_position, device)
        
        # NLoS component
        h_nlos = self.compute_nlos_component(n_users, device)
        
        # Rician combining
        K = self.config.rician_k_linear
        
        h = torch.sqrt(L_k.unsqueeze(1)) * (
            np.sqrt(K / (K + 1)) * h_los +
            np.sqrt(1 / (K + 1)) * h_nlos
        )
        
        return h


class ChannelGenerator:
    """
    Channel data generator for training.
    
    Generates batches of channel realizations with various user positions.
    """
    
    def __init__(self,
                 channel_config: ChannelConfig,
                 rimsa_config: RIMSAConfig,
                 n_users: int = 4,
                 area_size: Tuple[float, float] = (50.0, 50.0),
                 min_distance: float = 10.0,
                 max_distance: float = 100.0,
                 device: str = 'cpu'):
        """
        Initialize channel generator.
        
        Args:
            channel_config: Channel model parameters
            rimsa_config: RIMSA array parameters
            n_users: Number of users per sample
            area_size: Size of user area (x, y) in meters
            min_distance: Minimum user distance from RIMSA
            max_distance: Maximum user distance from RIMSA
            device: Computation device
        """
        self.channel = RicianChannel(channel_config, rimsa_config)
        self.n_users = n_users
        self.area_size = area_size
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.device = device
        
        # RIMSA at origin, z = 10m (elevated)
        self.rimsa_position = torch.tensor([0.0, 0.0, 10.0], device=device)
        
    def generate_user_positions(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate random user positions.
        
        Returns:
            User positions (batch_size, n_users, 3)
        """
        positions = torch.zeros(batch_size, self.n_users, 3, device=self.device)
        
        for b in range(batch_size):
            for k in range(self.n_users):
                # Random distance
                dist = self.min_distance + torch.rand(1, device=self.device) * (
                    self.max_distance - self.min_distance
                )
                
                # Random angle
                angle = torch.rand(1, device=self.device) * 2 * np.pi
                
                # Position (x, y, z=1.5m for mobile users)
                positions[b, k, 0] = dist * torch.cos(angle)
                positions[b, k, 1] = dist * torch.sin(angle)
                positions[b, k, 2] = 1.5  # User height
                
        return positions
    
    def generate_batch(self, batch_size: int = 64) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of training data.
        
        Returns:
            Dictionary with:
            - 'channel': Channel matrices (batch, n_users, n_elements)
            - 'positions': User positions (batch, n_users, 3)
            - 'pilot_signals': Received pilot signals (batch, n_rf_chains, pilot_length)
        """
        # Generate positions
        positions = self.generate_user_positions(batch_size)
        
        # Generate channels
        channels = []
        for b in range(batch_size):
            H = self.channel.generate_channel(
                positions[b],
                self.rimsa_position,
                self.device
            )
            channels.append(H)
            
        channels = torch.stack(channels)  # (batch, n_users, n_elements)
        
        # Generate pilot signals (simplified)
        # In practice, these would be received after uplink training
        pilot_length = 10
        n_rf = self.channel.rimsa_config.n_rf_chains
        
        # Pilot matrix (random orthogonal-like)
        pilots = torch.randn(batch_size, n_rf, pilot_length, 
                            dtype=torch.complex64, device=self.device)
        pilots = pilots / np.sqrt(pilot_length)
        
        return {
            'channel': channels,
            'positions': positions,
            'pilot_signals': pilots
        }
    
    def generate_dataset(self, 
                        n_samples: int = 10000,
                        batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate complete dataset for training.
        
        Returns:
            (channels, positions) tensors
        """
        all_channels = []
        all_positions = []
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for _ in range(n_batches):
            data = self.generate_batch(batch_size)
            all_channels.append(data['channel'])
            all_positions.append(data['positions'])
            
        channels = torch.cat(all_channels)[:n_samples]
        positions = torch.cat(all_positions)[:n_samples]
        
        return channels, positions


class PilotSignalProcessor:
    """
    Process received pilot signals for LLM input.
    
    Converts complex-valued pilots to real tensor format suitable for LLM.
    """
    
    def __init__(self, n_rf_chains: int):
        self.n_rf_chains = n_rf_chains
        
    def to_llm_input(self, pilots: torch.Tensor) -> torch.Tensor:
        """
        Convert complex pilots to LLM input format.
        
        From paper: Stack real and imaginary parts.
        
        Args:
            pilots: Complex tensor (batch, n_rf, pilot_length)
            
        Returns:
            Real tensor (batch, 2, n_rf, pilot_length)
        """
        real = pilots.real
        imag = pilots.imag
        
        return torch.stack([real, imag], dim=1)
    
    def from_llm_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Convert LLM output back to complex values.
        
        Args:
            output: Real tensor with real/imag stacked
            
        Returns:
            Complex tensor
        """
        real = output[:, 0]
        imag = output[:, 1]
        
        return torch.complex(real, imag)


if __name__ == "__main__":
    # Demo: Channel model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    channel_config = ChannelConfig(
        rician_k=10.0,
        path_loss_exponent=2.5,
        carrier_frequency=28e9
    )
    
    rimsa_config = RIMSAConfig(
        n_elements_x=16,
        n_elements_y=16,
        n_rimsa_x=1,
        n_rimsa_y=1,
        carrier_frequency=28e9
    )
    
    print(f"Channel Configuration:")
    print(f"  Rician K-factor: {channel_config.rician_k} dB")
    print(f"  Path loss exponent: {channel_config.path_loss_exponent}")
    print(f"  Carrier frequency: {channel_config.carrier_frequency/1e9} GHz")
    
    # Create channel generator
    generator = ChannelGenerator(
        channel_config=channel_config,
        rimsa_config=rimsa_config,
        n_users=4,
        device=device
    )
    
    # Generate batch
    batch = generator.generate_batch(batch_size=10)
    
    print(f"\nGenerated batch shapes:")
    print(f"  Channels: {batch['channel'].shape}")
    print(f"  Positions: {batch['positions'].shape}")
    print(f"  Pilot signals: {batch['pilot_signals'].shape}")
    
    # Channel statistics
    H = batch['channel']
    print(f"\nChannel statistics:")
    print(f"  Mean magnitude: {torch.abs(H).mean().item():.4f}")
    print(f"  Std magnitude: {torch.abs(H).std().item():.4f}")
    
    # User distances
    positions = batch['positions']
    distances = torch.norm(positions - generator.rimsa_position, dim=2)
    print(f"  User distances: {distances[0].cpu().numpy()} m")
    
    # Visualize channel correlation
    try:
        import matplotlib.pyplot as plt
        
        H_sample = H[0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Channel magnitude
        im1 = axes[0].imshow(np.abs(H_sample), aspect='auto', cmap='viridis')
        axes[0].set_xlabel('Element Index')
        axes[0].set_ylabel('User Index')
        axes[0].set_title('Channel Magnitude |H|')
        plt.colorbar(im1, ax=axes[0])
        
        # Channel phase
        im2 = axes[1].imshow(np.angle(H_sample), aspect='auto', cmap='hsv')
        axes[1].set_xlabel('Element Index')
        axes[1].set_ylabel('User Index')
        axes[1].set_title('Channel Phase ∠H')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('channel_visualization.png', dpi=150)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available")
        
    print("\nChannel Model Demo Complete!")
