"""
RIMSA Model - Reconfigurable Intelligent Metasurface Antenna System

Implements the RIMSA architecture with:
- Parallel coaxial feeding network
- Per-element phase control via varactor diodes
- 2D planar array configuration
- Beamforming computation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class RIMSAConfig:
    """Configuration for RIMSA array."""
    n_elements_x: int = 16  # Elements per RIMSA in x-direction
    n_elements_y: int = 16  # Elements per RIMSA in y-direction
    n_rimsa_x: int = 1  # Number of RIMSAs in x-direction
    n_rimsa_y: int = 1  # Number of RIMSAs in y-direction
    carrier_frequency: float = 28e9  # 28 GHz mmWave
    element_spacing: float = None  # Default: λ/2
    
    @property
    def n_elements_per_rimsa(self) -> int:
        return self.n_elements_x * self.n_elements_y
    
    @property
    def n_rf_chains(self) -> int:
        return self.n_rimsa_x * self.n_rimsa_y
    
    @property
    def n_total_elements(self) -> int:
        return self.n_elements_per_rimsa * self.n_rf_chains
    
    @property
    def wavelength(self) -> float:
        c = 3e8
        return c / self.carrier_frequency
    
    def __post_init__(self):
        if self.element_spacing is None:
            self.element_spacing = self.wavelength / 2


class RIMSASystem(nn.Module):
    """
    RIMSA Array System Model.
    
    Key features from paper:
    - Parallel coaxial feeding (not serial like DMA)
    - Varactor-based phase control (sub-nanosecond response)
    - Independent amplitude and phase modulation per element
    - Compact 2D integration
    
    Architecture:
    - NRx × NRy RIMSA antennas (RF chains)
    - Each RIMSA has NEx × NEy metamaterial elements
    - Total elements: Nx × Ny = NEx*NRx × NEy*NRy
    """
    
    def __init__(self, config: RIMSAConfig):
        super().__init__()
        self.config = config
        
        # Physical constants
        self.c = 3e8
        
        # Element positions (UPA - Uniform Planar Array)
        self._setup_element_positions()
        
    def _setup_element_positions(self):
        """Setup element position coordinates."""
        d = self.config.element_spacing
        Nx = self.config.n_elements_x * self.config.n_rimsa_x
        Ny = self.config.n_elements_y * self.config.n_rimsa_y
        
        # Element indices
        x_indices = torch.arange(Nx).float()
        y_indices = torch.arange(Ny).float()
        
        # Create position grid
        self.register_buffer('x_positions', x_indices * d)
        self.register_buffer('y_positions', y_indices * d)
        
    def compute_steering_vector(self, 
                                azimuth: torch.Tensor,
                                elevation: torch.Tensor,
                                device: str = 'cpu') -> torch.Tensor:
        """
        Compute array steering vector.
        
        From paper Eq. (6):
        [a_R(φ,θ)]_n = exp(j * 2π/λ * d_R * {i₁(n)sin(φ)cos(θ) + i₂(n)sin(θ)})
        
        Args:
            azimuth: Azimuth angle φ (radians)
            elevation: Elevation angle θ (radians)
            device: Computation device
            
        Returns:
            Steering vector of shape (n_total_elements,)
        """
        λ = self.config.wavelength
        d = self.config.element_spacing
        N = self.config.n_total_elements
        N_x = self.config.n_elements_x * self.config.n_rimsa_x
        
        # Element indices
        n = torch.arange(N, device=device).float()
        i1 = n % N_x  # x-index
        i2 = n // N_x  # y-index
        
        # Phase from geometry
        phase = 2 * np.pi * d / λ * (
            i1 * torch.sin(azimuth) * torch.cos(elevation) +
            i2 * torch.sin(elevation)
        )
        
        return torch.exp(1j * phase)
    
    def compute_beamforming_matrix(self,
                                   phase_config: torch.Tensor) -> torch.Tensor:
        """
        Construct RIMSA beamforming matrix V from phase configuration.
        
        From paper Eq. (3):
        V = diag(v_1, v_2, ..., v_NR)
        
        where v_nr = 1/√N_E * [v_{nr,1}, ..., v_{nr,NE}]ᵀ
        
        Args:
            phase_config: Phase values (n_rf_chains, n_elements_per_rimsa)
            
        Returns:
            V: Beamforming matrix (n_total_elements, n_rf_chains)
        """
        N_E = self.config.n_elements_per_rimsa
        N_R = self.config.n_rf_chains
        
        # Normalize and convert to complex
        phase_weights = torch.exp(1j * phase_config) / np.sqrt(N_E)
        
        # Build block-diagonal matrix
        V = torch.zeros(N_E * N_R, N_R, dtype=torch.complex64, device=phase_config.device)
        
        for r in range(N_R):
            start_idx = r * N_E
            end_idx = (r + 1) * N_E
            V[start_idx:end_idx, r] = phase_weights[r]
            
        return V
    
    def apply_beamforming(self,
                         signal: torch.Tensor,
                         V: torch.Tensor) -> torch.Tensor:
        """
        Apply RIMSA beamforming to signal.
        
        Args:
            signal: Input signal (n_rf_chains,)
            V: Beamforming matrix
            
        Returns:
            Output signal from all elements
        """
        return V @ signal
    
    def compute_received_signal(self,
                               tx_signal: torch.Tensor,
                               channel: torch.Tensor,
                               V: torch.Tensor,
                               W: torch.Tensor,
                               noise_power: float = 1e-10) -> torch.Tensor:
        """
        Compute received signal at users.
        
        From paper Eq. (4):
        r_k = h_k^H V w_k s_k + Σ_{i≠k} h_k^H V w_i s_i + n_k
        
        Args:
            tx_signal: Transmitted symbols (n_users,)
            channel: Channel matrix H (n_users, n_total_elements)
            V: RIMSA beamforming matrix
            W: Digital precoding matrix (n_rf_chains, n_users)
            noise_power: Noise variance
            
        Returns:
            Received signals (n_users,)
        """
        # Effective channel after beamforming
        H_eff = channel @ V  # (n_users, n_rf_chains)
        
        # Apply digital precoding
        y = H_eff @ W @ tx_signal  # (n_users,)
        
        # Add noise
        noise = torch.randn_like(y) * np.sqrt(noise_power / 2)
        noise = noise + 1j * torch.randn_like(y.real) * np.sqrt(noise_power / 2)
        
        return y + noise
    
    def compute_sinr(self,
                    channel: torch.Tensor,
                    V: torch.Tensor,
                    W: torch.Tensor,
                    noise_power: float = 1e-10) -> torch.Tensor:
        """
        Compute SINR for each user.
        
        From paper Eq. (9):
        SINR_k = |h_k^H V w_k|² / (Σ_{i≠k} |h_k^H V w_i|² + σ²)
        
        Args:
            channel: Channel matrix (n_users, n_total_elements)
            V: RIMSA beamforming matrix (n_total_elements, n_rf_chains)
            W: Digital precoding (n_rf_chains, n_users)
            noise_power: Noise variance
            
        Returns:
            SINR per user (n_users,)
        """
        n_users = channel.shape[0]
        
        # Effective channel
        H_eff = channel @ V  # (n_users, n_rf_chains)
        
        # Signal power for each user
        signal_power = torch.zeros(n_users, device=channel.device)
        interference = torch.zeros(n_users, device=channel.device)
        
        for k in range(n_users):
            h_k = H_eff[k]
            
            # Desired signal
            signal_power[k] = torch.abs(h_k @ W[:, k]) ** 2
            
            # Interference from other users
            for i in range(n_users):
                if i != k:
                    interference[k] += torch.abs(h_k @ W[:, i]) ** 2
                    
        sinr = signal_power / (interference + noise_power)
        return sinr
    
    def compute_sum_rate(self,
                        channel: torch.Tensor,
                        V: torch.Tensor,
                        W: torch.Tensor,
                        noise_power: float = 1e-10) -> torch.Tensor:
        """
        Compute sum rate capacity.
        
        From paper Eq. (11):
        R_sum = Σ_k log₂(1 + SINR_k)
        
        Args:
            channel, V, W, noise_power: Same as compute_sinr
            
        Returns:
            Sum rate (scalar)
        """
        sinr = self.compute_sinr(channel, V, W, noise_power)
        rates = torch.log2(1 + sinr)
        return torch.sum(rates)
    
    def compute_zf_precoding(self, channel: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Compute Zero-Forcing digital precoding matrix.
        
        Used as reference/regularization in training.
        
        Args:
            channel: Channel matrix (n_users, n_total_elements)
            V: RIMSA beamforming matrix
            
        Returns:
            W_ZF: ZF precoding matrix (n_rf_chains, n_users)
        """
        # Effective channel
        H_eff = channel @ V  # (n_users, n_rf_chains)
        
        # ZF precoding: W = H_eff^H (H_eff H_eff^H)^{-1}
        HH = H_eff @ H_eff.conj().T
        
        # Regularized inverse for stability
        reg = 1e-6 * torch.eye(HH.shape[0], device=HH.device, dtype=HH.dtype)
        HH_inv = torch.linalg.inv(HH + reg)
        
        W_ZF = H_eff.conj().T @ HH_inv
        
        # Power normalization
        power = torch.norm(W_ZF, dim=0, keepdim=True)
        W_ZF = W_ZF / (power + 1e-8)
        
        return W_ZF


class PhaseControlCircuit(nn.Module):
    """
    Model of varactor-based phase control circuit.
    
    Maps control voltage to phase shift.
    """
    
    def __init__(self, n_elements: int):
        super().__init__()
        self.n_elements = n_elements
        
        # Varactor parameters (simplified)
        self.v_min = 0.0  # Minimum control voltage
        self.v_max = 5.0  # Maximum control voltage
        self.phase_range = 2 * np.pi  # Full 360° range
        
    def voltage_to_phase(self, voltage: torch.Tensor) -> torch.Tensor:
        """Convert control voltage to phase shift."""
        # Normalize to [0, 1]
        normalized = (voltage - self.v_min) / (self.v_max - self.v_min)
        normalized = torch.clamp(normalized, 0, 1)
        
        # Map to phase [0, 2π]
        phase = normalized * self.phase_range
        return phase
    
    def phase_to_voltage(self, phase: torch.Tensor) -> torch.Tensor:
        """Convert desired phase to control voltage."""
        # Wrap phase to [0, 2π]
        phase = phase % (2 * np.pi)
        
        # Map to voltage
        normalized = phase / self.phase_range
        voltage = normalized * (self.v_max - self.v_min) + self.v_min
        
        return voltage


if __name__ == "__main__":
    # Demo: RIMSA system model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create RIMSA configuration
    config = RIMSAConfig(
        n_elements_x=8,
        n_elements_y=8,
        n_rimsa_x=2,
        n_rimsa_y=2,
        carrier_frequency=28e9
    )
    
    print(f"RIMSA Configuration:")
    print(f"  Elements per RIMSA: {config.n_elements_per_rimsa}")
    print(f"  RF chains: {config.n_rf_chains}")
    print(f"  Total elements: {config.n_total_elements}")
    print(f"  Wavelength: {config.wavelength*1000:.2f} mm")
    print(f"  Element spacing: {config.element_spacing*1000:.2f} mm")
    
    # Create system
    rimsa = RIMSASystem(config).to(device)
    
    # Generate random channel and phase configuration
    n_users = 4
    N_t = config.n_total_elements
    N_R = config.n_rf_chains
    N_E = config.n_elements_per_rimsa
    
    # Random channel
    H = (torch.randn(n_users, N_t, device=device) + 
         1j * torch.randn(n_users, N_t, device=device)) / np.sqrt(2 * N_t)
    
    # Random phase configuration
    phase_config = torch.rand(N_R, N_E, device=device) * 2 * np.pi
    
    # Compute beamforming matrix
    V = rimsa.compute_beamforming_matrix(phase_config)
    print(f"\nBeamforming matrix V shape: {V.shape}")
    
    # Compute ZF precoding
    W = rimsa.compute_zf_precoding(H, V)
    print(f"Precoding matrix W shape: {W.shape}")
    
    # Compute performance
    sinr = rimsa.compute_sinr(H, V, W, noise_power=1e-10)
    sum_rate = rimsa.compute_sum_rate(H, V, W, noise_power=1e-10)
    
    print(f"\nPerformance metrics:")
    print(f"  SINR per user: {10*torch.log10(sinr).cpu().numpy()} dB")
    print(f"  Sum rate: {sum_rate.item():.2f} bps/Hz")
    
    # Test steering vector
    azimuth = torch.tensor(np.pi / 4)  # 45°
    elevation = torch.tensor(np.pi / 6)  # 30°
    
    a = rimsa.compute_steering_vector(azimuth, elevation, device=device)
    print(f"\nSteering vector shape: {a.shape}")
    print(f"Steering vector magnitude: {torch.abs(a).mean().item():.4f}")
    
    print("\nRIMSA System Demo Complete!")
