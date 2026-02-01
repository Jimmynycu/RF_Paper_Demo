"""
Chu Limit Calculator

Implements the Wheeler-Chu fundamental limit for electrically small antennas.
The Chu limit defines the minimum possible quality factor (Q) for a given
electrical size (ka), which directly constrains achievable bandwidth.

References:
- Wheeler (1947): Fundamental Limitations of Small Antennas
- Chu (1948): Physical Limitations of Omni-Directional Antennas
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChuLimitParams:
    """Parameters for Chu limit calculation."""
    vswr: float = 1.93  # Corresponds to -10 dB return loss
    efficiency: float = 0.95  # Typical metallic antenna efficiency
    mode: str = 'TM'  # TM or TE mode


class ChuLimitCalculator:
    """
    Calculates the theoretical bandwidth limit based on the Chu-Wheeler theory.
    
    The fundamental relationship for single-mode ESA is:
    Q_min = 1/(ka)^3 + 1/(ka)  for TM mode
    
    Where:
    - k = 2π/λ is the wavenumber
    - a = radius of minimum enclosing sphere
    - ka = electrical size (dimensionless)
    - Q = quality factor (inversely related to bandwidth)
    """
    
    def __init__(self, params: Optional[ChuLimitParams] = None):
        self.params = params or ChuLimitParams()
        
    def compute_q_min(self, ka: np.ndarray) -> np.ndarray:
        """
        Compute minimum quality factor for given electrical size.
        
        Args:
            ka: Electrical size (wavenumber × radius)
            
        Returns:
            Q_min: Minimum achievable quality factor
        """
        if self.params.mode == 'TM':
            # TM mode Chu limit
            q_min = 1.0 / (ka ** 3) + 1.0 / ka
        elif self.params.mode == 'TE':
            # TE mode has same form
            q_min = 1.0 / (ka ** 3) + 1.0 / ka
        elif self.params.mode == 'TM+TE':
            # Combined TM and TE modes (lower Q)
            q_tm = 1.0 / (ka ** 3) + 1.0 / ka
            q_te = 1.0 / (ka ** 3) + 1.0 / ka
            q_min = q_tm * q_te / (q_tm + q_te)
        else:
            raise ValueError(f"Unknown mode: {self.params.mode}")
            
        return q_min
    
    def compute_bandwidth_limit(self, ka: np.ndarray) -> np.ndarray:
        """
        Compute maximum achievable fractional bandwidth.
        
        From the paper, Eq. (2):
        BW_chu = (s-1) / (s * Q_min) * η_rad
        
        where s is the VSWR bound and η_rad is radiation efficiency.
        
        For -10 dB bandwidth: s = 1.925 (approximately)
        
        Args:
            ka: Electrical size array
            
        Returns:
            bw_max: Maximum fractional bandwidth (as percentage)
        """
        s = self.params.vswr
        eta = self.params.efficiency
        
        q_min = self.compute_q_min(ka)
        
        # Fractional bandwidth formula
        bw_max = (s - 1) / (np.sqrt(s) * q_min) * eta
        
        return bw_max * 100  # Convert to percentage
    
    def compute_bandwidth_efficiency_product(self, ka: np.ndarray) -> np.ndarray:
        """
        Compute the bandwidth-efficiency product limit.
        
        B*η = BW * η
        
        This is the key metric for comparing ESA performance.
        """
        bw = self.compute_bandwidth_limit(ka) / 100  # Back to fraction
        eta = self.params.efficiency
        return bw * eta
    
    def generate_pareto_front(self, 
                              ka_min: float = 0.3,
                              ka_max: float = 2.0,
                              n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the true Pareto front based on Chu limit.
        
        This serves as reference points for MOEA/D decomposition.
        
        Args:
            ka_min: Minimum electrical size
            ka_max: Maximum electrical size  
            n_points: Number of points to sample
            
        Returns:
            ka_points: Electrical size values (objective 1)
            bw_points: Bandwidth limit values (objective 2)
        """
        ka_points = np.linspace(ka_min, ka_max, n_points)
        bw_points = self.compute_bandwidth_limit(ka_points)
        
        return ka_points, bw_points
    
    def apply_engineering_constraints(self,
                                     ka_points: np.ndarray,
                                     bw_points: np.ndarray,
                                     f_low: float = 1.5e9,
                                     f_high: float = 5.0e9,
                                     radius: float = 18.3e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply engineering constraints to the Pareto front.
        
        Engineering constraints arise from:
        1. Physical antenna size (radius R)
        2. Simulation/operating frequency range [f_L, f_U]
        
        Args:
            ka_points: Electrical size values
            bw_points: Theoretical bandwidth values
            f_low: Lower frequency bound (Hz)
            f_high: Upper frequency bound (Hz)  
            radius: Enclosing sphere radius (m)
            
        Returns:
            Constrained ka and bandwidth arrays
        """
        c = 3e8  # Speed of light
        
        # Calculate ka bounds from frequency constraints
        ka_at_f_low = 2 * np.pi * f_low / c * radius
        ka_at_f_high = 2 * np.pi * f_high / c * radius
        
        # Engineering constraint 1: Left bound (low frequency)
        # P_-10,L fixed at f_L, sweep P_-10,R from f_L to f_U
        bw_eng1 = 2 * (ka_points - ka_at_f_low) / (ka_points + ka_at_f_low) * 100
        
        # Engineering constraint 2: Right bound (high frequency)
        # P_-10,R fixed at f_U, sweep P_-10,L from f_L to f_U
        bw_eng2 = 2 * (ka_at_f_high - ka_points) / (ka_at_f_high + ka_points) * 100
        
        # Combine constraints - take minimum of theoretical and engineering
        bw_constrained = np.minimum(bw_points, np.maximum(bw_eng1, 0))
        bw_constrained = np.minimum(bw_constrained, np.maximum(bw_eng2, 0))
        
        return ka_points, bw_constrained
    
    def is_beyond_limit(self, ka: float, bw: float, efficiency: float = 0.95) -> bool:
        """
        Check if an antenna design exceeds the theoretical limit.
        
        Args:
            ka: Measured electrical size
            bw: Measured fractional bandwidth (%)
            efficiency: Measured radiation efficiency
            
        Returns:
            True if design exceeds Chu limit
        """
        bw_limit = self.compute_bandwidth_limit(np.array([ka]))[0]
        bw_eta_measured = bw * efficiency / 100
        bw_eta_limit = bw_limit * self.params.efficiency / 100
        
        return bw_eta_measured > bw_eta_limit


# Utility functions for antenna design
def calculate_ka(frequency: float, radius: float) -> float:
    """
    Calculate electrical size ka from frequency and physical radius.
    
    Args:
        frequency: Operating frequency in Hz
        radius: Antenna radius in meters
        
    Returns:
        ka: Electrical size (dimensionless)
    """
    c = 3e8
    k = 2 * np.pi * frequency / c
    return k * radius


def calculate_fractional_bandwidth(f_low: float, f_high: float) -> float:
    """
    Calculate fractional bandwidth from frequency bounds.
    
    Args:
        f_low: Lower -10 dB frequency  
        f_high: Upper -10 dB frequency
        
    Returns:
        Fractional bandwidth as percentage
    """
    f_center = (f_low + f_high) / 2
    return (f_high - f_low) / f_center * 100


if __name__ == "__main__":
    # Demo: Plot Chu limit curve
    import matplotlib.pyplot as plt
    
    calculator = ChuLimitCalculator()
    ka, bw = calculator.generate_pareto_front(ka_min=0.3, ka_max=2.0, n_points=100)
    
    # Apply engineering constraints  
    ka_const, bw_const = calculator.apply_engineering_constraints(
        ka, bw, f_low=1.5e9, f_high=5.0e9, radius=18.3e-3
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(ka, bw, 'b-', linewidth=2, label='Chu Limit (Performance Limit)')
    plt.plot(ka_const, bw_const, 'r--', linewidth=2, label='Engineering Constrained')
    plt.fill_between(ka_const, 0, bw_const, alpha=0.2, color='green', label='Feasible Region')
    
    plt.xlabel('Electrical Size (ka)', fontsize=12)
    plt.ylabel('Fractional Bandwidth (%)', fontsize=12)
    plt.title('Chu Limit Pareto Front for ESA Design', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0.3, 2.0])
    plt.ylim([0, 80])
    
    plt.tight_layout()
    plt.savefig('chu_limit_pareto_front.png', dpi=150)
    plt.show()
    
    print("Chu Limit Calculator Demo Complete!")
