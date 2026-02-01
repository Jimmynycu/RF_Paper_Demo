"""
ESA Configuration - 3D Electrically Small Antenna Structure

Implements the high degrees-of-freedom (DoF) antenna configuration
described in the paper:
- 27 customized positions in 1/8 spherical region
- Metal rods with variable radius, length, azimuth, and elevation
- D4 symmetry + mirror symmetry constraints
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import copy


@dataclass
class RodParameters:
    """Parameters for a single metal rod."""
    position_index: int  # Index in 1-27
    exists: bool = True  # Binary: rod present or not
    length: float = 5.0  # Length in mm
    rotate_angle: float = 45.0  # Elevation angle θ in degrees
    azimuth_group: int = 0  # Plane constraint: 0°, 30°, or 60°
    side_length: float = 1.0  # Square bar side length (fixed)
    
    def to_vector(self) -> np.ndarray:
        """Convert to optimization vector [exists, length, angle]."""
        return np.array([
            float(self.exists),
            self.length,
            self.rotate_angle
        ])
    
    @classmethod
    def from_vector(cls, idx: int, vec: np.ndarray, azimuth: int = 0):
        """Create from optimization vector."""
        return cls(
            position_index=idx,
            exists=vec[0] > 0.5,
            length=vec[1],
            rotate_angle=vec[2],
            azimuth_group=azimuth
        )


@dataclass  
class ESAConfiguration:
    """
    3D Electrically Small Antenna Configuration.
    
    Structure:
    - Center-fed dipole (driven element)
    - 27 parasitic metal rods arranged in 1/8 sphere
    - D4 + mirror symmetry applied
    
    Design parameters:
    - 27 binary variables (rod existence)
    - 54 continuous variables (length, rotation angle per rod)
    - Total: 81 parameters after PK constraints
    """
    
    num_rods: int = 27
    sphere_radius: float = 18.3  # mm
    dipole_length: float = 30.0  # mm (initial)
    dipole_gap: float = 1.0  # mm (feed gap)
    
    # Rod positions in spherical coordinates (r, θ, φ) for 1/8 sphere
    rod_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    rods: List[RodParameters] = field(default_factory=list)
    
    # Bounds for continuous parameters
    length_bounds: Tuple[float, float] = (1.0, 12.0)  # mm
    angle_bounds: Tuple[float, float] = (0.0, 180.0)  # degrees
    
    def __post_init__(self):
        """Initialize rod positions and parameters."""
        if not self.rod_positions:
            self._initialize_positions()
        if not self.rods:
            self._initialize_rods()
    
    def _initialize_positions(self):
        """
        Generate 27 uniform positions in 1/8 spherical region.
        
        The 1/8 sphere corresponds to:
        - θ ∈ [0, π/2] (elevation from z-axis)
        - φ ∈ [0, π/4] (azimuth from x-axis)
        """
        # Create 3x3x3 grid in the 1/8 sphere
        n_radial = 3
        n_theta = 3
        n_phi = 3
        
        r_values = np.linspace(0.4, 0.9, n_radial) * self.sphere_radius
        theta_values = np.linspace(15, 75, n_theta)  # degrees
        phi_values = np.array([0, 30, 60])  # Constrained to planes
        
        self.rod_positions = []
        for r in r_values:
            for theta in theta_values:
                for phi in phi_values:
                    self.rod_positions.append((r, theta, phi))
    
    def _initialize_rods(self):
        """Initialize rod parameters with default values."""
        self.rods = []
        for idx, (r, theta, phi) in enumerate(self.rod_positions):
            rod = RodParameters(
                position_index=idx,
                exists=True,
                length=np.random.uniform(*self.length_bounds),
                rotate_angle=np.random.uniform(*self.angle_bounds),
                azimuth_group=int(phi)
            )
            self.rods.append(rod)
    
    def get_binary_vector(self) -> np.ndarray:
        """Get binary existence vector for all rods."""
        return np.array([float(rod.exists) for rod in self.rods])
    
    def get_continuous_vector(self) -> np.ndarray:
        """Get continuous parameters (length, angle) for existing rods."""
        params = []
        for rod in self.rods:
            if rod.exists:
                params.extend([rod.length, rod.rotate_angle])
        return np.array(params)
    
    def get_full_vector(self) -> np.ndarray:
        """Get complete parameter vector [binary; continuous]."""
        binary = self.get_binary_vector()
        continuous = []
        for rod in self.rods:
            continuous.extend([rod.length, rod.rotate_angle])
        return np.concatenate([binary, np.array(continuous)])
    
    def set_from_vector(self, vec: np.ndarray):
        """
        Set configuration from optimization vector.
        
        Vector format: [b_1, ..., b_27, l_1, θ_1, l_2, θ_2, ..., l_27, θ_27]
        """
        n = self.num_rods
        binary = vec[:n]
        continuous = vec[n:].reshape(n, 2)
        
        for i, rod in enumerate(self.rods):
            rod.exists = binary[i] > 0.5
            rod.length = np.clip(continuous[i, 0], *self.length_bounds)
            rod.rotate_angle = np.clip(continuous[i, 1], *self.angle_bounds)
    
    def apply_d4_symmetry(self):
        """
        Apply D4 symmetry constraints.
        
        D4 symmetry: 4-fold rotational symmetry about z-axis
        Mirror symmetry: reflection across xoy plane
        
        This reduces effective DoF by factor of 8.
        """
        # Group rods by their base position
        # For full implementation, would map to symmetric partners
        pass  # Simplified - symmetry applied during evaluation
    
    def get_cartesian_coordinates(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert rod positions to Cartesian coordinates.
        
        Returns list of (start_point, end_point) for each existing rod.
        """
        coordinates = []
        
        for rod, (r, theta_pos, phi) in zip(self.rods, self.rod_positions):
            if not rod.exists:
                continue
                
            # Convert position to Cartesian
            theta_rad = np.radians(theta_pos)
            phi_rad = np.radians(phi)
            
            x = r * np.sin(theta_rad) * np.cos(phi_rad)
            y = r * np.sin(theta_rad) * np.sin(phi_rad)
            z = r * np.cos(theta_rad)
            
            start = np.array([x, y, z])
            
            # Rod extends along direction determined by rotate_angle
            rot_rad = np.radians(rod.rotate_angle)
            # Simplified: rod extends radially outward with rotation
            dx = rod.length * np.sin(rot_rad) * np.cos(phi_rad)
            dy = rod.length * np.sin(rot_rad) * np.sin(phi_rad)  
            dz = rod.length * np.cos(rot_rad)
            
            end = start + np.array([dx, dy, dz])
            coordinates.append((start, end))
            
        return coordinates
    
    def apply_full_symmetry(self) -> 'ESAConfiguration':
        """
        Create full antenna by applying D4 + mirror symmetry.
        
        The 1/8 structure is reflected to create full spherical coverage.
        """
        full_config = copy.deepcopy(self)
        # Would expand to full 216 rods (27 × 8)
        return full_config
    
    def count_active_rods(self) -> int:
        """Count number of existing rods."""
        return sum(1 for rod in self.rods if rod.exists)
    
    def get_parameter_count(self) -> int:
        """Get total number of optimization parameters."""
        n_binary = self.num_rods
        n_continuous = 2 * self.count_active_rods()
        return n_binary + n_continuous
    
    def mutate_binary(self, mutation_rate: float = 0.1):
        """Apply binary mutation to rod existence."""
        for rod in self.rods:
            if np.random.random() < mutation_rate:
                rod.exists = not rod.exists
    
    def mutate_continuous(self, mutation_rate: float = 0.2, strength: float = 0.1):
        """Apply polynomial mutation to continuous parameters."""
        for rod in self.rods:
            if rod.exists:
                if np.random.random() < mutation_rate:
                    # Length mutation
                    delta = (self.length_bounds[1] - self.length_bounds[0]) * strength
                    rod.length += np.random.uniform(-delta, delta)
                    rod.length = np.clip(rod.length, *self.length_bounds)
                    
                if np.random.random() < mutation_rate:
                    # Angle mutation
                    delta = (self.angle_bounds[1] - self.angle_bounds[0]) * strength
                    rod.rotate_angle += np.random.uniform(-delta, delta)
                    rod.rotate_angle = np.clip(rod.rotate_angle, *self.angle_bounds)
    
    def crossover(self, other: 'ESAConfiguration', crossover_rate: float = 0.9) -> 'ESAConfiguration':
        """
        Perform crossover with another configuration.
        
        For binary: uniform crossover
        For continuous: arithmetic crossover
        """
        child = copy.deepcopy(self)
        
        if np.random.random() > crossover_rate:
            return child
            
        for i in range(self.num_rods):
            # Binary crossover
            if np.random.random() < 0.5:
                child.rods[i].exists = other.rods[i].exists
                
            # Continuous crossover (if both exist)
            if child.rods[i].exists and other.rods[i].exists:
                alpha = np.random.random()
                child.rods[i].length = alpha * self.rods[i].length + (1-alpha) * other.rods[i].length
                child.rods[i].rotate_angle = alpha * self.rods[i].rotate_angle + (1-alpha) * other.rods[i].rotate_angle
                
        return child


def create_random_configuration(seed: Optional[int] = None) -> ESAConfiguration:
    """Create a random ESA configuration."""
    if seed is not None:
        np.random.seed(seed)
        
    config = ESAConfiguration()
    
    # Random existence (50% chance)
    for rod in config.rods:
        rod.exists = np.random.random() > 0.5
        if rod.exists:
            rod.length = np.random.uniform(*config.length_bounds)
            rod.rotate_angle = np.random.uniform(*config.angle_bounds)
            
    return config


def create_prior_guided_configuration(seed: Optional[int] = None) -> ESAConfiguration:
    """
    Create configuration guided by prior knowledge from SOTA ESA designs.
    
    Prior knowledge:
    1. Rods near driven dipole strongly affect EM response → higher probability
    2. Top-loading rods enhance performance → favor certain angles
    """
    if seed is not None:
        np.random.seed(seed)
        
    config = ESAConfiguration()
    
    for rod in config.rods:
        r, theta, phi = config.rod_positions[rod.position_index]
        
        # Central region has higher activation probability
        if r < 0.6 * config.sphere_radius:
            p_exist = 0.7
        else:
            p_exist = 0.4
            
        # Top region (low theta) also higher probability
        if theta < 30:
            p_exist = min(0.8, p_exist + 0.2)
            
        rod.exists = np.random.random() < p_exist
        
        if rod.exists:
            # Prior on length: favor 8-12 mm range
            if np.random.random() < 0.6:
                rod.length = np.random.uniform(8, 12)
            else:
                rod.length = np.random.uniform(*config.length_bounds)
                
            # Prior on angle: favor 100-150 degrees for top-loading
            if theta < 30 and np.random.random() < 0.5:
                rod.rotate_angle = np.random.uniform(100, 150)
            else:
                rod.rotate_angle = np.random.uniform(*config.angle_bounds)
                
    return config


if __name__ == "__main__":
    # Demo: Create and visualize configuration
    config = create_prior_guided_configuration(seed=42)
    
    print(f"ESA Configuration Summary:")
    print(f"  Total rods defined: {config.num_rods}")
    print(f"  Active rods: {config.count_active_rods()}")
    print(f"  Sphere radius: {config.sphere_radius} mm")
    
    print(f"\nParameter vector length: {len(config.get_full_vector())}")
    print(f"Binary vector: {config.get_binary_vector()[:5]}... (first 5)")
    
    # Visualize in 3D
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere outline
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = config.sphere_radius * np.outer(np.cos(u), np.sin(v))
        y = config.sphere_radius * np.outer(np.sin(u), np.sin(v))
        z = config.sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, alpha=0.1, color='gray')
        
        # Draw rods
        coordinates = config.get_cartesian_coordinates()
        for start, end in coordinates:
            ax.plot([start[0], end[0]], 
                   [start[1], end[1]], 
                   [start[2], end[2]], 
                   'b-', linewidth=2)
            ax.scatter(*start, c='green', s=30)
            ax.scatter(*end, c='red', s=20)
        
        # Draw dipole
        ax.plot([0, 0], [0, 0], [-config.dipole_length/2, config.dipole_length/2], 
               'k-', linewidth=4, label='Dipole')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('ESA Configuration (1/8 sphere shown)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('esa_configuration.png', dpi=150)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    
    print("\nESA Configuration Demo Complete!")
