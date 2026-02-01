"""
MOEA/D - Multi-Objective Evolutionary Algorithm based on Decomposition

Implements the Chu-Limit-Guided MOEA/D framework from the paper:
- Performance-limit-guided problem decomposition
- Population-to-reference-points reassignment
- Mixed-variable offspring reproduction
- Prior knowledge-guided initialization

Reference:
Kuang et al., "Chu-Limit-Guided Decomposition-Based Multiobjective 
Large-Scale Optimization for Generative Broadband ESA Design"
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import copy
from tqdm import tqdm

# Import local modules
import sys
from pathlib import Path

# Add parent directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

try:
    from .chu_limit import ChuLimitCalculator
    from .esa_configuration import ESAConfiguration, create_prior_guided_configuration
    from .evaluator import AntennaEvaluator, AntennaPerformance
except ImportError:
    from chu_limit import ChuLimitCalculator
    from esa_configuration import ESAConfiguration, create_prior_guided_configuration
    from evaluator import AntennaEvaluator, AntennaPerformance


@dataclass
class MOEADConfig:
    """Configuration for MOEA/D algorithm."""
    population_size: int = 50
    max_generations: int = 20
    neighborhood_size: int = 5
    crossover_rate: float = 0.9
    mutation_rate_binary: float = 0.1
    mutation_rate_continuous: float = 0.2
    mutation_strength: float = 0.1
    neighborhood_selection_prob: float = 0.9  # δ in paper
    max_replacements: int = 2
    use_coarse_evaluation: bool = True
    seed: Optional[int] = None


@dataclass
class Individual:
    """Single solution in the population."""
    config: ESAConfiguration
    performance: Optional[AntennaPerformance] = None
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(2))
    subproblem_index: int = -1
    
    @property
    def is_valid(self) -> bool:
        return self.performance is not None and self.performance.is_valid
    
    def update_objectives(self):
        """Update objective values from performance."""
        if self.performance and self.performance.is_valid:
            # Objective 1: ka (minimize)
            self.objectives[0] = self.performance.ka
            # Objective 2: -BW (minimize, so we maximize BW)
            self.objectives[1] = -self.performance.bandwidth
        else:
            self.objectives = np.array([np.inf, 0])


class MOEAD:
    """
    Chu-Limit-Guided MOEA/D for ESA Optimization.
    
    Key innovations:
    1. Reference points from true Pareto front (Chu limit)
    2. Population reassignment based on ka proximity
    3. Mixed-variable reproduction (binary + continuous)
    4. Prior knowledge-guided initialization
    """
    
    def __init__(self, 
                 config: MOEADConfig,
                 evaluator: AntennaEvaluator,
                 chu_calculator: ChuLimitCalculator):
        self.config = config
        self.evaluator = evaluator
        self.chu_calculator = chu_calculator
        
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Initialize algorithm state
        self.population: List[Individual] = []
        self.reference_points: np.ndarray = None
        self.neighborhoods: List[List[int]] = []
        self.ideal_point: np.ndarray = np.array([np.inf, np.inf])
        
        # Statistics
        self.history = {
            'generation': [],
            'igd': [],
            'beyond_limit_count': [],
            'best_bandwidth': []
        }
        
    def initialize(self):
        """Initialize reference points, neighborhoods, and population."""
        print("Initializing MOEA/D...")
        
        # Step 1: Generate reference points from Chu limit
        self._initialize_reference_points()
        
        # Step 2: Build neighborhoods
        self._build_neighborhoods()
        
        # Step 3: Generate initial population with prior knowledge
        self._initialize_population()
        
        print(f"Initialization complete. Population size: {len(self.population)}")
        print(f"Reference points: {len(self.reference_points)}")
        
    def _initialize_reference_points(self):
        """
        Generate reference points from the true Pareto front.
        
        Uses Chu limit + engineering constraints to define the PF.
        """
        ka, bw = self.chu_calculator.generate_pareto_front(
            ka_min=0.5, 
            ka_max=1.5,
            n_points=self.config.population_size
        )
        
        # Apply engineering constraints
        ka, bw = self.chu_calculator.apply_engineering_constraints(
            ka, bw,
            f_low=self.evaluator.freq_start,
            f_high=self.evaluator.freq_stop
        )
        
        # Reference points: (ka, -BW) since we minimize both
        self.reference_points = np.column_stack([ka, -bw])
        
    def _build_neighborhoods(self):
        """
        Build neighborhood structure based on reference point distances.
        
        Each subproblem has T closest neighbors.
        """
        n = len(self.reference_points)
        T = self.config.neighborhood_size
        
        # Compute pairwise distances in ka dimension only
        ka_values = self.reference_points[:, 0]
        dist_matrix = np.abs(ka_values[:, None] - ka_values[None, :])
        
        self.neighborhoods = []
        for i in range(n):
            sorted_indices = np.argsort(dist_matrix[i])
            neighbors = sorted_indices[:T].tolist()
            self.neighborhoods.append(neighbors)
            
    def _initialize_population(self):
        """
        Initialize population using prior knowledge.
        
        Uses non-uniform sampling based on SOTA ESA designs.
        """
        self.population = []
        
        with tqdm(total=self.config.population_size, desc="Initializing population") as pbar:
            attempts = 0
            max_attempts = self.config.population_size * 10
            
            while len(self.population) < self.config.population_size and attempts < max_attempts:
                # Create configuration with prior knowledge
                config = create_prior_guided_configuration()
                
                # Evaluate (coarse mesh for speed)
                performance = self.evaluator.evaluate(config)
                
                if performance.is_valid:
                    ind = Individual(config=config, performance=performance)
                    ind.update_objectives()
                    self.population.append(ind)
                    
                    # Update ideal point
                    self._update_ideal_point(ind.objectives)
                    pbar.update(1)
                    
                attempts += 1
                
        print(f"Created {len(self.population)} valid individuals from {attempts} attempts")
        
    def _update_ideal_point(self, objectives: np.ndarray):
        """Update ideal point (best seen for each objective)."""
        self.ideal_point = np.minimum(self.ideal_point, objectives)
        
    def population_reassignment(self):
        """
        Reassign population to subproblems based on ka proximity.
        
        Algorithm (from paper):
        1. Sort individuals by bandwidth (descending)
        2. Assign each to closest unassigned reference point
        """
        n = len(self.population)
        
        # Sort by bandwidth (descending) - objective[1] is -BW
        sorted_indices = np.argsort([ind.objectives[1] for ind in self.population])
        
        # Track which reference points are assigned
        available_refs = set(range(len(self.reference_points)))
        
        for idx in sorted_indices:
            ind = self.population[idx]
            
            if not available_refs:
                break
                
            # Find closest available reference point by ka
            best_ref = None
            best_dist = np.inf
            
            for ref_idx in available_refs:
                dist = abs(ind.objectives[0] - self.reference_points[ref_idx, 0])
                if dist < best_dist:
                    best_dist = dist
                    best_ref = ref_idx
                    
            if best_ref is not None:
                ind.subproblem_index = best_ref
                available_refs.remove(best_ref)
                
    def tchebycheff_scalar(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """
        Compute Tchebycheff scalarizing function value.
        
        g^te(x|z) = max_i |f_i(x) - z_i|
        """
        diff = np.abs(objectives - ref_point)
        return np.max(diff)
    
    def offspring_reproduction(self, 
                               parent: Individual,
                               pool: List[Individual]) -> Individual:
        """
        Generate offspring using mixed-variable operators.
        
        Binary part: DE/rand/1 mutation + uniform crossover
        Continuous part: DE/rand/1 + polynomial mutation
        """
        if len(pool) < 3:
            return copy.deepcopy(parent)
            
        # Select parents for DE
        indices = np.random.choice(len(pool), 3, replace=False)
        p1, p2, p3 = pool[indices[0]], pool[indices[1]], pool[indices[2]]
        
        # Create offspring configuration
        child_config = copy.deepcopy(parent.config)
        
        # Binary mutation (XOR-based DE)
        F = 0.5
        for i in range(child_config.num_rods):
            if np.random.random() < self.config.mutation_rate_binary:
                # DE/rand/1 for binary
                xor_val = int(p1.config.rods[i].exists) ^ int(p2.config.rods[i].exists)
                if np.random.random() < F and xor_val:
                    child_config.rods[i].exists = not parent.config.rods[i].exists
                    
        # Uniform crossover for binary
        for i in range(child_config.num_rods):
            if np.random.random() < 0.5:
                child_config.rods[i].exists = p1.config.rods[i].exists
                
        # Continuous mutation (DE + polynomial)
        for i in range(child_config.num_rods):
            if not child_config.rods[i].exists:
                continue
                
            # Check if parents have this rod
            if not (p1.config.rods[i].exists and p2.config.rods[i].exists and p3.config.rods[i].exists):
                continue
                
            # DE/rand/1 for continuous
            if np.random.random() < self.config.mutation_rate_continuous:
                diff = p1.config.rods[i].length - p2.config.rods[i].length
                child_config.rods[i].length = p3.config.rods[i].length + F * diff
                
            if np.random.random() < self.config.mutation_rate_continuous:
                diff = p1.config.rods[i].rotate_angle - p2.config.rods[i].rotate_angle
                child_config.rods[i].rotate_angle = p3.config.rods[i].rotate_angle + F * diff
                
        # Polynomial mutation
        child_config.mutate_continuous(
            mutation_rate=self.config.mutation_rate_continuous / 2,
            strength=self.config.mutation_strength
        )
        
        # Boundary repair
        for rod in child_config.rods:
            rod.length = np.clip(rod.length, *child_config.length_bounds)
            rod.rotate_angle = np.clip(rod.rotate_angle, *child_config.angle_bounds)
            
        # Evaluate offspring
        performance = self.evaluator.evaluate(child_config)
        
        offspring = Individual(config=child_config, performance=performance)
        offspring.update_objectives()
        
        return offspring
    
    def update_neighbors(self, offspring: Individual, parent_idx: int):
        """
        Update neighboring solutions if offspring is better.
        
        Uses Tchebycheff approach for comparison.
        """
        if not offspring.is_valid:
            return
            
        parent = self.population[parent_idx]
        neighbors = self.neighborhoods[parent.subproblem_index]
        
        replacement_count = 0
        
        for neighbor_idx in neighbors:
            if replacement_count >= self.config.max_replacements:
                break
                
            neighbor = self.population[neighbor_idx]
            ref_point = self.reference_points[neighbor.subproblem_index]
            
            # Compare using Tchebycheff
            offspring_value = self.tchebycheff_scalar(offspring.objectives, ref_point)
            neighbor_value = self.tchebycheff_scalar(neighbor.objectives, ref_point)
            
            if offspring_value < neighbor_value:
                self.population[neighbor_idx] = copy.deepcopy(offspring)
                self.population[neighbor_idx].subproblem_index = neighbor.subproblem_index
                replacement_count += 1
                
    def compute_igd(self) -> float:
        """
        Compute Inverted Generational Distance to true Pareto front.
        
        Lower IGD = better convergence and diversity.
        """
        valid_pop = [ind for ind in self.population if ind.is_valid]
        if not valid_pop:
            return np.inf
            
        pop_objectives = np.array([ind.objectives for ind in valid_pop])
        
        # For each reference point, find minimum distance to population
        distances = []
        for ref in self.reference_points:
            dists = np.linalg.norm(pop_objectives - ref, axis=1)
            distances.append(np.min(dists))
            
        return np.mean(distances)
    
    def count_beyond_limit(self) -> int:
        """Count solutions that exceed the Chu limit."""
        count = 0
        for ind in self.population:
            if ind.is_valid:
                bw_limit = self.chu_calculator.compute_bandwidth_limit(np.array([ind.performance.ka]))[0]
                if ind.performance.bandwidth > bw_limit * ind.performance.total_efficiency:
                    count += 1
        return count
    
    def run(self) -> List[Individual]:
        """
        Execute the MOEA/D optimization.
        
        Returns:
            Final Pareto-optimal population
        """
        # Initialize
        self.initialize()
        
        print("\nStarting optimization...")
        
        for gen in range(1, self.config.max_generations + 1):
            # Population reassignment
            self.population_reassignment()
            
            # Evolution
            for i in range(len(self.population)):
                parent = self.population[i]
                
                # Select mating pool
                if np.random.random() < self.config.neighborhood_selection_prob:
                    pool_indices = self.neighborhoods[parent.subproblem_index]
                else:
                    pool_indices = list(range(len(self.population)))
                    
                pool = [self.population[j] for j in pool_indices]
                
                # Generate offspring
                offspring = self.offspring_reproduction(parent, pool)
                
                # Update ideal point
                if offspring.is_valid:
                    self._update_ideal_point(offspring.objectives)
                    
                # Update neighbors
                self.update_neighbors(offspring, i)
                
            # Compute statistics
            igd = self.compute_igd()
            beyond_count = self.count_beyond_limit()
            best_bw = max([ind.performance.bandwidth for ind in self.population if ind.is_valid], default=0)
            
            self.history['generation'].append(gen)
            self.history['igd'].append(igd)
            self.history['beyond_limit_count'].append(beyond_count)
            self.history['best_bandwidth'].append(best_bw)
            
            print(f"Gen {gen:3d}: IGD={igd:.4f}, Beyond-limit={beyond_count}, Best BW={best_bw:.1f}%")
            
        return self.get_pareto_front()
    
    def get_pareto_front(self) -> List[Individual]:
        """Extract non-dominated solutions from population."""
        valid_pop = [ind for ind in self.population if ind.is_valid]
        
        if not valid_pop:
            return []
            
        pareto = []
        
        for ind in valid_pop:
            dominated = False
            for other in valid_pop:
                if ind is other:
                    continue
                # Check if other dominates ind
                if (other.objectives[0] <= ind.objectives[0] and 
                    other.objectives[1] <= ind.objectives[1] and
                    (other.objectives[0] < ind.objectives[0] or 
                     other.objectives[1] < ind.objectives[1])):
                    dominated = True
                    break
                    
            if not dominated:
                pareto.append(ind)
                
        return pareto


def run_optimization(seed: int = 42) -> Tuple[List[Individual], MOEAD]:
    """
    Run complete ESA optimization with default settings.
    
    Returns:
        (pareto_front, moead_instance)
    """
    # Setup
    config = MOEADConfig(
        population_size=50,
        max_generations=20,
        neighborhood_size=5,
        seed=seed
    )
    
    evaluator = AntennaEvaluator(use_coarse_mesh=True)
    chu_calculator = ChuLimitCalculator()
    
    # Create and run MOEA/D
    moead = MOEAD(config, evaluator, chu_calculator)
    pareto_front = moead.run()
    
    return pareto_front, moead


if __name__ == "__main__":
    print("=" * 60)
    print("Paper 1: Chu-Limit-Guided MOEA/D for ESA Design")
    print("=" * 60)
    
    # Run with smaller population for demo
    config = MOEADConfig(
        population_size=20,
        max_generations=10,
        neighborhood_size=3,
        seed=42
    )
    
    evaluator = AntennaEvaluator(use_coarse_mesh=True)
    chu_calculator = ChuLimitCalculator()
    
    moead = MOEAD(config, evaluator, chu_calculator)
    pareto_front = moead.run()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Pareto front size: {len(pareto_front)}")
    print(f"Beyond-limit solutions: {moead.count_beyond_limit()}")
    
    print("\nTop solutions:")
    for i, ind in enumerate(sorted(pareto_front, key=lambda x: -x.performance.bandwidth)[:5]):
        print(f"  {i+1}. ka={ind.performance.ka:.3f}, BW={ind.performance.bandwidth:.1f}%, "
              f"η={ind.performance.total_efficiency*100:.1f}%")
              
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Pareto front
        ax1 = axes[0]
        ka_ref, bw_ref = chu_calculator.generate_pareto_front()
        ax1.plot(ka_ref, bw_ref, 'b-', linewidth=2, label='Chu Limit')
        
        for ind in moead.population:
            if ind.is_valid:
                color = 'red' if ind in pareto_front else 'gray'
                alpha = 1.0 if ind in pareto_front else 0.3
                ax1.scatter(ind.performance.ka, ind.performance.bandwidth, 
                           c=color, alpha=alpha, s=50)
                
        ax1.set_xlabel('Electrical Size (ka)')
        ax1.set_ylabel('Fractional Bandwidth (%)')
        ax1.set_title('Optimization Results vs Chu Limit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Convergence
        ax2 = axes[1]
        ax2.plot(moead.history['generation'], moead.history['igd'], 'b-o', label='IGD')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('IGD')
        ax2.set_title('Convergence History')
        ax2.grid(True, alpha=0.3)
        
        ax2_right = ax2.twinx()
        ax2_right.plot(moead.history['generation'], moead.history['beyond_limit_count'], 
                      'r-s', label='Beyond-limit')
        ax2_right.set_ylabel('Beyond-limit Count', color='red')
        
        plt.tight_layout()
        plt.savefig('moead_results.png', dpi=150)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
        
    print("\nMOEA/D Optimization Complete!")
