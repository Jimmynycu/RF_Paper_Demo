#!/usr/bin/env python3
"""
Spectral Eclipse - Main Run Script

Run all three paper implementations:
1. Paper 1: Chu-Limit-Guided MOEA/D for ESA Design
2. Paper 2: PINN for FSS Inverse Design
3. Paper 3: LLM-RIMSA for Metasurface Antenna Control

Usage:
    python run_all.py [--paper 1|2|3|all] [--quick]
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def check_dependencies():
    """Check and report on available dependencies."""
    deps = {
        'numpy': False,
        'torch': False,
        'tqdm': False,
        'matplotlib': False,
        'transformers': False
    }
    
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        pass
        
    try:
        import torch
        deps['torch'] = True
    except ImportError:
        pass
        
    try:
        import tqdm
        deps['tqdm'] = True
    except ImportError:
        pass
        
    try:
        import matplotlib
        deps['matplotlib'] = True
    except ImportError:
        pass
        
    try:
        import transformers
        deps['transformers'] = True
    except ImportError:
        pass
        
    return deps


def run_paper1(quick: bool = False):
    """Run Paper 1: Chu-Limit-Guided MOEA/D for ESA Design."""
    print("\n" + "=" * 70)
    print("PAPER 1: Chu-Limit-Guided MOEA/D for ESA Design")
    print("=" * 70)
    
    from paper1_chu_limit_moead.moead import MOEADConfig, MOEAD
    from paper1_chu_limit_moead.evaluator import AntennaEvaluator
    from paper1_chu_limit_moead.chu_limit import ChuLimitCalculator
    
    # Configuration (reduced for quick mode)
    pop_size = 10 if quick else 50
    generations = 5 if quick else 20
    
    config = MOEADConfig(
        population_size=pop_size,
        max_generations=generations,
        neighborhood_size=3,
        seed=42
    )
    
    evaluator = AntennaEvaluator(use_coarse_mesh=True)
    chu_calculator = ChuLimitCalculator()
    
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}")
    print()
    
    # Run optimization
    moead = MOEAD(config, evaluator, chu_calculator)
    pareto_front = moead.run()
    
    # Results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Pareto front size: {len(pareto_front)}")
    print(f"Beyond-limit solutions: {moead.count_beyond_limit()}")
    
    if pareto_front:
        print("\nTop solutions by bandwidth:")
        for i, ind in enumerate(sorted(pareto_front, key=lambda x: -x.performance.bandwidth)[:3]):
            print(f"  {i+1}. ka={ind.performance.ka:.3f}, BW={ind.performance.bandwidth:.1f}%, "
                  f"η={ind.performance.total_efficiency*100:.1f}%")
                  
    return moead


def run_paper2(quick: bool = False):
    """Run Paper 2: PINN for FSS Inverse Design."""
    print("\n" + "=" * 70)
    print("PAPER 2: PINN for FSS Inverse Design")
    print("=" * 70)
    
    from paper2_pinn_fss.fss_designer import FSSDesigner, TrainingConfig, DesignGoal
    from paper2_pinn_fss.mode_matching import FSSParameters
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuration
    steps = 500 if quick else 5000
    
    fss_params = FSSParameters(
        unit_cell_size=10e-3,
        dielectric_thickness=2e-3,
        relative_permittivity=3.2
    )
    
    design_goal = DesignGoal.bandstop_at_frequency(
        center_freq=15e9,
        passband_freqs=(12e9, 18e9),
        device=device
    )
    
    training_config = TrainingConfig(
        max_steps=steps,
        learning_rate=1e-3,
        grid_resolution=32 if quick else 64,
        device=device
    )
    
    print(f"Device: {device}")
    print(f"Training steps: {steps}")
    print(f"Target: Band-stop at 15 GHz")
    print()
    
    # Create designer and train
    designer = FSSDesigner(fss_params, design_goal, training_config)
    result = designer.train(verbose=True)
    
    # Results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Design error: {result.design_error:.4f}")
    print(f"Training time: {result.training_time:.1f} seconds")
    
    print("\nS21 Performance:")
    for freq, s21 in result.achieved_s21.items():
        print(f"  {freq/1e9:.0f} GHz: {s21:.3f}")
        
    return result, designer


def run_paper3(quick: bool = False):
    """Run Paper 3: LLM-RIMSA for Metasurface Antenna Control."""
    print("\n" + "=" * 70)
    print("PAPER 3: LLM-RIMSA for Metasurface Antenna Control")
    print("=" * 70)
    
    from paper3_llm_rimsa.trainer import run_llm_rimsa_training
    
    # Configuration
    epochs = 5 if quick else 50
    
    print(f"Training epochs: {epochs}")
    print()
    
    # Run training
    trainer, benchmark = run_llm_rimsa_training(n_epochs=epochs)
    
    # Results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"LLM-RIMSA Sum Rate: {benchmark['llm_rimsa_mean']:.2f} ± {benchmark['llm_rimsa_std']:.2f} bps/Hz")
    print(f"Random Phase:       {benchmark['random_phase_mean']:.2f} ± {benchmark['random_phase_std']:.2f} bps/Hz")
    print(f"ZF Only:            {benchmark['zf_only_mean']:.2f} ± {benchmark['zf_only_std']:.2f} bps/Hz")
    
    improvement = (benchmark['llm_rimsa_mean'] / benchmark['random_phase_mean'] - 1) * 100
    print(f"\nImprovement over random: {improvement:.1f}%")
    
    return trainer, benchmark


def main():
    parser = argparse.ArgumentParser(
        description='Run Spectral Eclipse paper implementations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Run all papers with full settings
  python run_all.py --quick            # Quick run with reduced settings
  python run_all.py --paper 1          # Run only Paper 1
  python run_all.py --paper 2 --quick  # Quick run of Paper 2
        """
    )
    
    parser.add_argument('--paper', type=str, default='all',
                       choices=['1', '2', '3', 'all'],
                       help='Which paper to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with reduced settings')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SPECTRAL ECLIPSE - Research Paper Implementations")
    print("=" * 70)
    
    # Check dependencies
    deps = check_dependencies()
    print("\nDependency Status:")
    for pkg, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {pkg}")
    
    print(f"\nPaper Requirements:")
    print(f"  Paper 1 (MOEA/D): numpy, tqdm {'- Ready!' if deps['numpy'] and deps['tqdm'] else '- Missing deps'}")
    print(f"  Paper 2 (PINN):   torch {'- Ready!' if deps['torch'] else '- Install: pip install torch'}")
    print(f"  Paper 3 (LLM):    torch {'- Ready!' if deps['torch'] else '- Install: pip install torch'}")
        
    print(f"\nMode: {'Quick' if args.quick else 'Full'}")
    print(f"Papers to run: {args.paper}")
    
    start_time = time.time()
    
    results = {}
    
    if args.paper in ['1', 'all']:
        try:
            results['paper1'] = run_paper1(quick=args.quick)
        except Exception as e:
            print(f"Paper 1 failed: {e}")
            import traceback
            traceback.print_exc()
            
    if args.paper in ['2', 'all']:
        try:
            results['paper2'] = run_paper2(quick=args.quick)
        except Exception as e:
            print(f"Paper 2 failed: {e}")
            import traceback
            traceback.print_exc()
            
    if args.paper in ['3', 'all']:
        try:
            results['paper3'] = run_paper3(quick=args.quick)
        except Exception as e:
            print(f"Paper 3 failed: {e}")
            import traceback
            traceback.print_exc()
            
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f} seconds")
    
    return results


if __name__ == "__main__":
    main()
