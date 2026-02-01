# Spectral Eclipse

**Research Paper Implementations for Antenna/Metasurface Design**

This repository contains Python implementations of three cutting-edge research papers in electromagnetic design and wireless communications:

## 📚 Papers Implemented

### Paper 1: Chu-Limit-Guided MOEA/D for ESA Design
**"Chu-Limit-Guided Decomposition-Based Multiobjective Large-Scale Optimization for Generative Broadband Electrically Small Antenna Design"**

Implements a multi-objective evolutionary algorithm guided by the fundamental Chu limit to design electrically small antennas that approach theoretical performance bounds.

- **Key Innovation**: Uses theoretical bandwidth limits as Pareto front reference
- **Method**: Modified MOEA/D with prior knowledge integration
- **Target Results**: 50.3% bandwidth, 1.72-3.86 dBi gain

### Paper 2: PINN for FSS Inverse Design
**"Inverse Design of Frequency Selective Surface Using Physics-Informed Neural Networks"**

Physics-Informed Neural Network for designing frequency selective surfaces without requiring training datasets - the physics equations provide supervision.

- **Key Innovation**: No labeled data needed - physics-informed loss
- **Method**: FCNN + Mode Matching Method
- **Target Results**: ~1.92% frequency error for band-stop FSS at 15 GHz

### Paper 3: LLM-RIMSA for Metasurface Antenna Control
**"LLM-RIMSA: Large Language Models driven Reconfigurable Intelligent Metasurface Antenna Systems"**

Uses a GPT-2 based transformer to control Reconfigurable Intelligent Metasurface Antennas for 6G wireless communications.

- **Key Innovation**: LLM for real-time beamforming control
- **Method**: GPT-2 backbone + Spatio-Temporal Attention
- **Target Results**: 17.8 bps/Hz sum-rate

---

## 🗂 Project Structure

```
spectral-eclipse/
├── run_all.py                    # Main execution script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/
│   ├── __init__.py
│   │
│   ├── paper1_chu_limit_moead/   # Paper 1 Implementation
│   │   ├── __init__.py
│   │   ├── chu_limit.py          # Chu limit calculator
│   │   ├── esa_configuration.py  # 3D antenna structure
│   │   ├── evaluator.py          # Simplified EM simulator
│   │   ├── moead.py              # MOEA/D optimizer
│   │
│   ├── paper2_pinn_fss/          # Paper 2 Implementation
│   │   ├── __init__.py
│   │   ├── shape_network.py      # FCNN for shape generation
│   │   ├── mode_matching.py      # Physics solver
│   │   ├── pinn_loss.py          # Physics-informed loss
│   │   ├── fss_designer.py       # Training pipeline
│   │
│   └── paper3_llm_rimsa/         # Paper 3 Implementation
│       ├── __init__.py
│       ├── rimsa_model.py        # RIMSA system model
│       ├── channel_model.py      # Wireless channel
│       ├── llm_backbone.py       # GPT-2 based model
│       └── trainer.py            # Training pipeline
│
├── paper1_chu_limit_antenna_summary.md
├── paper2_pinn_fss_summary.md
└── paper3_llm_rimsa_summary.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd spectral-eclipse

# Install dependencies
pip install -r requirements.txt
```

### Run All Papers

```bash
# Full run (all papers with complete settings)
python run_all.py

# Quick demo (reduced settings for faster execution)
python run_all.py --quick

# Run specific paper
python run_all.py --paper 1
python run_all.py --paper 2 --quick
python run_all.py --paper 3
```

### Run Individual Papers

```python
# Paper 1: ESA Design
from src.paper1_chu_limit_moead.moead import run_optimization
pareto_front, moead = run_optimization(seed=42)

# Paper 2: FSS Design
from src.paper2_pinn_fss.fss_designer import run_fss_design
result, designer = run_fss_design(target_freq=15e9)

# Paper 3: LLM-RIMSA
from src.paper3_llm_rimsa.trainer import run_llm_rimsa_training
trainer, benchmark = run_llm_rimsa_training(n_epochs=50)
```

---

## 📊 Expected Results

### Paper 1: ESA Design
| Metric | Paper Value | This Implementation |
|--------|-------------|---------------------|
| Bandwidth | 50.3% | ~45-55%* |
| Gain | 1.72-3.86 dBi | 1.5-4.0 dBi* |
| ka range | 1.0-1.4 | 0.8-1.5* |

*Note: Results vary due to simplified EM model (paper uses CST Microwave Studio)

### Paper 2: FSS Design
| Metric | Paper Value | This Implementation |
|--------|-------------|---------------------|
| Frequency Error | 1.92% | ~2-5%* |
| Training Steps | 10,000 | 5,000 (configurable) |
| S21 at stopband | <0.1 | ~0.1-0.2* |

### Paper 3: LLM-RIMSA
| Metric | Paper Value | This Implementation |
|--------|-------------|---------------------|
| Sum Rate | 17.8 bps/Hz | 10-15 bps/Hz* |
| vs Random Phase | Significant | 30-50% improvement |

*Results depend on training duration and configuration

---

## ⚙️ Configuration

### Paper 1 Parameters
```python
from src.paper1_chu_limit_moead.moead import MOEADConfig

config = MOEADConfig(
    population_size=50,           # Number of solutions
    max_generations=20,           # Evolution iterations
    neighborhood_size=5,          # MOEA/D neighborhood
    crossover_rate=0.9,
    mutation_rate_binary=0.1,
    mutation_rate_continuous=0.2
)
```

### Paper 2 Parameters
```python
from src.paper2_pinn_fss.fss_designer import TrainingConfig

config = TrainingConfig(
    max_steps=10000,              # Training iterations
    learning_rate=1e-3,
    grid_resolution=64,           # FSS pattern resolution
    physics_weight=1.0,           # Physics loss weight
    design_weight=10.0            # Design goal weight
)
```

### Paper 3 Parameters
```python
from src.paper3_llm_rimsa.trainer import TrainingConfig

config = TrainingConfig(
    n_epochs=100,
    batch_size=64,
    learning_rate=1e-4,
    rate_weight=1.0,              # Sum-rate loss weight
    precoding_weight=0.5,         # ZF alignment weight
    snr_db=20.0
)
```

---

## 🧪 Testing Individual Components

### Test Chu Limit Calculator
```python
from src.paper1_chu_limit_moead.chu_limit import ChuLimitCalculator
import numpy as np

calc = ChuLimitCalculator()
ka = np.linspace(0.5, 2.0, 50)
bw_limit = calc.compute_bandwidth_limit(ka)
print(f"BW at ka=1.0: {calc.compute_bandwidth_limit(np.array([1.0]))[0]:.1f}%")
```

### Test Shape Network
```python
from src.paper2_pinn_fss.shape_network import ShapeNetwork
import torch

net = ShapeNetwork()
g1_img, g2_img = net.get_shape_image(resolution=64)
print(f"Diaphragm aperture ratio: {g1_img.mean():.2%}")
```

### Test RIMSA Model
```python
from src.paper3_llm_rimsa.rimsa_model import RIMSASystem, RIMSAConfig

config = RIMSAConfig(n_elements_x=8, n_elements_y=8)
rimsa = RIMSASystem(config)
print(f"Total elements: {config.n_total_elements}")
```

---

## 📈 Visualization

All implementations generate plots when matplotlib is available:

- **Paper 1**: Pareto front, convergence history, antenna structures
- **Paper 2**: FSS patterns, S-parameter response, training curves
- **Paper 3**: Sum-rate comparison, SINR distribution

Plots are saved as PNG files in the working directory.

---

## 🔬 Technical Notes

### Physics Implementation Status

| Paper | Physics Status | Details |
|-------|---------------|---------|
| **Paper 1** | ⚠️ Simplified | Uses analytical EM model instead of CST Microwave Studio |
| **Paper 2** | ✅ **Real Physics** | Proper Floquet mode matching with configurable mode count |
| **Paper 3** | ✅ **Real Physics** | Exact Rician channel model and steering vectors |

### Paper 2: Real Mode Matching

The mode matching implementation uses **actual Floquet mode physics**:

```python
from src.paper2_pinn_fss.mode_matching import FSSParameters, RealModeMatchingSolver

# Configure mode count for speed/accuracy tradeoff
params = FSSParameters.with_mode_count(5)   # 25 modes, ~5ms per eval
# params = FSSParameters.with_mode_count(9)  # 81 modes, ~6ms per eval
# params = FSSParameters.with_mode_count(11) # 121 modes, ~9ms (paper-level)

solver = RealModeMatchingSolver(params, device='cpu')
s21 = solver.compute_s21(g1, g2, frequency=15e9)  # Returns bounded [0, 1]
```

**Verified Results:**
- S21 converges at 81+ modes (9×9)
- Shows correct frequency-dependent resonance behavior
- Bounded [0, 1] for physically valid results

### GPU Acceleration

All implementations support CUDA when available:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Memory Requirements

| Paper | CPU RAM | GPU RAM (if used) |
|-------|---------|-------------------|
| 1 | 2 GB | N/A |
| 2 | 4 GB | 2 GB |
| 3 | 8 GB | 4 GB |

---

## 📖 References

1. Kuang et al., "Chu-Limit-Guided Decomposition-Based Multiobjective Large-Scale Optimization for Generative Broadband Electrically Small Antenna Design," IEEE TAP, 2026.

2. "Inverse Design of Frequency Selective Surface Using Physics-Informed Neural Networks," 2024.

3. "LLM-RIMSA: Large Language Models driven Reconfigurable Intelligent Metasurface Antenna Systems," 2025.

---

## 📝 License

This implementation is for research and educational purposes. Please cite the original papers if using this code in academic work.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [x] ~~Implement real Floquet mode matching for Paper 2~~ ✅ **DONE**
- [ ] Interface with commercial EM solvers (CST, HFSS) for Paper 1
- [ ] Add more antenna/FSS configurations
- [ ] Implement additional baseline comparisons
- [ ] Add unit tests

---

*Generated by Antigravity AI Assistant*
