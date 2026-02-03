# Antenna AI Engineer Interview - Slide Summary

## Overview
21-slide presentation covering 3 research papers bridging RF engineering and AI/ML.

---

## Paper 1: Chu-Limit ESA (Slides 4-8)

### Problem
- Small antennas (ka < 0.5) have narrow bandwidth due to physics
- Wheeler's Radiansphere (1947): near-field stores energy
- Chu Limit (1948): fundamental Q vs size tradeoff

### Solution: MOEA/D + Chu Limit
- 135+ dimensional optimization (27 rods × 5 params)
- Tchebycheff decomposition with Pareto front on Chu limit
- Coarse→Fine EM evaluation (10x faster)

### Results
| Metric | Achieved |
|--------|----------|
| Bandwidth | **50.3%** |
| Beyond Chu | **3.3%** |
| BW-Efficiency | 0.402 |

---

## Paper 2: PINN FSS (Slides 9-13)

### Problem
- FSS inverse design needs 100,000+ simulations
- Traditional NN: months of data generation

### Solution: Physics-Informed Neural Network
- FCNN: 3 layers × 32 neurons
- Dual loss: Physics (Maxwell residual) + Design (S21 target)
- Zero-data paradigm: Maxwell's equations = free supervision

### Results
| Metric | Value |
|--------|-------|
| Target | 15 GHz |
| Achieved | 15.288 GHz |
| Error | **1.92%** |
| Training | ~1 hour (vs months) |

---

## Paper 3: LLM-RIMSA (Slides 14-18)

### Problem
- 6G mmWave needs real-time RIS control
- 256 elements × 2π phase = huge search space
- Traditional optimization: ~100ms (too slow)

### Solution: GPT-2 Backbone + SE Attention
- 6 transformer layers, 8-head self-attention
- d_model = 256
- One-shot inference vs iterative optimization

### Results
| Method | Sum Rate | Inference |
|--------|----------|-----------|
| SDR/Alternating | Optimal | ~100ms |
| **LLM-RIMSA** | **96% optimal** | **<1ms** |

---

## Key Takeaways

1. **Paper 1**: Use physics limits as optimization guides, not barriers
2. **Paper 2**: PINNs eliminate data requirements via physics supervision  
3. **Paper 3**: Transformers enable real-time control for complex RF systems

---

## Navigation
- **↑/↓ keys**: Navigate slides
- **Home/End**: Jump to first/last slide
- **Click**: Next slide
- **🔴 Laser**: Presentation pointer
