#!/usr/bin/env python3
"""
Educational Antenna Research Demo - FastAPI Backend

Server-Sent Events (SSE) streaming for real-time training visualization.
Supports all three papers with educational annotations.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")

# =============================================================================
# Data Models for Educational Content
# =============================================================================

@dataclass
class EducationalStep:
    """A single step in the learning process with annotations."""
    step_id: int
    step_type: str  # "concept", "equation", "visualization", "demo"
    title: str
    content: str
    equation: Optional[str] = None  # LaTeX
    visualization_data: Optional[dict] = None
    
@dataclass
class GlossaryTerm:
    """Fact-checked antenna terminology."""
    term: str
    simple: str
    formula: Optional[str]
    analogy: Optional[str]
    verified_source: str

# =============================================================================
# Fact-Checked Glossary (IEEE/Wikipedia verified)
# =============================================================================

GLOSSARY = {
    "ka": GlossaryTerm(
        term="Electrical Size (ka)",
        simple="How big the antenna is compared to the wavelength. ka = (2π/λ) × radius",
        formula="ka = \\frac{2\\pi}{\\lambda} \\cdot a = \\frac{2\\pi f}{c} \\cdot a",
        analogy="Like comparing a drum to a sound wave - small drum = high frequency",
        verified_source="IEEE, Wikipedia - Electrically small antenna"
    ),
    "q_factor": GlossaryTerm(
        term="Quality Factor (Q)",
        simple="How picky the antenna is about frequency. Higher Q = narrower bandwidth",
        formula="Q \\approx \\frac{\\text{Energy Stored}}{\\text{Energy Radiated per cycle}}",
        analogy="Like a tuning fork: high Q rings longer but only at one exact note",
        verified_source="IEEE, RF engineering textbooks"
    ),
    "chu_limit": GlossaryTerm(
        term="Chu-Harrington Limit",
        simple="The theoretical minimum Q for a given antenna size. Physics says you can't beat this!",
        formula="Q_{min} \\geq \\frac{1}{(ka)^3} + \\frac{1}{ka}",
        analogy="Like a speed limit - physics won't let you go faster",
        verified_source="Chu 1948, Harrington 1960, IEEE TAP"
    ),
    "bandwidth": GlossaryTerm(
        term="Bandwidth",
        simple="The range of frequencies where the antenna works well",
        formula="BW = \\frac{f_{high} - f_{low}}{f_{center}} \\times 100\\%",
        analogy="Like a radio dial - wider bandwidth = more stations you can tune",
        verified_source="IEEE, ARRL Antenna Handbook"
    ),
    "fss": GlossaryTerm(
        term="Frequency Selective Surface (FSS)",
        simple="A flat surface with metal patterns that lets some frequencies through and blocks others",
        formula=None,
        analogy="Like a strainer in the kitchen - lets water through but keeps pasta",
        verified_source="IEEE, MDPI journals"
    ),
    "pinn": GlossaryTerm(
        term="Physics-Informed Neural Network",
        simple="A neural network that uses physics equations in its loss function instead of training data",
        formula="\\mathcal{L} = \\|\\text{physics residual}\\|^2",
        analogy="Like a student who learns by checking if their answer obeys the laws of physics",
        verified_source="Raissi et al. 2019, Nature"
    ),
    "ris": GlossaryTerm(
        term="Reconfigurable Intelligent Surface (RIS)",
        simple="A smart wall with tiny adjustable elements that can steer radio waves",
        formula=None,
        analogy="Like a disco ball that can aim each mirror independently",
        verified_source="IEEE, MathWorks, 6G Academy"
    ),
    "beamforming": GlossaryTerm(
        term="Beamforming",
        simple="Focusing radio signals in specific directions using an array of antennas",
        formula="y = \\sum_n w_n \\cdot e^{j\\phi_n} \\cdot x_n",
        analogy="Like a flashlight vs a bare bulb - same power but directed",
        verified_source="IEEE, Analog Devices"
    ),
    "sinr": GlossaryTerm(
        term="SINR (Signal-to-Interference-plus-Noise Ratio)",
        simple="How strong your signal is compared to interference and noise",
        formula="\\text{SINR} = \\frac{|\\text{desired signal}|^2}{|\\text{interference}|^2 + \\sigma^2}",
        analogy="Like trying to hear someone at a loud party",
        verified_source="IEEE, wireless communication textbooks"
    ),
}

# =============================================================================
# FastAPI Application
# =============================================================================

if HAS_FASTAPI:
    app = FastAPI(
        title="Antenna Research Demo",
        description="Educational demo for Chu-Limit ESA, PINN FSS, and LLM-RIMSA papers",
        version="1.0.0"
    )
    
    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Serve static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    # Serve paper images
    paper_images_path = Path(__file__).parent / "paper_images"
    if paper_images_path.exists():
        app.mount("/paper_images", StaticFiles(directory=str(paper_images_path)), name="paper_images")
    
    # ==========================================================================
    # API Endpoints
    # ==========================================================================
    
    @app.get("/")
    async def root():
        """Serve the main demo dashboard."""
        dashboard_path = Path(__file__).parent / "demo_dashboard.html"
        if dashboard_path.exists():
            return FileResponse(dashboard_path)
        return HTMLResponse("<h1>Demo Dashboard not found. Run the setup first.</h1>")
    
    @app.get("/api/glossary")
    async def get_glossary():
        """Get all fact-checked terminology."""
        return {k: asdict(v) for k, v in GLOSSARY.items()}
    
    @app.get("/api/glossary/{term}")
    async def get_term(term: str):
        """Get a specific glossary term."""
        if term in GLOSSARY:
            return asdict(GLOSSARY[term])
        return {"error": f"Term '{term}' not found"}
    
    # ==========================================================================
    # Paper 1: Chu-Limit MOEA/D Streaming
    # ==========================================================================
    
    @app.get("/api/paper1/stream")
    async def stream_paper1(
        quick: bool = Query(True, description="Use quick mode for faster demo"),
        population_size: int = Query(20, description="Population size"),
        max_generations: int = Query(10, description="Max generations")
    ):
        """Stream MOEA/D optimization generation-by-generation."""
        
        async def generate() -> AsyncGenerator[str, None]:
            try:
                from paper1_chu_limit_moead.moead import MOEAD, MOEADConfig
                from paper1_chu_limit_moead.evaluator import AntennaEvaluator
                from paper1_chu_limit_moead.chu_limit import ChuLimitCalculator
                
                # Configure based on quick mode
                if quick:
                    config = MOEADConfig(
                        population_size=min(population_size, 15),
                        max_generations=min(max_generations, 8),
                        seed=42
                    )
                else:
                    config = MOEADConfig(
                        population_size=population_size,
                        max_generations=max_generations,
                        seed=42
                    )
                
                evaluator = AntennaEvaluator()
                chu_calc = ChuLimitCalculator()
                moead = MOEAD(config, evaluator, chu_calc)
                
                # Initialize and yield initial state
                moead.initialize()
                yield f"data: {json.dumps({'type': 'init', 'population_size': config.population_size, 'max_generations': config.max_generations})}\n\n"
                
                # Run optimization with streaming
                for gen in range(config.max_generations):
                    # Evolution step
                    for i in range(config.population_size):
                        pool = [moead.population[j] for j in moead.neighborhoods[i]]
                        offspring = moead.offspring_reproduction(moead.population[i], pool)
                        moead.update_neighbors(offspring, i)
                    
                    moead.population_reassignment()
                    
                    # Compute metrics
                    igd = moead.compute_igd()
                    beyond_count = moead.count_beyond_limit()
                    
                    # Serialize population for visualization
                    pop_data = []
                    for ind in moead.population:
                        if ind.performance:
                            chu_bw = chu_calc.compute_bandwidth_limit_single(ind.performance.ka)
                            pop_data.append({
                                "ka": float(ind.performance.ka),
                                "bandwidth": float(ind.performance.fractional_bandwidth * 100),
                                "chu_limit_bw": float(chu_bw),
                                "beyond_limit": ind.performance.fractional_bandwidth * 100 > chu_bw
                            })
                    
                    # Educational annotation
                    annotation = ""
                    if gen == 0:
                        annotation = "Initial random population - most solutions are far from the Chu limit"
                    elif gen < config.max_generations // 3:
                        annotation = "Early evolution - solutions spreading across the Pareto front"
                    elif gen < 2 * config.max_generations // 3:
                        annotation = "Mid evolution - solutions converging toward the Chu limit curve"
                    else:
                        annotation = "Late evolution - some solutions may exceed the theoretical limit!"
                    
                    yield f"data: {json.dumps({'type': 'generation', 'generation': gen + 1, 'population': pop_data, 'igd': igd, 'beyond_limit_count': beyond_count, 'annotation': annotation})}\n\n"
                    
                    await asyncio.sleep(0.5)  # Pause for visualization
                
                yield f"data: {json.dumps({'type': 'complete', 'message': 'Optimization finished!'})}\n\n"
                
            except ImportError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Import error: {str(e)}. Make sure all dependencies are installed.'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # ==========================================================================
    # Paper 2: PINN FSS Streaming
    # ==========================================================================
    
    @app.get("/api/paper2/stream")
    async def stream_paper2(
        quick: bool = Query(True, description="Use quick mode"),
        target_freq_ghz: float = Query(15.0, description="Target stop frequency in GHz"),
        max_steps: int = Query(500, description="Max training steps")
    ):
        """Stream PINN training step-by-step."""
        
        async def generate() -> AsyncGenerator[str, None]:
            try:
                import torch
                from paper2_pinn_fss.fss_designer import FSSDesigner, TrainingConfig
                from paper2_pinn_fss.mode_matching import FSSParameters
                from paper2_pinn_fss.pinn_loss import DesignGoal
                import base64
                import io
                
                # Configure
                fss_params = FSSParameters()
                
                # Create design goal
                target_freq = target_freq_ghz * 1e9
                passband_low = (target_freq_ghz - 3) * 1e9
                passband_high = (target_freq_ghz + 3) * 1e9
                
                design_goal = DesignGoal(
                    frequencies=[passband_low, target_freq, passband_high],
                    target_s21=[0.95, 0.1, 0.95]  # Pass, Block, Pass
                )
                
                if quick:
                    config = TrainingConfig(
                        max_steps=min(max_steps, 300),
                        log_interval=30,
                        grid_resolution=32
                    )
                else:
                    config = TrainingConfig(
                        max_steps=max_steps,
                        log_interval=50,
                        grid_resolution=64
                    )
                
                designer = FSSDesigner(fss_params, design_goal, config)
                
                yield f"data: {json.dumps({'type': 'init', 'target_freq_ghz': target_freq_ghz, 'max_steps': config.max_steps})}\n\n"
                
                # Training loop with streaming
                optimizer = torch.optim.Adam(designer.network.parameters(), lr=config.learning_rate)
                
                for step in range(config.max_steps):
                    optimizer.zero_grad()
                    
                    # Get shape
                    g1, g2 = designer.network.get_shape_image(config.grid_resolution)
                    
                    # Compute losses
                    loss_dict = designer.loss_fn(g1, g2)
                    total_loss = loss_dict['total']
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    # Stream at intervals
                    if step % config.log_interval == 0 or step == config.max_steps - 1:
                        # Get pattern as base64 for visualization
                        with torch.no_grad():
                            g1_img, g2_img = designer.network.get_shape_image(32)
                            pattern = g1_img.detach().cpu().numpy().tolist()
                        
                        # Educational annotation
                        if step < config.max_steps // 4:
                            annotation = "Early training - network is learning basic patterns"
                        elif step < config.max_steps // 2:
                            annotation = "Physics loss guiding toward valid EM structure"
                        elif step < 3 * config.max_steps // 4:
                            annotation = "Design loss pushing S21 toward target"
                        else:
                            annotation = "Refinement - pattern approaching final form"
                        
                        yield f"data: {json.dumps({'type': 'step', 'step': step, 'loss_total': float(total_loss.item()), 'loss_physics': float(loss_dict.get('physics', 0)), 'loss_design': float(loss_dict.get('design', 0)), 'pattern': pattern, 'annotation': annotation})}\n\n"
                        
                        await asyncio.sleep(0.1)
                
                yield f"data: {json.dumps({'type': 'complete', 'message': 'Training finished!'})}\n\n"
                
            except ImportError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Import error: {str(e)}'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # ==========================================================================
    # Paper 3: LLM-RIMSA Streaming
    # ==========================================================================
    
    @app.get("/api/paper3/stream")
    async def stream_paper3(
        quick: bool = Query(True, description="Use quick mode"),
        n_users: int = Query(4, description="Number of users"),
        n_epochs: int = Query(20, description="Number of epochs")
    ):
        """Stream LLM-RIMSA training epoch-by-epoch."""
        
        async def generate() -> AsyncGenerator[str, None]:
            try:
                import torch
                from paper3_llm_rimsa.trainer import LLMRIMSATrainer, TrainingConfig
                from paper3_llm_rimsa.rimsa_model import RIMSAConfig
                from paper3_llm_rimsa.channel_model import ChannelConfig
                from paper3_llm_rimsa.llm_backbone import LLMConfig
                
                # Configure for quick demo
                if quick:
                    rimsa_config = RIMSAConfig(n_elements_x=4, n_elements_y=4, n_rf_chains=2)
                    channel_config = ChannelConfig(n_users=min(n_users, 4))
                    llm_config = LLMConfig(d_model=64, n_layers=2, n_heads=4)
                    training_config = TrainingConfig(
                        n_epochs=min(n_epochs, 15),
                        batch_size=16
                    )
                else:
                    rimsa_config = RIMSAConfig(n_elements_x=8, n_elements_y=8, n_rf_chains=4)
                    channel_config = ChannelConfig(n_users=n_users)
                    llm_config = LLMConfig(d_model=128, n_layers=4, n_heads=8)
                    training_config = TrainingConfig(
                        n_epochs=n_epochs,
                        batch_size=32
                    )
                
                trainer = LLMRIMSATrainer(llm_config, rimsa_config, channel_config, training_config)
                
                yield f"data: {json.dumps({'type': 'init', 'n_users': channel_config.n_users, 'n_elements': rimsa_config.n_total_elements, 'n_epochs': training_config.n_epochs})}\n\n"
                
                # Training with streaming
                for epoch in range(training_config.n_epochs):
                    metrics = trainer.train_epoch(n_batches=20)
                    
                    # Get phase configuration for visualization
                    with torch.no_grad():
                        sample_channel = trainer.channel_gen.generate_batch(1)
                        phase, _ = trainer.model(sample_channel)
                        phase_config = phase[0].cpu().numpy().tolist()
                    
                    # Educational annotation
                    if epoch < training_config.n_epochs // 3:
                        annotation = "Early learning - model exploring phase configurations"
                    elif epoch < 2 * training_config.n_epochs // 3:
                        annotation = "Transformer learning user-channel relationships"
                    else:
                        annotation = "Refinement - optimizing multi-user SINR balance"
                    
                    yield f"data: {json.dumps({'type': 'epoch', 'epoch': epoch + 1, 'sum_rate': float(metrics['sum_rate']), 'loss': float(metrics['loss']), 'phase_config': phase_config[:16], 'annotation': annotation})}\n\n"
                    
                    await asyncio.sleep(0.3)
                
                # Benchmark comparison
                benchmark = trainer.benchmark_vs_baselines(n_samples=100)
                
                yield f"data: {json.dumps({'type': 'benchmark', 'random_phase': float(benchmark['random_phase']), 'zf_only': float(benchmark['zf_only']), 'llm_rimsa': float(benchmark['llm_rimsa'])})}\n\n"
                
                yield f"data: {json.dumps({'type': 'complete', 'message': 'Training finished!'})}\n\n"
                
            except ImportError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Import error: {str(e)}'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    if not HAS_FASTAPI:
        print("Please install FastAPI: pip install fastapi uvicorn[standard]")
        return
    
    print("=" * 60)
    print("📡 Antenna Research Demo Server")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("\nEndpoints:")
    print("  GET /                    - Demo Dashboard")
    print("  GET /api/glossary        - Fact-checked terminology")
    print("  GET /api/paper1/stream   - Chu-Limit MOEA/D (SSE)")
    print("  GET /api/paper2/stream   - PINN FSS (SSE)")
    print("  GET /api/paper3/stream   - LLM-RIMSA (SSE)")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
