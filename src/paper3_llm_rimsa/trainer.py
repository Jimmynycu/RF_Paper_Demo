"""
LLM-RIMSA Trainer - Training and Inference Pipeline

Implements the complete training workflow:
1. Data generation and loading
2. Hybrid loss computation (rate + precoding + Frobenius)
3. Training loop with monitoring
4. Evaluation and benchmarking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from tqdm import tqdm
import time

try:
    from .rimsa_model import RIMSASystem, RIMSAConfig
    from .channel_model import ChannelGenerator, ChannelConfig
    from .llm_backbone import LLMRIMSAModel, LLMConfig
except ImportError:
    from rimsa_model import RIMSASystem, RIMSAConfig
    from channel_model import ChannelGenerator, ChannelConfig
    from llm_backbone import LLMRIMSAModel, LLMConfig


@dataclass
class TrainingConfig:
    """Configuration for LLM-RIMSA training."""
    # Training parameters
    batch_size: int = 64
    n_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights (from paper)
    rate_weight: float = 1.0
    precoding_weight: float = 0.5
    frobenius_weight: float = 0.1
    
    # Noise parameters
    snr_db: float = 20.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    
    # Device
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
    @property
    def noise_power(self) -> float:
        return 10 ** (-self.snr_db / 10)


class HybridLoss(nn.Module):
    """
    Hybrid Loss Function for LLM-RIMSA.
    
    From paper Eq. (12):
    L = α*L_rate + β*L_precoding + γ*L_F
    
    where:
    - L_rate = -R_sum (negative sum rate)
    - L_precoding = ||W - W_ZF||² (precoding deviation from ZF)
    - L_F = ||V||_F (Frobenius regularization on beamforming)
    """
    
    def __init__(self, 
                 rimsa: RIMSASystem,
                 rate_weight: float = 1.0,
                 precoding_weight: float = 0.5,
                 frobenius_weight: float = 0.1,
                 noise_power: float = 1e-10):
        super().__init__()
        
        self.rimsa = rimsa
        self.rate_weight = rate_weight
        self.precoding_weight = precoding_weight
        self.frobenius_weight = frobenius_weight
        self.noise_power = noise_power
        
    def forward(self,
                phase: torch.Tensor,
                precoding: torch.Tensor,
                channel: torch.Tensor,
                V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss.
        
        Args:
            phase: Phase matrix (batch, n_rf, n_elements)
            precoding: Precoding matrix (batch, n_rf, n_users)
            channel: Channel matrix (batch, n_users, n_total_elements)
            V: Beamforming matrix (batch, n_total_elements, n_rf)
            
        Returns:
            Dictionary with individual losses and total
        """
        batch_size = channel.shape[0]
        total_rate_loss = torch.tensor(0.0, device=channel.device)
        total_precoding_loss = torch.tensor(0.0, device=channel.device)
        total_frobenius_loss = torch.tensor(0.0, device=channel.device)
        
        for b in range(batch_size):
            H = channel[b]  # (n_users, n_total_elements)
            V_b = V[b]  # (n_total_elements, n_rf)
            W = precoding[b]  # (n_rf, n_users)
            
            # Rate loss: negative sum rate
            sum_rate = self.rimsa.compute_sum_rate(H, V_b, W, self.noise_power)
            rate_loss = -sum_rate
            total_rate_loss = total_rate_loss + rate_loss
            
            # Precoding loss: deviation from ZF
            W_zf = self.rimsa.compute_zf_precoding(H, V_b)
            precoding_loss = torch.norm(W - W_zf) ** 2
            total_precoding_loss = total_precoding_loss + precoding_loss
            
            # Frobenius loss: regularization
            frobenius_loss = torch.norm(V_b) ** 2
            total_frobenius_loss = total_frobenius_loss + frobenius_loss
            
        # Average over batch
        rate_loss = total_rate_loss / batch_size
        precoding_loss = total_precoding_loss / batch_size
        frobenius_loss = total_frobenius_loss / batch_size
        
        # Total weighted loss
        total_loss = (self.rate_weight * rate_loss + 
                     self.precoding_weight * precoding_loss +
                     self.frobenius_weight * frobenius_loss)
        
        return {
            'total': total_loss,
            'rate': rate_loss,
            'precoding': precoding_loss,
            'frobenius': frobenius_loss
        }


class LLMRIMSATrainer:
    """
    Trainer for LLM-RIMSA model.
    
    Handles:
    - Data generation
    - Model training with hybrid loss
    - Evaluation and benchmarking
    - Checkpointing
    """
    
    def __init__(self,
                 llm_config: LLMConfig,
                 rimsa_config: RIMSAConfig,
                 channel_config: ChannelConfig,
                 training_config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            llm_config: LLM model configuration
            rimsa_config: RIMSA system configuration
            channel_config: Channel model configuration
            training_config: Training hyperparameters
        """
        self.llm_config = llm_config
        self.rimsa_config = rimsa_config
        self.channel_config = channel_config
        self.training_config = training_config
        self.device = training_config.device
        
        # Create models
        self.model = LLMRIMSAModel(llm_config).to(self.device)
        self.rimsa = RIMSASystem(rimsa_config).to(self.device)
        
        # Create data generator
        self.data_generator = ChannelGenerator(
            channel_config=channel_config,
            rimsa_config=rimsa_config,
            n_users=llm_config.n_users,
            device=self.device
        )
        
        # Create loss function
        self.loss_fn = HybridLoss(
            rimsa=self.rimsa,
            rate_weight=training_config.rate_weight,
            precoding_weight=training_config.precoding_weight,
            frobenius_weight=training_config.frobenius_weight,
            noise_power=training_config.noise_power
        )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.n_epochs,
            eta_min=training_config.learning_rate / 100
        )
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'rate_loss': [],
            'sum_rate': [],
            'learning_rate': []
        }
        
    def train_epoch(self, n_batches: int = 100) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            n_batches: Number of batches per epoch
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_rate_loss = 0.0
        total_sum_rate = 0.0
        
        for _ in range(n_batches):
            # Generate batch
            batch = self.data_generator.generate_batch(self.training_config.batch_size)
            channel = batch['channel']
            pilots = batch['pilot_signals']
            
            # Forward pass
            outputs = self.model(pilots)
            phase = outputs['phase']
            precoding = outputs['precoding']
            
            # Get beamforming matrix
            V = self.model.get_beamforming_matrix(phase)
            
            # Compute loss
            losses = self.loss_fn(phase, precoding, channel, V)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += losses['total'].item()
            total_rate_loss += losses['rate'].item()
            total_sum_rate -= losses['rate'].item()  # Negative of rate loss
            
        # Average metrics
        metrics = {
            'total_loss': total_loss / n_batches,
            'rate_loss': total_rate_loss / n_batches,
            'sum_rate': total_sum_rate / n_batches
        }
        
        return metrics
    
    def evaluate(self, n_batches: int = 20) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            n_batches: Number of evaluation batches
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_sum_rate = 0.0
        total_sinr = []
        
        with torch.no_grad():
            for _ in range(n_batches):
                batch = self.data_generator.generate_batch(self.training_config.batch_size)
                channel = batch['channel']
                pilots = batch['pilot_signals']
                
                # Forward pass
                outputs = self.model(pilots)
                phase = outputs['phase']
                precoding = outputs['precoding']
                
                # Get beamforming matrix
                V = self.model.get_beamforming_matrix(phase)
                
                # Compute sum rate for each sample
                for b in range(channel.shape[0]):
                    H = channel[b]
                    V_b = V[b]
                    W = precoding[b]
                    
                    sum_rate = self.rimsa.compute_sum_rate(
                        H, V_b, W, self.training_config.noise_power
                    )
                    total_sum_rate += sum_rate.item()
                    
                    sinr = self.rimsa.compute_sinr(
                        H, V_b, W, self.training_config.noise_power
                    )
                    total_sinr.extend(sinr.cpu().numpy())
                    
        n_samples = n_batches * self.training_config.batch_size
        
        return {
            'avg_sum_rate': total_sum_rate / n_samples,
            'avg_sinr_db': 10 * np.log10(np.mean(total_sinr)),
            'min_sinr_db': 10 * np.log10(np.min(total_sinr)),
            'max_sinr_db': 10 * np.log10(np.max(total_sinr))
        }
    
    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            verbose: Print progress
            
        Returns:
            Training history
        """
        start_time = time.time()
        
        if verbose:
            print("=" * 60)
            print("LLM-RIMSA Training")
            print("=" * 60)
            print(f"Device: {self.device}")
            print(f"Model parameters: {self.model.count_parameters()}")
            print(f"Epochs: {self.training_config.n_epochs}")
            print(f"Batch size: {self.training_config.batch_size}")
            print("=" * 60)
            
        best_sum_rate = -float('inf')
        
        for epoch in range(1, self.training_config.n_epochs + 1):
            # Train
            train_metrics = self.train_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(train_metrics['total_loss'])
            self.history['rate_loss'].append(train_metrics['rate_loss'])
            self.history['sum_rate'].append(train_metrics['sum_rate'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Evaluate periodically
            if epoch % self.training_config.eval_interval == 0 or epoch == 1:
                eval_metrics = self.evaluate()
                
                if eval_metrics['avg_sum_rate'] > best_sum_rate:
                    best_sum_rate = eval_metrics['avg_sum_rate']
                    
                if verbose:
                    print(f"Epoch {epoch:4d}: Loss={train_metrics['total_loss']:.4f}, "
                          f"Sum Rate={eval_metrics['avg_sum_rate']:.2f} bps/Hz, "
                          f"Avg SINR={eval_metrics['avg_sinr_db']:.1f} dB")
                          
            elif verbose and epoch % self.training_config.log_interval == 0:
                print(f"Epoch {epoch:4d}: Loss={train_metrics['total_loss']:.4f}, "
                      f"Train Sum Rate={train_metrics['sum_rate']:.2f} bps/Hz")
                      
        training_time = time.time() - start_time
        
        if verbose:
            print("=" * 60)
            print(f"Training complete in {training_time:.1f} seconds")
            print(f"Best sum rate: {best_sum_rate:.2f} bps/Hz")
            print("=" * 60)
            
        return self.history
    
    def benchmark_vs_baselines(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Compare LLM-RIMSA against baseline methods.
        
        Baselines:
        - Random phase: Random beamforming
        - ZF only: Zero-forcing precoding without optimization
        - LLM-RIMSA: Proposed method
        """
        self.model.eval()
        
        results = {
            'llm_rimsa': [],
            'random_phase': [],
            'zf_only': []
        }
        
        n_batches = (n_samples + self.training_config.batch_size - 1) // self.training_config.batch_size
        
        with torch.no_grad():
            for _ in tqdm(range(n_batches), desc="Benchmarking"):
                batch = self.data_generator.generate_batch(self.training_config.batch_size)
                channel = batch['channel']
                pilots = batch['pilot_signals']
                
                # LLM-RIMSA
                outputs = self.model(pilots)
                V_llm = self.model.get_beamforming_matrix(outputs['phase'])
                W_llm = outputs['precoding']
                
                for b in range(channel.shape[0]):
                    H = channel[b]
                    
                    # LLM-RIMSA performance
                    rate_llm = self.rimsa.compute_sum_rate(
                        H, V_llm[b], W_llm[b], self.training_config.noise_power
                    )
                    results['llm_rimsa'].append(rate_llm.item())
                    
                    # Random phase baseline
                    phase_random = torch.rand(
                        self.llm_config.n_rf_chains,
                        self.llm_config.n_elements_per_rimsa,
                        device=self.device
                    ) * 2 * np.pi
                    V_random = self.rimsa.compute_beamforming_matrix(phase_random)
                    W_random = self.rimsa.compute_zf_precoding(H, V_random)
                    rate_random = self.rimsa.compute_sum_rate(
                        H, V_random, W_random, self.training_config.noise_power
                    )
                    results['random_phase'].append(rate_random.item())
                    
                    # ZF-only baseline (fixed equal phase)
                    phase_equal = torch.zeros(
                        self.llm_config.n_rf_chains,
                        self.llm_config.n_elements_per_rimsa,
                        device=self.device
                    )
                    V_equal = self.rimsa.compute_beamforming_matrix(phase_equal)
                    W_zf = self.rimsa.compute_zf_precoding(H, V_equal)
                    rate_zf = self.rimsa.compute_sum_rate(
                        H, V_equal, W_zf, self.training_config.noise_power
                    )
                    results['zf_only'].append(rate_zf.item())
                    
        # Compute statistics
        summary = {}
        for method, rates in results.items():
            summary[f'{method}_mean'] = np.mean(rates)
            summary[f'{method}_std'] = np.std(rates)
            summary[f'{method}_max'] = np.max(rates)
            
        return summary
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'llm_config': self.llm_config,
            'rimsa_config': self.rimsa_config
        }, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


def run_llm_rimsa_training(n_epochs: int = 50) -> Tuple[LLMRIMSATrainer, Dict]:
    """
    Convenience function to run LLM-RIMSA training.
    
    Args:
        n_epochs: Number of training epochs
        
    Returns:
        (trainer, benchmark_results)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configurations
    llm_config = LLMConfig(
        hidden_dim=256,
        n_heads=8,
        n_layers=4,
        n_rf_chains=4,
        n_elements_per_rimsa=64,
        n_users=4,
        pilot_length=10,
        freeze_backbone=False,
        use_pretrained=False
    )
    
    rimsa_config = RIMSAConfig(
        n_elements_x=8,
        n_elements_y=8,
        n_rimsa_x=2,
        n_rimsa_y=2,
        carrier_frequency=28e9
    )
    
    channel_config = ChannelConfig(
        rician_k=10.0,
        path_loss_exponent=2.5,
        carrier_frequency=28e9
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        n_epochs=n_epochs,
        learning_rate=1e-4,
        rate_weight=1.0,
        precoding_weight=0.5,
        frobenius_weight=0.1,
        snr_db=20.0,
        log_interval=10,
        eval_interval=10,
        device=device
    )
    
    # Create trainer
    trainer = LLMRIMSATrainer(
        llm_config=llm_config,
        rimsa_config=rimsa_config,
        channel_config=channel_config,
        training_config=training_config
    )
    
    # Train
    trainer.train(verbose=True)
    
    # Benchmark
    print("\nBenchmarking against baselines...")
    benchmark = trainer.benchmark_vs_baselines(n_samples=200)
    
    print("\nBenchmark Results:")
    print(f"  LLM-RIMSA:    {benchmark['llm_rimsa_mean']:.2f} ± {benchmark['llm_rimsa_std']:.2f} bps/Hz")
    print(f"  Random Phase: {benchmark['random_phase_mean']:.2f} ± {benchmark['random_phase_std']:.2f} bps/Hz")
    print(f"  ZF Only:      {benchmark['zf_only_mean']:.2f} ± {benchmark['zf_only_std']:.2f} bps/Hz")
    
    return trainer, benchmark


if __name__ == "__main__":
    print("=" * 60)
    print("Paper 3: LLM-RIMSA for RIMSA Control")
    print("=" * 60)
    
    # Run training (reduced epochs for demo)
    trainer, benchmark = run_llm_rimsa_training(n_epochs=20)
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training curve
        axes[0].plot(trainer.history['epoch'], trainer.history['sum_rate'], 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Sum Rate (bps/Hz)')
        axes[0].set_title('Training Progress')
        axes[0].grid(True, alpha=0.3)
        
        # Benchmark comparison
        methods = ['LLM-RIMSA', 'Random Phase', 'ZF Only']
        means = [benchmark['llm_rimsa_mean'], benchmark['random_phase_mean'], benchmark['zf_only_mean']]
        stds = [benchmark['llm_rimsa_std'], benchmark['random_phase_std'], benchmark['zf_only_std']]
        
        colors = ['green', 'gray', 'orange']
        bars = axes[1].bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        axes[1].set_ylabel('Sum Rate (bps/Hz)')
        axes[1].set_title('Performance Comparison')
        axes[1].grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('llm_rimsa_results.png', dpi=150)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available")
        
    print("\nLLM-RIMSA Training Complete!")
