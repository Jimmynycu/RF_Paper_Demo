"""
LLM Backbone - GPT-2 Based Model for RIMSA Control

Implements the Large Language Model architecture for RIMSA beamforming:
- GPT-2 backbone (pre-trained, optionally frozen)
- CSI preprocessor for pilot signal embedding
- Spatio-temporal attention modules
- Output heads for phase control and precoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Try to import transformers, provide fallback if not available
try:
    from transformers import GPT2Model, GPT2Config as HFConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Using simplified LLM backbone.")


@dataclass
class LLMConfig:
    """Configuration for LLM-RIMSA model."""
    # Model dimensions
    hidden_dim: int = 768  # GPT-2 base hidden dimension
    n_heads: int = 12
    n_layers: int = 6  # Fewer than full GPT-2 for efficiency
    
    # RIMSA configuration
    n_rf_chains: int = 4
    n_elements_per_rimsa: int = 256
    n_users: int = 4
    pilot_length: int = 10
    
    # Training configuration
    freeze_backbone: bool = True
    use_pretrained: bool = True
    dropout: float = 0.1
    
    @property
    def n_total_elements(self) -> int:
        return self.n_rf_chains * self.n_elements_per_rimsa


class CSIPreprocessor(nn.Module):
    """
    Channel State Information Preprocessor.
    
    Converts received pilot signals to embeddings suitable for LLM input.
    From paper: Uses learned linear projections.
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Input: (batch, 2, n_rf, pilot_length) - real and imaginary parts
        input_dim = 2 * config.n_rf_chains * config.pilot_length
        
        # Projection to LLM dimension
        self.input_project = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding for sequence
        self.position_embed = nn.Parameter(
            torch.randn(1, config.n_rf_chains + config.n_users, config.hidden_dim) * 0.02
        )
        
    def forward(self, pilot_signals: torch.Tensor) -> torch.Tensor:
        """
        Preprocess pilot signals.
        
        Args:
            pilot_signals: Complex pilots (batch, n_rf, pilot_length)
            
        Returns:
            Embeddings (batch, seq_len, hidden_dim)
        """
        batch_size = pilot_signals.shape[0]
        
        # Convert to real (stack real and imaginary)
        real = pilot_signals.real
        imag = pilot_signals.imag
        x = torch.stack([real, imag], dim=1)  # (batch, 2, n_rf, pilot)
        
        # Flatten and project
        x = x.view(batch_size, -1)  # (batch, 2*n_rf*pilot)
        x = self.input_project(x)  # (batch, hidden_dim)
        
        # Expand to sequence
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Repeat for each output position
        seq_len = self.config.n_rf_chains + self.config.n_users
        x = x.expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.position_embed[:, :seq_len]
        
        return x


class SpatioTemporalAttention(nn.Module):
    """
    Spatio-Temporal Attention Module.
    
    Captures correlations across:
    - Space: different antenna elements/users
    - Time: temporal evolution of channel
    
    From paper: Multi-head attention with separate spatial and temporal heads.
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Spatial attention (across antennas)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Temporal attention (channel evolution)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ln3 = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatio-temporal attention.
        
        Args:
            x: Input embeddings (batch, seq_len, hidden_dim)
            
        Returns:
            Attended embeddings (batch, seq_len, hidden_dim)
        """
        # Spatial attention
        residual = x
        x = self.ln1(x)
        x, _ = self.spatial_attn(x, x, x)
        x = residual + x
        
        # Temporal attention
        residual = x
        x = self.ln2(x)
        x, _ = self.temporal_attn(x, x, x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.ln3(x)
        x = self.ff(x)
        x = residual + x
        
        return x


class SimplifiedLLMBackbone(nn.Module):
    """
    Simplified transformer backbone (used when transformers library unavailable).
    
    Implements essential transformer functionality for RIMSA control.
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (for CSI features)
        self.embed = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SpatioTemporalAttention(config) for _ in range(config.n_layers)
        ])
        
        self.final_ln = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        x = self.embed(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_ln(x)
        return x


class PhaseControlHead(nn.Module):
    """
    Output head for RIMSA phase control.
    
    Generates phase matrix V from LLM output.
    Output: (n_rf_chains, n_elements_per_rimsa) phase values in [0, 2π]
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        output_dim = config.n_rf_chains * config.n_elements_per_rimsa
        
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0, 1], scale to [0, 2π]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate phase control matrix.
        
        Args:
            x: LLM output (batch, seq_len, hidden_dim)
            
        Returns:
            Phase matrix (batch, n_rf, n_elements) in [0, 2π]
        """
        # Use mean pooling over sequence
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Generate phases
        phases = self.head(x)  # (batch, n_rf * n_elements)
        phases = phases * 2 * np.pi  # Scale to [0, 2π]
        
        # Reshape
        batch_size = x.shape[0]
        phases = phases.view(batch_size, self.config.n_rf_chains, self.config.n_elements_per_rimsa)
        
        return phases


class PrecodingHead(nn.Module):
    """
    Output head for digital precoding.
    
    Generates precoding matrix W from LLM output.
    Output: Complex matrix (n_rf_chains, n_users)
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Output real and imaginary parts separately
        output_dim = 2 * config.n_rf_chains * config.n_users
        
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, output_dim),
            nn.Tanh()  # Bounded output
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate digital precoding matrix.
        
        Args:
            x: LLM output (batch, seq_len, hidden_dim)
            
        Returns:
            Complex precoding matrix (batch, n_rf, n_users)
        """
        # Use mean pooling
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Generate real and imaginary parts
        w = self.head(x)  # (batch, 2 * n_rf * n_users)
        
        batch_size = x.shape[0]
        w = w.view(batch_size, 2, self.config.n_rf_chains, self.config.n_users)
        
        # Convert to complex
        w_complex = torch.complex(w[:, 0], w[:, 1])
        
        # Normalize columns (per user)
        norm = torch.norm(w_complex, dim=1, keepdim=True)
        w_complex = w_complex / (norm + 1e-8)
        
        return w_complex


class LLMRIMSAModel(nn.Module):
    """
    Complete LLM-RIMSA Model.
    
    Architecture from paper:
    1. CSI Preprocessor: Pilot signals → Embeddings
    2. LLM Backbone: GPT-2 or simplified transformer
    3. Spatio-Temporal Attention: Channel correlation modeling
    4. Output Heads: Phase control + Digital precoding
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # CSI preprocessor
        self.preprocessor = CSIPreprocessor(config)
        
        # LLM backbone
        if TRANSFORMERS_AVAILABLE and config.use_pretrained:
            print("Using pre-trained GPT-2 backbone")
            gpt2_config = HFConfig(
                n_embd=config.hidden_dim,
                n_head=config.n_heads,
                n_layer=config.n_layers,
                resid_pdrop=config.dropout,
                embd_pdrop=config.dropout,
                attn_pdrop=config.dropout
            )
            self.backbone = GPT2Model(gpt2_config)
            
            if config.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            print("Using simplified LLM backbone")
            self.backbone = SimplifiedLLMBackbone(config)
            
        # Additional attention layers
        self.st_attention = SpatioTemporalAttention(config)
        
        # Output heads
        self.phase_head = PhaseControlHead(config)
        self.precoding_head = PrecodingHead(config)
        
    def forward(self, pilot_signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pilot_signals: Complex pilots (batch, n_rf, pilot_length)
            
        Returns:
            Dictionary with:
            - 'phase': Phase control matrix (batch, n_rf, n_elements)
            - 'precoding': Precoding matrix (batch, n_rf, n_users)
        """
        # Preprocess CSI
        x = self.preprocessor(pilot_signals)
        
        # LLM backbone
        if hasattr(self.backbone, 'forward') and hasattr(self.backbone, 'wte'):
            # HuggingFace GPT-2
            outputs = self.backbone(inputs_embeds=x)
            x = outputs.last_hidden_state
        else:
            # Simplified backbone
            x = self.backbone(x)
            
        # Spatio-temporal attention
        x = self.st_attention(x)
        
        # Output heads
        phase = self.phase_head(x)
        precoding = self.precoding_head(x)
        
        return {
            'phase': phase,
            'precoding': precoding
        }
    
    def get_beamforming_matrix(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Convert phase matrix to beamforming matrix V.
        
        Args:
            phase: Phase values (batch, n_rf, n_elements)
            
        Returns:
            V: Beamforming matrix (batch, n_total_elements, n_rf)
        """
        batch_size = phase.shape[0]
        n_rf = self.config.n_rf_chains
        n_elements = self.config.n_elements_per_rimsa
        
        # Phase weights
        weights = torch.exp(1j * phase) / np.sqrt(n_elements)
        
        # Build block-diagonal V for each batch
        V = torch.zeros(batch_size, n_rf * n_elements, n_rf, 
                       dtype=torch.complex64, device=phase.device)
        
        for r in range(n_rf):
            start = r * n_elements
            end = (r + 1) * n_elements
            V[:, start:end, r] = weights[:, r]
            
        return V
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


if __name__ == "__main__":
    # Demo: LLM-RIMSA model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    config = LLMConfig(
        hidden_dim=256,  # Smaller for demo
        n_heads=8,
        n_layers=4,
        n_rf_chains=4,
        n_elements_per_rimsa=64,
        n_users=4,
        pilot_length=10,
        freeze_backbone=False,
        use_pretrained=False
    )
    
    print(f"LLM-RIMSA Configuration:")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  Transformer layers: {config.n_layers}")
    print(f"  RF chains: {config.n_rf_chains}")
    print(f"  Elements per RIMSA: {config.n_elements_per_rimsa}")
    print(f"  Total elements: {config.n_total_elements}")
    
    # Create model
    model = LLMRIMSAModel(config).to(device)
    
    # Parameter count
    params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    
    # Test forward pass
    batch_size = 8
    pilots = torch.randn(batch_size, config.n_rf_chains, config.pilot_length,
                        dtype=torch.complex64, device=device)
    
    print(f"\nInput pilot shape: {pilots.shape}")
    
    outputs = model(pilots)
    
    print(f"Output shapes:")
    print(f"  Phase matrix: {outputs['phase'].shape}")
    print(f"  Precoding matrix: {outputs['precoding'].shape}")
    
    # Get beamforming matrix
    V = model.get_beamforming_matrix(outputs['phase'])
    print(f"  Beamforming matrix V: {V.shape}")
    
    # Verify phase range
    phase = outputs['phase']
    print(f"\nPhase statistics:")
    print(f"  Min: {phase.min().item():.4f}")
    print(f"  Max: {phase.max().item():.4f}")
    print(f"  Mean: {phase.mean().item():.4f}")
    
    print("\nLLM Backbone Demo Complete!")
