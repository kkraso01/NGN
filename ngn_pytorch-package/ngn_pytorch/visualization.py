"""
NGN Utils - Visualization Tools

This module provides visualization tools for NGN models,
including attention heatmaps, adjacency matrices, and
gradient flow analysis.
"""

from typing import List, Dict, Optional, Tuple, Any
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NGNVisualizer:
    """
    Visualization tools for NGN models.

    Provides methods to visualize:
    - Attention weight heatmaps
    - Learned adjacency matrices
    - Gradient flow
    - Layer communication patterns
    """

    def __init__(self, output_dir: str = 'visualizations/'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up matplotlib
        plt.style.use('default')
        sns.set_palette("husl")

    def plot_layer_graph_diagram(self, adjacency_matrix: torch.Tensor, epoch: int = 0, save: bool = True) -> str:
        """
        Plot a diagram of the learned layer graph (nodes and edges).
        Placeholder for future implementation (e.g., using networkx).
        """
        logger.info("plot_layer_graph_diagram is a placeholder (not implemented)")
        return ""

    def plot_attention_heads_analysis(self, attention_weights: list, epoch: int = 0, save: bool = True) -> str:
        """
        Plot advanced analysis of attention heads (e.g., per-head entropy, diversity).
        Placeholder for future implementation.
        """
        logger.info("plot_attention_heads_analysis is a placeholder (not implemented)")
        return ""

    def plot_interpretability_summary(self, debug_info: dict, epoch: int = 0, save: bool = True) -> str:
        """
        Plot a summary figure combining attention, adjacency, and communication type for interpretability.
        Placeholder for future implementation.
        """
        logger.info("plot_interpretability_summary is a placeholder (not implemented)")
        return ""

    def plot_attention_heatmap(
        self,
        attention_weights: List[Tensor] | Tensor,
        epoch: int,
        batch_idx: Optional[int] = None,
        save: bool = True
    ) -> str:
        """
        Plot attention weight heatmaps for each layer.

        Args:
            attention_weights: List of attention weight tensors
            epoch: Current training epoch
            batch_idx: Batch index (optional)
            save: Whether to save the plot

        Returns:
            filename: Path to saved plot
        """
        if attention_weights is None:
            logger.warning("No attention weights to plot")
            return ""

        if isinstance(attention_weights, Tensor):
            attention_weights = [attention_weights]

        if len(attention_weights) == 0:
            logger.warning("No attention weights to plot")
            return ""

        num_layers = len(attention_weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))

        if num_layers == 1:
            axes = [axes]

        for i, attn in enumerate(attention_weights):
            attn_np = attn.detach().cpu().numpy()
            if attn_np.ndim == 4:  # (batch, heads, seq_len, seq_len)
                attn_np = attn_np.mean(axis=(0, 1))
            elif attn_np.ndim == 3:
                attn_np = attn_np.mean(axis=0)

            # Plot heatmap
            sns.heatmap(
                attn_np,
                ax=axes[i],
                cmap='viridis',
                square=True,
                cbar=True,
                vmin=0,
                vmax=1
            )

            axes[i].set_title(f'Layer {i} Attention')
            axes[i].set_xlabel('Key Layer')
            axes[i].set_ylabel('Query Position')

        plt.tight_layout()

        filename = ""
        if save:
            batch_str = f"_batch_{batch_idx}" if batch_idx is not None else ""
            filename = f'attention_epoch_{epoch}{batch_str}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {filepath}")

        plt.close()
        return str(filename)

    def plot_adjacency_matrix(
        self,
        adjacency_matrix: Tensor,
        epoch: int,
        threshold: float = 0.1,
        save: bool = True
    ) -> str:
        """
        Plot the learned adjacency matrix.

        Args:
            adjacency_matrix: (num_layers, num_layers) adjacency matrix
            epoch: Current training epoch
            threshold: Threshold for highlighting strong connections
            save: Whether to save the plot

        Returns:
            filename: Path to saved plot
        """
        adj_np = adjacency_matrix.detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Raw adjacency matrix
        sns.heatmap(
            adj_np,
            ax=ax1,
            cmap='RdYlBu_r',
            square=True,
            cbar=True,
            annot=True,
            fmt='.2f'
        )
        ax1.set_title(f'Learned Adjacency Matrix (Epoch {epoch})')
        ax1.set_xlabel('To Layer')
        ax1.set_ylabel('From Layer')

        # Thresholded adjacency (sparsity)
        thresholded = (adj_np > threshold).astype(float)
        sns.heatmap(
            thresholded,
            ax=ax2,
            cmap='Greys',
            square=True,
            cbar=False,
            annot=True,
            fmt='.0f'
        )
        ax2.set_title(f'Sparse Adjacency (Threshold={threshold})')
        ax2.set_xlabel('To Layer')
        ax2.set_ylabel('From Layer')

        plt.tight_layout()

        filename = ""
        if save:
            filename = f'adjacency_epoch_{epoch}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved adjacency matrix to {filepath}")

        plt.close()
        return str(filename)

    def plot_gradient_flow(
        self,
        named_parameters: Dict[str, Tensor],
        epoch: int,
        save: bool = True
    ) -> str:
        """
        Plot gradient flow through the network.

        Args:
            named_parameters: Model parameters with gradients
            epoch: Current training epoch
            save: Whether to save the plot

        Returns:
            filename: Path to saved plot
        """
        grad_norms = {}
        layer_names = []

        for name, param in named_parameters.items():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                # Group by layer
                layer_name = name.split('.')[0]
                if layer_name not in grad_norms:
                    grad_norms[layer_name] = []
                    layer_names.append(layer_name)
                grad_norms[layer_name].append(grad_norm)

        # Average per layer
        avg_grad_norms = [np.mean(grad_norms[layer]) for layer in layer_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(layer_names, avg_grad_norms, color='skyblue')

        ax.set_yscale('log')
        ax.set_title(f'Gradient Flow (Epoch {epoch})')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Gradient Norm (log scale)')

        # Add value labels
        for bar, norm in zip(bars, avg_grad_norms):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{norm:.2e}',
                ha='center',
                va='bottom',
                rotation=45
            )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        filename = ""
        if save:
            filename = f'gradient_flow_epoch_{epoch}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved gradient flow plot to {filepath}")

        plt.close()
        return str(filename)

    def plot_layer_communication_patterns(
        self,
        debug_info: Dict[str, Any],
        epoch: int,
        save: bool = True
    ) -> str:
        """
        Plot comprehensive layer communication analysis.

        Args:
            debug_info: Debug information from model forward pass
            epoch: Current training epoch
            save: Whether to save the plot

        Returns:
            filename: Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Attention entropy over layers
        if 'attention_weights' in debug_info:
            entropies = []
            for attn in debug_info['attention_weights']:
                if isinstance(attn, list):
                    attn = attn[0]
                entropy = -torch.sum(
                    attn * torch.log(attn + 1e-8),
                    dim=-1
                ).mean().item()
                entropies.append(entropy)

            axes[0, 0].bar(range(len(entropies)), entropies, color='lightcoral')
            axes[0, 0].set_title('Attention Entropy by Layer')
            axes[0, 0].set_xlabel('Layer')
            axes[0, 0].set_ylabel('Entropy')

        # 2. Adjacency matrix strength distribution
        if 'adjacency_matrix' in debug_info:
            adj = debug_info['adjacency_matrix'].flatten().detach().cpu().numpy()
            axes[0, 1].hist(adj, bins=20, alpha=0.7, color='lightblue')
            axes[0, 1].set_title('Adjacency Weight Distribution')
            axes[0, 1].set_xlabel('Connection Strength')
            axes[0, 1].set_ylabel('Frequency')

        # 3. Sparsity mask
        if 'sparsity_mask' in debug_info:
            sparsity = debug_info['sparsity_mask'].detach().cpu().numpy()
            sns.heatmap(
                sparsity,
                ax=axes[1, 0],
                cmap='Greys',
                square=True,
                cbar=False
            )
            axes[1, 0].set_title('Sparsity Mask')
            axes[1, 0].set_xlabel('To Layer')
            axes[1, 0].set_ylabel('From Layer')

        # 4. Communication type indicator
        comm_type = debug_info.get('communication_type', 'attention')
        axes[1, 1].text(
            0.5, 0.5,
            f'Communication Type:\n{comm_type}',
            ha='center',
            va='center',
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen")
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.suptitle(f'Layer Communication Analysis (Epoch {epoch})', fontsize=16)
        plt.tight_layout()

        filename = ""
        if save:
            filename = f'communication_analysis_epoch_{epoch}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved communication analysis to {filepath}")

        plt.close()
        return str(filename)

    def create_training_summary_video(
        self,
        epochs: List[int],
        attention_files: List[str],
        adjacency_files: List[str],
        output_filename: str = 'training_evolution.mp4'
    ):
        """
        Create a video showing training evolution.

        Note: Requires ffmpeg and imageio. This is a placeholder for future implementation.
        """
        logger.info("Video creation not implemented yet")
        # TODO: Implement video creation from saved plots
        pass
