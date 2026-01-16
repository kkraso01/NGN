"""
NGN Core Components - LayerGraph Base Class

This module defines the base LayerGraph class that learns which layers
should communicate with each other in a neural network.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import torch
from torch import nn, Tensor


class LayerGraph(nn.Module, ABC):
    """
    Base class for learned layer connectivity graphs.

    This abstract class defines the interface for learning which layers
    in a neural network should communicate with each other. Subclasses
    implement different communication mechanisms (attention, RNN, etc.).

    Args:
        num_layers: Number of layers in the backbone network
        feature_dims: List of feature dimensions for each layer
        use_residual: Whether to use residual connections for stability
        use_layer_norm: Whether to apply layer normalization
    """

    def __init__(
        self,
        num_layers: int,
        feature_dims: List[int],
        use_residual: bool = True,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.feature_dims = feature_dims
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Learnable adjacency matrix: which layers connect to which
        # Shape: (num_layers, num_layers)
        self.adjacency = nn.Parameter(
            torch.ones(num_layers, num_layers) / num_layers
        )

        # Layer normalization for stability
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in feature_dims
            ])

    @abstractmethod
    def forward(
        self,
        layer_outputs: List[Tensor]
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Apply learned layer graph communication.

        Args:
            layer_outputs: List of feature tensors from each layer
                          Shape: [f_1, f_2, ..., f_L] where f_i has shape
                          (batch_size, channels_i, height_i, width_i)

        Returns:
            refined_outputs: List of refined feature tensors after cross-layer communication
            debug_info: Dictionary with debugging information (attention weights, etc.)
        """
        pass

    def get_adjacency_matrix(self) -> Tensor:
        """Return the current learned adjacency matrix."""
        return self.adjacency.detach()

    def get_sparsity_mask(self, threshold: float = 0.1) -> Tensor:
        """
        Get binary mask for sparse connections based on adjacency matrix.

        Args:
            threshold: Minimum connection strength to keep

        Returns:
            Binary mask of shape (num_layers, num_layers)
        """
        return (self.adjacency > threshold).float()

    def apply_residual_connection(
        self,
        original: Tensor,
        update: Tensor,
        layer_idx: int
    ) -> Tensor:
        """
        Apply residual connection with optional layer normalization.

        Args:
            original: Original layer features
            update: Cross-layer update to add
            layer_idx: Index of the layer for normalization

        Returns:
            Refined features with residual connection
        """
        if self.use_residual:
            refined = original + update
        else:
            refined = update

        if self.use_layer_norm:
            # Handle both 2D (batch_size, feature_dim) and 4D (batch_size, channels, height, width) tensors
            if refined.dim() == 2:
                # 2D case: (batch_size, feature_dim)
                batch_size, feature_dim = refined.shape
                refined_flat = refined.view(batch_size, feature_dim, 1).transpose(1, 2)
                refined_flat = self.layer_norms[layer_idx](refined_flat)
                refined = refined_flat.transpose(1, 2).view(batch_size, feature_dim)
            elif refined.dim() == 4:
                # 4D case: (batch_size, channels, height, width)
                batch_size, channels, height, width = refined.shape
                refined_flat = refined.view(batch_size, channels, -1).transpose(1, 2)
                refined_flat = self.layer_norms[layer_idx](refined_flat)
                refined = refined_flat.transpose(1, 2).view(batch_size, channels, height, width)
            else:
                raise ValueError(f"Unsupported tensor dimension: {refined.dim()}. Expected 2D or 4D.")

        return refined