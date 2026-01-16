"""
NGN Communication Module - Shared Attention Aggregator

This module implements the SharedAttentionAggregator class that uses
multi-head attention for cross-layer communication in NGN.
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch import nn, Tensor

from .graph import LayerGraph


class SharedAttentionAggregator(LayerGraph):
    """
    Shared multi-head attention for cross-layer communication.

    This class implements the core NGN innovation: a shared attention
    mechanism that allows each layer to attend to all other layers,
    learning which inter-layer connections are most beneficial.

    Args:
        num_layers: Number of layers in the backbone
        feature_dims: Feature dimensions for each layer
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability for attention (default: 0.1)
        use_residual: Whether to use residual connections (default: True)
        use_layer_norm: Whether to apply layer normalization (default: True)
    """

    def __init__(
        self,
        num_layers: int,
        feature_dims: List[int],
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            feature_dims=feature_dims,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm
        )

        # Assume all layers have the same feature dimension for simplicity
        # In practice, we might need projection layers for different dims
        embed_dim = feature_dims[0]  # Use first layer's dimension as reference

        # Shared multi-head attention module used by all layers
        self.shared_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Optional projection layers if feature dimensions differ
        self.projections = nn.ModuleList()
        for dim in feature_dims:
            if dim != embed_dim:
                self.projections.append(nn.Linear(dim, embed_dim))
            else:
                self.projections.append(nn.Identity())

        # Update layer norms for the shared embed_dim
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in feature_dims
            ])

    def forward(
        self,
        layer_outputs: List[Tensor]
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Apply shared attention across all layers.

        Args:
            layer_outputs: List of feature tensors from each layer
                          Shape: [f_1, f_2, ..., f_L]

        Returns:
            refined_outputs: List of refined feature tensors
            debug_info: Dictionary containing attention weights and adjacency
        """
        assert len(layer_outputs) == self.num_layers, \
            f"Expected {self.num_layers} layer outputs, got {len(layer_outputs)}"

        # Project all layer outputs to common dimension if needed
        projected_outputs = []
        for i, feat in enumerate(layer_outputs):
            proj_feat = self.projections[i](feat.flatten(2).transpose(1, 2))
            projected_outputs.append(proj_feat)

        # Stack all layer features for attention
        # Shape: (batch_size, num_layers, embed_dim)
        stacked_features = torch.stack(projected_outputs, dim=1)

        refined_outputs = []
        attention_weights_list = []

        # For each layer, attend to all other layers
        for i in range(self.num_layers):
            # Query: current layer features
            query = projected_outputs[i].unsqueeze(1)  # (batch, 1, embed_dim)

            # Keys and values: all layer features
            keys_values = stacked_features  # (batch, num_layers, embed_dim)

            # Apply attention
            attn_output, attn_weights = self.shared_attention(
                query=query,
                key=keys_values,
                value=keys_values
            )

            # Remove the singleton dimension
            attn_output = attn_output.squeeze(1)  # (batch, embed_dim)

            # Reshape back to spatial dimensions (assuming we can infer)
            # This is a simplification; in practice, we need to handle spatial dims
            batch_size, seq_len, embed_dim = attn_output.shape
            # Assume square spatial dimension for simplicity
            spatial_size = int((seq_len * embed_dim / layer_outputs[i].shape[1]) ** 0.5)
            height = width = spatial_size

            try:
                attn_output_spatial = attn_output.view(
                    batch_size, layer_outputs[i].shape[1], height, width
                )
            except RuntimeError:
                # Fallback: use original spatial dimensions
                original_h, original_w = layer_outputs[i].shape[2], layer_outputs[i].shape[3]
                attn_output_spatial = attn_output.view(
                    batch_size, layer_outputs[i].shape[1], original_h, original_w
                )

            # Apply residual connection and layer norm
            refined = self.apply_residual_connection(
                original=layer_outputs[i],
                update=attn_output_spatial,
                layer_idx=i
            )

            refined_outputs.append(refined)
            attention_weights_list.append(attn_weights)

        # Prepare debug information
        debug_info = {
            'attention_weights': attention_weights_list,
            'adjacency_matrix': self.get_adjacency_matrix(),
            'sparsity_mask': self.get_sparsity_mask()
        }

        return refined_outputs, debug_info


class SharedRNNAggregator(LayerGraph):
    """
    Alternative: Shared RNN for cross-layer communication.

    Similar to DIANet (AAAI 2020), this uses a shared LSTM/GRU
    to aggregate information across layers sequentially.
    """

    def __init__(
        self,
        num_layers: int,
        feature_dims: List[int],
        hidden_dim: int = 256,
        rnn_type: str = 'lstm',
        use_residual: bool = True,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            feature_dims=feature_dims,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm
        )

        embed_dim = feature_dims[0]

        # Shared RNN cell
        if rnn_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTMCell(embed_dim, hidden_dim)
        elif rnn_type.lower() == 'gru':
            self.rnn_cell = nn.GRUCell(embed_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Projection to shared dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity()
            for dim in feature_dims
        ])

        # Output projection back to layer dimensions
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in feature_dims
        ])

    def forward(
        self,
        layer_outputs: List[Tensor]
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Apply shared RNN across layers sequentially.
        """
        refined_outputs = []
        hidden_state = None
        cell_state = None if isinstance(self.rnn_cell, nn.LSTMCell) else None

        for i, feat in enumerate(layer_outputs):
            # Project to shared dimension
            proj_feat = self.projections[i](
                feat.mean(dim=[2, 3])  # Global average pool
            )

            # Update RNN state
            if isinstance(self.rnn_cell, nn.LSTMCell):
                hidden_state, cell_state = self.rnn_cell(proj_feat, (hidden_state, cell_state))
            else:
                hidden_state = self.rnn_cell(proj_feat, hidden_state)

            # Project back to layer dimension and reshape
            update = self.output_projections[i](hidden_state)
            update_spatial = update.unsqueeze(-1).unsqueeze(-1).expand_as(feat)

            # Apply residual connection
            refined = self.apply_residual_connection(feat, update_spatial, i)
            refined_outputs.append(refined)

        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'final_hidden': hidden_state.detach() if hidden_state is not None else None
        }

        return refined_outputs, debug_info