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
        embed_dim: Optional[int] = None,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            feature_dims=feature_dims,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm
        )

        self.embed_dim = embed_dim or feature_dims[0]

        # Shared multi-head attention module used by all layers
        self.shared_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.input_projections = nn.ModuleList([
            nn.Linear(dim, self.embed_dim) if dim != self.embed_dim else nn.Identity()
            for dim in feature_dims
        ])
        self.output_projections = nn.ModuleList([
            nn.Linear(self.embed_dim, dim) for dim in feature_dims
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
        tokens = []
        for i, feat in enumerate(layer_outputs):
            if feat.dim() == 4:
                pooled = feat.mean(dim=(2, 3))
            elif feat.dim() == 2:
                pooled = feat
            else:
                raise ValueError(f"Unsupported tensor dimension: {feat.dim()}. Expected 2D or 4D.")
            tokens.append(self.input_projections[i](pooled))

        stacked_tokens = torch.stack(tokens, dim=1)  # (batch, num_layers, embed_dim)

        _, attn_weights = self.shared_attention(
            query=stacked_tokens,
            key=stacked_tokens,
            value=stacked_tokens,
            need_weights=True,
            average_attn_weights=False
        )

        adjacency_gate = torch.sigmoid(self.adjacency)
        gate_for_attention = adjacency_gate.transpose(0, 1)

        gated_weights = attn_weights * gate_for_attention.unsqueeze(0).unsqueeze(0)
        gated_weights = gated_weights / gated_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        gated_output = torch.einsum('bhij,bje->bhie', gated_weights, stacked_tokens)
        gated_output = gated_output.mean(dim=1)

        refined_outputs = []
        for i, feat in enumerate(layer_outputs):
            update_token = self.output_projections[i](gated_output[:, i])
            if feat.dim() == 4:
                update = update_token[:, :, None, None].expand_as(feat)
            else:
                update = update_token

            refined = self.apply_residual_connection(
                original=layer_outputs[i],
                update=update,
                layer_idx=i
            )
            refined_outputs.append(refined)

        debug_info = {
            'attention_weights': attn_weights,
            'gated_attention_weights': gated_weights,
            'adjacency_matrix': self.adjacency,
            'adjacency_gate': adjacency_gate,
            'communication_type': 'dynamic'
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
