"""
NGN Communication Module - Shared Attention Aggregator

This module implements the SharedAttentionAggregator class that uses
multi-head attention for cross-layer communication in NGN.
"""


from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    ):
        super().__init__(
            num_layers=num_layers,
            feature_dims=feature_dims,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm
        )

        embed_dim = feature_dims[0]
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Shared multi-head attention (Keras)
        self.shared_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout,
            name="shared_attention"
        )

        # Optional projection layers if feature dimensions differ
        self.projections = []
        for dim in feature_dims:
            if dim != embed_dim:
                self.projections.append(layers.Dense(embed_dim))
            else:
                self.projections.append(layers.Lambda(lambda x: x))

        # Output projections back to layer dimensions
        self.output_projections = [layers.Dense(dim) for dim in feature_dims]

        # Update layer norms for the shared embed_dim
        if self.use_layer_norm:
            self.layer_norms = [layers.LayerNormalization(axis=-1) for _ in feature_dims]
        else:
            self.layer_norms = None

    def call(
        self,
        layer_outputs: List[tf.Tensor]
    ) -> Tuple[List[tf.Tensor], Dict[str, tf.Tensor]]:
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
            # Flatten spatial dims if needed (B, C, H, W) -> (B, H*W, C)
            if len(feat.shape) == 4:
                batch_size, channels, height, width = tf.unstack(tf.shape(feat))
                feat_flat = tf.transpose(feat, [0, 2, 3, 1])  # (B, H, W, C)
                feat_flat = tf.reshape(feat_flat, [batch_size, height * width, channels])
            else:
                feat_flat = feat
            proj_feat = self.projections[i](feat_flat)
            projected_outputs.append(proj_feat)

        # Stack all layer features for attention: (B, num_layers, seq, C) -> (B, num_layers, C)
        # For simplicity, use the mean over sequence dimension if present
        pooled_outputs = [tf.reduce_mean(p, axis=1) if len(p.shape) == 3 else p for p in projected_outputs]
        stacked_features = tf.stack(pooled_outputs, axis=1)  # (B, num_layers, C)

        refined_outputs = []
        attention_weights_list = []

        for i in range(self.num_layers):
            # Query: current layer features (B, 1, C)
            query = tf.expand_dims(pooled_outputs[i], axis=1)
            # Keys/values: all layers (B, num_layers, C)
            attn_output, attn_weights = self.shared_attention(
                query=query,
                value=stacked_features,
                key=stacked_features,
                return_attention_scores=True
            )
            attn_output = tf.squeeze(attn_output, axis=1)  # (B, C)
            attn_output = self.output_projections[i](attn_output)  # (B, feature_dims[i])

            # Reshape back to original spatial dims if needed
            if len(layer_outputs[i].shape) == 4:
                batch_size, channels, height, width = tf.unstack(tf.shape(layer_outputs[i]))
                attn_output_spatial = tf.reshape(attn_output, [batch_size, channels, 1, 1])
                attn_output_spatial = tf.tile(attn_output_spatial, [1, 1, height, width])
            else:
                attn_output_spatial = attn_output

            refined = self.apply_residual_connection(
                original=layer_outputs[i],
                update=attn_output_spatial,
                layer_idx=i
            )
            refined_outputs.append(refined)
            attention_weights_list.append(attn_weights)

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
    ):
        super().__init__(
            num_layers=num_layers,
            feature_dims=feature_dims,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm
        )
        embed_dim = feature_dims[0]
        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        # Shared RNN cell
        if self.rnn_type == 'lstm':
            self.rnn_cell = layers.LSTMCell(hidden_dim)
        elif self.rnn_type == 'gru':
            self.rnn_cell = layers.GRUCell(hidden_dim)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        # Projection to shared dimension
        self.projections = [layers.Dense(embed_dim) if dim != embed_dim else layers.Lambda(lambda x: x) for dim in feature_dims]
        # Output projection back to layer dimensions
        self.output_projections = [layers.Dense(dim) for dim in feature_dims]

    def call(
        self,
        layer_outputs: List[tf.Tensor]
    ) -> Tuple[List[tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Apply shared RNN across layers sequentially.
        """
        refined_outputs = []
        hidden_state = None
        cell_state = None
        for i, feat in enumerate(layer_outputs):
            # Global average pool if 4D
            if len(feat.shape) == 4:
                proj_input = tf.reduce_mean(feat, axis=[2, 3])
            else:
                proj_input = feat
            proj_feat = self.projections[i](proj_input)
            # Update RNN state
            if self.rnn_type == 'lstm':
                if hidden_state is None:
                    hidden_state = [tf.zeros((tf.shape(proj_feat)[0], self.hidden_dim)),
                                    tf.zeros((tf.shape(proj_feat)[0], self.hidden_dim))]
                hidden_state = self.rnn_cell(proj_feat, hidden_state)
                h = hidden_state[0]
            else:
                if hidden_state is None:
                    hidden_state = tf.zeros((tf.shape(proj_feat)[0], self.hidden_dim))
                hidden_state = self.rnn_cell(proj_feat, [hidden_state])[0]
                h = hidden_state
            # Project back to layer dimension and reshape
            update = self.output_projections[i](h)
            if len(feat.shape) == 4:
                batch_size, channels, height, width = tf.unstack(tf.shape(feat))
                update_spatial = tf.reshape(update, [batch_size, channels, 1, 1])
                update_spatial = tf.tile(update_spatial, [1, 1, height, width])
            else:
                update_spatial = update
            refined = self.apply_residual_connection(feat, update_spatial, i)
            refined_outputs.append(refined)
        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'final_hidden': h if hidden_state is not None else None
        }
        return refined_outputs, debug_info