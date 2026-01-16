"""
NGN Backbone Wrappers - Transformer Implementation

This module provides transformer-based backbones (ViT, BERT, etc.)
integrated with NGN layer graphs.
"""

from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .graph import LayerGraph


class NGNTransformer(keras.Model):
    """
    Transformer backbone wrapped with NGN layer graph.

    Args:
        num_layers: Number of transformer layers
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        num_classes: Number of output classes
        layer_graph: LayerGraph instance for inter-layer communication
    """

    def __init__(
        self,
        num_layers: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        vocab_size: int = 10000,
        max_len: int = 100,
        num_classes: int = 10,
        layer_graph: Optional[LayerGraph] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_classes = num_classes

        # Positional encoding
        self.pos_encoding = self._get_positional_encoding(max_len, embed_dim)

        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, embed_dim)

        # Transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                keras.Sequential([
                    layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim),
                    layers.LayerNormalization(epsilon=1e-6),
                    layers.Dense(ff_dim, activation='relu'),
                    layers.Dense(embed_dim),
                    layers.LayerNormalization(epsilon=1e-6)
                ], name=f'transformer_{i}')
            )

        # Feature dims for each layer (all same for transformer)
        self.feature_dims = [embed_dim] * num_layers

        # Layer graph for NGN
        self.layer_graph = layer_graph
        if self.layer_graph is None:
            self.layer_graph = IdentityLayerGraph(num_layers, self.feature_dims)

        # Classifier
        self.classifier = layers.Dense(num_classes)

    def _get_positional_encoding(self, max_len, embed_dim):
        """Generate positional encodings."""
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))
        angle_rads = pos * angle_rates

        # Apply sin to even indices, cos to odd
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass through NGN-Transformer.

        Args:
            x: Input tensor (batch, seq_len) - token indices

        Returns:
            logits: Classification logits (batch, num_classes)
            debug_info: Dictionary with intermediate features and debug info
        """
        seq_len = tf.shape(x)[1]

        # Embedding + positional encoding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded += self.pos_encoding[:, :seq_len, :]

        # Pass through transformer layers, collecting outputs
        layer_outputs = []
        current_input = embedded

        for transformer_layer in self.transformer_layers:
            output = transformer_layer(current_input)
            layer_outputs.append(output)
            current_input = output

        # Apply NGN layer graph
        refined_outputs, graph_debug = self.layer_graph(layer_outputs)

        # Use final refined features for classification
        final_refined = refined_outputs[-1]  # (batch, seq_len, embed_dim)

        # Global average pool over sequence
        pooled = tf.reduce_mean(final_refined, axis=1)  # (batch, embed_dim)

        # Classify
        logits = self.classifier(pooled)

        # Prepare debug info
        debug_info = {
            'layer_outputs': layer_outputs,
            'refined_outputs': refined_outputs,
            'pooled_features': pooled,
            **graph_debug
        }

        return logits, debug_info


class IdentityLayerGraph(LayerGraph):
    """
    Identity layer graph that performs no communication.
    """

    def __init__(self, num_layers: int, feature_dims: List[int]):
        super().__init__(num_layers, feature_dims, use_residual=False, use_layer_norm=False)

    def call(
        self,
        layer_outputs: List[tf.Tensor]
    ) -> Tuple[List[tf.Tensor], Dict[str, tf.Tensor]]:
        """Return layer outputs unchanged."""
        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'communication_type': 'identity'
        }
        return layer_outputs, debug_info