"""
NGN Backbone Wrappers - RNN Implementation

This module provides RNN-based backbones (LSTM, GRU networks)
integrated with NGN layer graphs.
"""

from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .graph import LayerGraph


class NGNLSTM(keras.Model):
    """
    LSTM backbone wrapped with NGN layer graph.

    Args:
        num_layers: Number of LSTM layers
        hidden_dim: Hidden dimension for LSTM
        vocab_size: Vocabulary size for embedding
        embed_dim: Embedding dimension
        num_classes: Number of output classes
        layer_graph: LayerGraph instance for inter-layer communication
    """

    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 256,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        num_classes: int = 10,
        layer_graph: Optional[LayerGraph] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, embed_dim)

        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(layers.LSTM(hidden_dim, return_sequences=True, return_state=True, name=f'lstm_{i}'))

        # Feature dims for each layer (all same for RNN)
        self.feature_dims = [hidden_dim] * num_layers

        # Layer graph for NGN
        self.layer_graph = layer_graph
        if self.layer_graph is None:
            self.layer_graph = IdentityLayerGraph(num_layers, self.feature_dims)

        # Classifier
        self.classifier = layers.Dense(num_classes)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass through NGN-LSTM.

        Args:
            x: Input tensor (batch, seq_len) - token indices

        Returns:
            logits: Classification logits (batch, num_classes)
            debug_info: Dictionary with intermediate features and debug info
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Pass through LSTM layers, collecting outputs
        layer_outputs = []
        current_input = embedded
        states = []

        for i, lstm_layer in enumerate(self.lstm_layers):
            output, state_h, state_c = lstm_layer(current_input)
            layer_outputs.append(output)  # (batch, seq_len, hidden_dim)
            states.append((state_h, state_c))
            current_input = output

        # Apply NGN layer graph
        refined_outputs, graph_debug = self.layer_graph(layer_outputs)

        # Use final refined features for classification
        final_refined = refined_outputs[-1]  # (batch, seq_len, hidden_dim)

        # Global average pool over sequence
        pooled = tf.reduce_mean(final_refined, axis=1)  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(pooled)

        # Prepare debug info
        debug_info = {
            'layer_outputs': layer_outputs,
            'refined_outputs': refined_outputs,
            'pooled_features': pooled,
            'states': states,
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