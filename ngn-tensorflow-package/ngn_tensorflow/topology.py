"""
NGN Topology Module - Dynamic Routing and Graph Learning

This module handles advanced topology learning features like
per-input routing and hierarchical graphs.
"""

from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .graph import LayerGraph


class DynamicRoutingGate(layers.Layer):
    """
    Learns per-input routing decisions.

    This gate network determines which layers should be activated
    or emphasized for different inputs, enabling conditional computation.
    """

    def __init__(self, feature_dim: int, num_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.gate_network = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_layers, activation='sigmoid')
        ])

    def call(self, features: tf.Tensor) -> tf.Tensor:
        """
        Compute routing weights for each layer.

        Args:
            features: Input features (batch, channels, height, width)

        Returns:
            routing_weights: (batch, num_layers) - how much to activate each layer
        """
        return self.gate_network(features)


class HierarchicalLayerGraph(LayerGraph):
    """
    Hierarchical graph with intra-group and inter-group connections.

    Organizes layers into groups and learns connectivity at two levels:
    - Dense connections within groups
    - Sparse connections between groups
    """

    def __init__(
        self,
        num_layers: int,
        feature_dims: List[int],
        groups: List[int],  # e.g., [2, 2, 2, 2] for 4 groups of 2 layers each
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__(num_layers, feature_dims, use_residual, use_layer_norm)

        self.groups = groups
        self.num_groups = len(groups)

        # Validate groups sum to num_layers
        assert sum(groups) == num_layers, "Groups must sum to total number of layers"

        # Inter-group adjacency (sparse)
        self.inter_adjacency = self.add_weight(
            shape=(self.num_groups, self.num_groups),
            initializer=tf.constant_initializer(1.0 / self.num_groups),
            trainable=True,
            name='inter_adjacency'
        )

        # Intra-group connections are dense (all-to-all within group)
        # We'll handle this in call method

    def call(
        self,
        layer_outputs: List[tf.Tensor]
    ) -> Tuple[List[tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Apply hierarchical communication.
        """
        # For now, implement simple version
        # TODO: Implement full hierarchical logic

        # Placeholder: just return original outputs with adjacency info
        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'inter_adjacency': self.inter_adjacency,
            'groups': self.groups
        }

        return layer_outputs, debug_info

    def get_adjacency_matrix(self) -> tf.Tensor:
        """Return hierarchical adjacency matrix."""
        # Build full adjacency from inter and intra connections
        full_adj = tf.zeros((self.num_layers, self.num_layers), dtype=tf.float32)

        group_start = 0
        for i, group_size in enumerate(self.groups):
            group_end = group_start + group_size

            # Intra-group: dense connections
            indices = tf.meshgrid(tf.range(group_start, group_end), tf.range(group_start, group_end), indexing='ij')
            intra_indices = tf.stack([tf.reshape(idx, [-1]) for idx in indices], axis=1)
            full_adj = tf.tensor_scatter_nd_update(full_adj, intra_indices, tf.ones(tf.shape(intra_indices)[0], dtype=tf.float32))

            # Inter-group: sparse connections
            for j, other_size in enumerate(self.groups):
                if i != j:
                    other_start = sum(self.groups[:j])
                    other_end = other_start + other_size
                    indices = tf.meshgrid(tf.range(group_start, group_end), tf.range(other_start, other_end), indexing='ij')
                    inter_indices = tf.stack([tf.reshape(idx, [-1]) for idx in indices], axis=1)
                    values = tf.fill([tf.shape(inter_indices)[0]], self.inter_adjacency[i, j])
                    full_adj = tf.tensor_scatter_nd_update(full_adj, inter_indices, values)

            group_start = group_end

        return full_adj