"""
NGN Core Components - LayerGraph Base Class

This module defines the base LayerGraph class that learns which layers
should communicate with each other in a neural network.
"""


from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LayerGraph(keras.layers.Layer, ABC):
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
    ):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dims = feature_dims
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Learnable adjacency matrix: which layers connect to which
        # Shape: (num_layers, num_layers)
        # TensorFlow variables are created in build()
        self.adjacency_initializer = tf.constant_initializer(1.0 / num_layers)
        self.adjacency = None  # Will be created in build()

        # Layer normalization for stability
        if self.use_layer_norm:
            self.layer_norms = [layers.LayerNormalization(axis=-1) for dim in feature_dims]
        else:
            self.layer_norms = None

    def build(self, input_shape=None):
        # Create adjacency as a trainable variable
        if self.adjacency is None:
            self.adjacency = self.add_weight(
                shape=(self.num_layers, self.num_layers),
                initializer=self.adjacency_initializer,
                trainable=True,
                name="adjacency"
            )

    @abstractmethod
    def call(
        self,
        layer_outputs: List[tf.Tensor]
    ) -> Tuple[List[tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Apply learned layer graph communication.

        Args:
            layer_outputs: List of feature tensors from each layer
                          Shape: [f_1, f_2, ..., f_L] where f_i has shape
                          (batch_size, ...)

        Returns:
            refined_outputs: List of refined feature tensors after cross-layer communication
            debug_info: Dictionary with debugging information (attention weights, etc.)
        """
        pass

    def get_adjacency_matrix(self) -> tf.Tensor:
        """Return the current learned adjacency matrix."""
        return tf.identity(self.adjacency)

    def get_sparsity_mask(self, threshold: float = 0.1) -> tf.Tensor:
        """
        Get binary mask for sparse connections based on adjacency matrix.

        Args:
            threshold: Minimum connection strength to keep

        Returns:
            Binary mask of shape (num_layers, num_layers)
        """
        return tf.cast(self.adjacency > threshold, tf.float32)

    def apply_residual_connection(
        self,
        original: tf.Tensor,
        update: tf.Tensor,
        layer_idx: int
    ) -> tf.Tensor:
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
            # Flatten spatial dimensions for layer norm, then reshape back
            shape = tf.shape(refined)
            if len(refined.shape) == 4:
                batch_size, channels, height, width = shape[0], shape[1], shape[2], shape[3]
                refined_flat = tf.transpose(refined, [0, 2, 3, 1])  # (B, H, W, C)
                refined_flat = tf.reshape(refined_flat, [batch_size, height * width, channels])
                refined_flat = self.layer_norms[layer_idx](refined_flat)
                refined_flat = tf.reshape(refined_flat, [batch_size, height, width, channels])
                refined = tf.transpose(refined_flat, [0, 3, 1, 2])  # (B, C, H, W)
            else:
                refined = self.layer_norms[layer_idx](refined)
        return refined