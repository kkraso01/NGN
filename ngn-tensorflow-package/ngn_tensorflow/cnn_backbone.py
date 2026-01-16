"""
NGN Backbone Wrappers - ResNet Implementation

This module provides wrappers for popular backbone architectures
to integrate them with NGN layer graphs.
"""

from typing import List, Dict, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications

from .graph import LayerGraph


class NGNResNet(keras.Model):
    """
    ResNet backbone wrapped with NGN layer graph.

    This class loads a pre-trained ResNet and extracts intermediate
    layer features for NGN communication.

    Args:
        backbone_name: Name of ResNet variant ('resnet18', 'resnet50', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        layer_graph: LayerGraph instance for inter-layer communication
    """

    def __init__(
        self,
        backbone_name: str = 'resnet50',
        num_classes: int = 10,
        pretrained: bool = False,
        layer_graph: Optional[LayerGraph] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Load backbone - TensorFlow has ResNet50, but not ResNet18 directly
        # We'll use ResNet50 as default, can extend for others
        if backbone_name == 'resnet50':
            self.backbone = applications.ResNet50(
                include_top=False,
                weights='imagenet' if pretrained else None,
                input_shape=(32, 32, 3)  # Changed for CIFAR-10
            )
            # For ResNet50, intermediate layers are conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out
            self.layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Use 'resnet50'.")

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Layer graph for NGN
        self.layer_graph = layer_graph
        if self.layer_graph is None:
            # Default: no communication (vanilla ResNet)
            self.layer_graph = IdentityLayerGraph(
                num_layers=len(self.layer_names),
                feature_dims=self.feature_dims
            )

        # Final classifier
        self.classifier = layers.Dense(num_classes)

        # Build intermediate model for feature extraction
        self._build_intermediate_model()

    def _build_intermediate_model(self):
        """Build a model that outputs intermediate features."""
        inputs = self.backbone.input
        outputs = [self.backbone.get_layer(name).output for name in self.layer_names]
        self.intermediate_model = keras.Model(inputs=inputs, outputs=outputs)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass through NGN-ResNet.

        Args:
            x: Input tensor (batch, height, width, 3)

        Returns:
            logits: Classification logits (batch, num_classes)
            debug_info: Dictionary with intermediate features and debug info
        """
        # Get intermediate features
        layer_outputs = self.intermediate_model(x)

        # Apply NGN layer graph
        refined_outputs, graph_debug = self.layer_graph(layer_outputs)

        # Use refined final layer features for classification
        final_refined = refined_outputs[-1]

        # Global average pool
        pooled = tf.reduce_mean(final_refined, axis=[1, 2])

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

    def get_layer_features(self, x: tf.Tensor) -> List[tf.Tensor]:
        """Get layer features for input x."""
        return self.intermediate_model(x)


class IdentityLayerGraph(LayerGraph):
    """
    Identity layer graph that performs no communication.

    This is used as default when no layer graph is provided,
    effectively making the model a vanilla backbone.
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


class NGNBackbone(keras.Model):
    """
    Generic backbone wrapper for NGN.

    This class can wrap any backbone that provides intermediate features.
    """

    def __init__(
        self,
        backbone: keras.Model,
        layer_hooks: List[str],  # Names of layers to hook
        feature_dims: List[int],  # Feature dims for each hooked layer
        layer_graph: LayerGraph,
        output_dim: int,
        num_classes: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.layer_hooks = layer_hooks
        self.feature_dims = feature_dims
        self.layer_graph = layer_graph

        # Build intermediate model
        inputs = self.backbone.input
        outputs = [self.backbone.get_layer(name).output for name in self.layer_hooks]
        self.intermediate_model = keras.Model(inputs=inputs, outputs=outputs)

        # Classifier
        self.classifier = layers.Dense(num_classes)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Forward pass with NGN communication."""
        # Get intermediate features
        layer_outputs = self.intermediate_model(x)

        # Apply layer graph
        refined_outputs, graph_debug = self.layer_graph(layer_outputs)

        # Use final refined features
        final_features = refined_outputs[-1]

        # Pool and classify (assuming spatial features)
        if len(final_features.shape) == 4:  # (batch, height, width, channels)
            pooled = tf.reduce_mean(final_features, axis=[1, 2])
        else:
            pooled = final_features

        logits = self.classifier(pooled)

        debug_info = {
            'layer_outputs': layer_outputs,
            'refined_outputs': refined_outputs,
            **graph_debug
        }

        return logits, debug_info