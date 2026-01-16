"""
NGN Backbone Wrappers - ResNet Implementation

This module provides wrappers for popular backbone architectures
to integrate them with NGN layer graphs.
"""

from typing import List, Dict, Tuple, Optional, Union
import torch
from torch import nn, Tensor
import torchvision.models as models

from .graph import LayerGraph


class NGNResNet(nn.Module):
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
        backbone_name: str = 'resnet18',
        num_classes: int = 10,
        pretrained: bool = False,
        layer_graph: Optional[LayerGraph] = None
    ):
        super().__init__()

        # Load backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
            self.feature_dims = [64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Modify final layer for our num_classes
        self.backbone.fc = nn.Identity()  # Remove original classifier

        # Layer graph for NGN
        self.layer_graph = layer_graph
        if self.layer_graph is None:
            # Default: no communication (vanilla ResNet)
            self.layer_graph = IdentityLayerGraph(
                num_layers=len(self.layer_names),
                feature_dims=self.feature_dims
            )

        # Final classifier
        self.classifier = nn.Linear(self.feature_dims[-1], num_classes)

        # Hooks to capture intermediate features
        self.intermediate_features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture layer outputs."""
        def hook_fn(name):
            def hook(module, input, output):
                self.intermediate_features[name] = output
            return hook

        for name in self.layer_names:
            layer = getattr(self.backbone, name)
            layer.register_forward_hook(hook_fn(name))

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through NGN-ResNet.

        Args:
            x: Input tensor (batch, 3, height, width)

        Returns:
            logits: Classification logits (batch, num_classes)
            debug_info: Dictionary with intermediate features and debug info
        """
        # Clear previous features
        self.intermediate_features = {}

        # Forward through backbone to get final features
        final_features = self.backbone(x)

        # Collect layer outputs in order
        layer_outputs = [self.intermediate_features[name] for name in self.layer_names]

        # Apply NGN layer graph
        refined_outputs, graph_debug = self.layer_graph(layer_outputs)

        # Use refined final layer features for classification
        final_refined = refined_outputs[-1]

        # Global average pool
        pooled = final_refined.mean(dim=[2, 3])

        # Classify
        logits = self.classifier(pooled)

        # Prepare debug info
        debug_info = {
            'layer_outputs': layer_outputs,
            'refined_outputs': refined_outputs,
            'final_features': final_features,
            'pooled_features': pooled,
            **graph_debug
        }

        return logits, debug_info

    def get_layer_features(self) -> List[Tensor]:
        """Get the most recent layer features."""
        return [self.intermediate_features[name] for name in self.layer_names]


class IdentityLayerGraph(LayerGraph):
    """
    Identity layer graph that performs no communication.

    This is used as default when no layer graph is provided,
    effectively making the model a vanilla backbone.
    """

    def __init__(self, num_layers: int, feature_dims: List[int]):
        super().__init__(num_layers, feature_dims, use_residual=False, use_layer_norm=False)

    def forward(
        self,
        layer_outputs: List[Tensor]
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """Return layer outputs unchanged."""
        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'communication_type': 'identity'
        }
        return layer_outputs, debug_info


class NGNBackbone(nn.Module):
    """
    Generic backbone wrapper for NGN.

    This class can wrap any backbone that provides intermediate features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_hooks: List[str],  # Names of layers to hook
        feature_dims: List[int],  # Feature dims for each hooked layer
        layer_graph: LayerGraph,
        output_dim: int,
        num_classes: int
    ):
        super().__init__()

        self.backbone = backbone
        self.layer_hooks = layer_hooks
        self.feature_dims = feature_dims
        self.layer_graph = layer_graph

        # Hooks
        self.intermediate_features = {}
        self._register_hooks()

        # Classifier
        self.classifier = nn.Linear(output_dim, num_classes)

    def _register_hooks(self):
        """Register hooks for intermediate layers."""
        def hook_fn(name):
            def hook(module, input, output):
                self.intermediate_features[name] = output
            return hook

        for name in self.layer_hooks:
            if '.' in name:
                # Nested module (e.g., 'encoder.layer.0')
                parts = name.split('.')
                module = self.backbone
                for part in parts[:-1]:
                    module = getattr(module, part)
                layer_name = parts[-1]
                layer = getattr(module, layer_name)
            else:
                layer = getattr(self.backbone, name)

            layer.register_forward_hook(hook_fn(name))

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with NGN communication."""
        self.intermediate_features = {}

        # Forward through backbone
        _ = self.backbone(x)

        # Collect layer outputs
        layer_outputs = [self.intermediate_features[name] for name in self.layer_hooks]

        # Apply layer graph
        refined_outputs, graph_debug = self.layer_graph(layer_outputs)

        # Use final refined features
        final_features = refined_outputs[-1]

        # Pool and classify (assuming spatial features)
        if len(final_features.shape) == 4:  # (batch, channels, height, width)
            pooled = final_features.mean(dim=[2, 3])
        else:
            pooled = final_features

        logits = self.classifier(pooled)

        debug_info = {
            'layer_outputs': layer_outputs,
            'refined_outputs': refined_outputs,
            **graph_debug
        }

        return logits, debug_info