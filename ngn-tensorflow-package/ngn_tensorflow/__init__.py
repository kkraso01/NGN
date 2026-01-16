"""NGN-TensorFlow: Neural Graph Network - TensorFlow Implementation

A minimal, clean implementation of NGN (Neural Graph Network) using TensorFlow
that enables dynamic inter-layer communication in neural networks.

Core Components:
- LayerGraph: Base class for learning layer connectivity
- SharedAttentionAggregator: Attention-based cross-layer communication  
- SharedRNNAggregator: RNN-based cross-layer communication
- DynamicRoutingGate: Per-input adaptive topology
- HierarchicalLayerGraph: Multi-level graph structure

Backbones:
- NGNResNet: ResNet wrapped with NGN
- NGNLSTM: LSTM wrapped with NGN
- NGNTransformer: Transformer wrapped with NGN

Training:
- NGNTrainer: Training loop with stability monitoring
- NGNLoss: Combined classification and regularization loss
- GradientMonitor: Gradient health tracking
- Callbacks: StabilityCallback, EarlyStoppingCallback

Utilities:
- NGNVisualizer: Visualize attention weights, adjacency matrices
- GraphAnalyzer: Analyze learned graph structure
- NGNProfiler: Profile memory, compute, latency

Examples:
    from ngn_tensorflow import NGNResNet, SharedAttentionAggregator
    
    # Create attention-based layer graph
    layer_graph = SharedAttentionAggregator(
        num_layers=4,
        feature_dims=[64, 128, 256, 512]
    )
    
    # Wrap ResNet with NGN
    model = NGNResNet(
        backbone_name='resnet50',
        num_classes=10,
        layer_graph=layer_graph
    )
"""

__version__ = "0.1.0"
__author__ = "NGN Research Team"
__license__ = "MIT"

# Core components
from .graph import LayerGraph
from .communication import SharedAttentionAggregator, SharedRNNAggregator
from .topology import DynamicRoutingGate, HierarchicalLayerGraph

# Backbones
from .cnn_backbone import NGNResNet, NGNBackbone, IdentityLayerGraph
from .rnn_backbone import NGNLSTM
from .transformer_backbone import NGNTransformer

# Training
from .trainer import NGNTrainer
from .losses import NGNLoss, compute_accuracy
from .stability import (
    GradientMonitor,
    TrainingCallback,
    StabilityCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
)

# Utils
from .visualization import NGNVisualizer
from .analysis import GraphAnalyzer
from .profiling import NGNProfiler, benchmark_models, compare_ngn_to_baseline

__all__ = [
    # Core
    'LayerGraph',
    'SharedAttentionAggregator',
    'SharedRNNAggregator',
    'DynamicRoutingGate',
    'HierarchicalLayerGraph',
    # Backbones
    'NGNResNet',
    'NGNBackbone',
    'IdentityLayerGraph',
    'NGNLSTM',
    'NGNTransformer',
    # Training
    'NGNTrainer',
    'NGNLoss',
    'GradientMonitor',
    'TrainingCallback',
    'StabilityCallback',
    'EarlyStoppingCallback',
    'LearningRateSchedulerCallback',
    'compute_accuracy',
    # Utils
    'NGNVisualizer',
    'GraphAnalyzer',
    'NGNProfiler',
    'benchmark_models',
    'compare_ngn_to_baseline',
]