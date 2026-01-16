# NGN-PyTorch: Neural Graph Network (PyTorch-Only Version)

A PyTorch-only implementation of Neural Graph Networks for dynamic inter-layer communication in neural networks.

## Overview

NGN treats each layer in a neural network as a node in a learnable graph, enabling bidirectional, input-dependent information flow beyond fixed forward propagation.

## Installation

```bash
pip install -r requirements.txt
python setup.py develop
```

## Quick Start

```python
from ngn_pytorch.core.communication import SharedAttentionAggregator
from ngn_pytorch.training.trainer import NGNTrainer

# Create your model with NGN layer communication
layer_graph = SharedAttentionAggregator(
    num_layers=3,
    feature_dims=[64, 128, 256],
    num_heads=8
)

# Train with built-in stability monitoring
trainer = NGNTrainer(model, optimizer)
trainer.train_epoch(train_loader, epoch=0)
```

## Key Features

- **Dynamic Layer Communication**: Learn which layers should communicate
- **Shared Attention**: Multi-head attention for cross-layer information flow
- **Stability Monitoring**: Built-in gradient clipping and monitoring
- **PyTorch Native**: No external logging dependencies

## Project Structure

```
ngn_pytorch/
├── core/              # Core NGN components
├── backbones/         # Model backbones (CNN, Transformer, RNN)
├── training/          # Training utilities and stability monitoring
├── utils/             # Visualization and analysis tools
└── experiments/       # Example experiments
```