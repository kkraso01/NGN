# NGN Implementation Guide: PyTorch Best Practices

## Why PyTorch for NGN?

1. **Flexibility:** Easy to implement custom forward passes (bidirectional updates, dynamic routing)
2. **Debugging:** torch.autograd tracks gradients through complex compute graphs
3. **Ecosystem:** torchvision (backbones), TensorBoard (logging), WandB (experiment tracking)
4. **Community:** Most papers use PyTorch (easier to compare with DIANet, MRLA, DLA implementations)

## Project Structure

```
ngn/
├── core/
│   ├── __init__.py
│   ├── graph.py           # LayerGraph: adjacency learning + routing
│   ├── communication.py   # SharedAttentionAggregator, SharedRNNAggregator
│   └── stability.py       # GradientMonitor, StabilityCallback
│
├── backbones/
│   ├── __init__.py
│   ├── resnet.py          # ResNet wrapped for NGN
│   ├── transformer.py     # Transformer/ViT (future)
│   └── base.py            # Abstract backbone class
│
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Main training loop
│   ├── losses.py          # Classification + regularization losses
│   └── callbacks.py       # Logging, checkpointing, visualization
│
├── experiments/
│   ├── synthetic/
│   │   ├── train_xor.py
│   │   └── data.py
│   ├── vision/
│   │   ├── train_cifar10.py
│   │   ├── train_cifar100.py
│   │   └── train_imagenet.py (future)
│   └── nlp/
│       └── train_language_model.py (future)
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py   # Plot adjacency, attention, gradients
│   ├── analysis.py        # Interpret learned graphs
│   ├── profiling.py       # Memory, compute, latency
│   └── helpers.py         # Common utilities
│
├── requirements.txt
├── setup.py
└── README.md
```

## Key Implementation Principles

### 1. **Modular Design**
- `LayerGraph` knows nothing about backbones
- `SharedAttentionAggregator` is backbone-agnostic
- Easy to swap components

### 2. **Type Hints & Documentation**
```python
from typing import List, Dict, Tuple, Optional
import torch
from torch import nn, Tensor

class LayerGraph(nn.Module):
    """Learnable inter-layer communication graph.
    
    Args:
        num_layers: Number of layers in backbone
        feature_dims: List of feature dimensions per layer
        attention_heads: Number of attention heads
    """
    def __init__(
        self,
        num_layers: int,
        feature_dims: List[int],
        attention_heads: int = 8
    ) -> None:
        super().__init__()
        ...
```

### 3. **Comprehensive Logging**
```python
import logging
import torch.utils.tensorboard as tb

logger = logging.getLogger(__name__)
writer = tb.SummaryWriter('logs/')

# Log everything:
# - Attention weights per layer pair
# - Gradient norms (watch for explosion)
# - Adjacency matrix evolution
# - Layer-wise accuracy contributions
```

### 4. **Stability by Default**
```python
class SafeLayerGraph(nn.Module):
    def __init__(self, ...):
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.max_grad_norm = 1.0  # Gradient clipping
        self.use_residual = True  # Always add skip connection
```

## Core Implementation Roadmap

### Step 1: Define LayerGraph Base Class
```python
# ngn/core/graph.py
class LayerGraph(nn.Module):
    """Base class for layer connectivity graphs."""
    
    def __init__(self, num_layers, feature_dims):
        # Learn adjacency: which layers connect to which
        self.adjacency = nn.Parameter(torch.ones(num_layers, num_layers))
        
    def forward(self, layer_outputs: List[Tensor]) -> List[Tensor]:
        """Route features through learned graph."""
        raise NotImplementedError
```

### Step 2: Implement AttentionLayerGraph
```python
# ngn/core/communication.py
class AttentionLayerGraph(LayerGraph):
    """Uses shared multi-head attention for cross-layer communication."""
    
    def __init__(self, num_layers, feature_dims, num_heads=8):
        self.shared_attention = nn.MultiheadAttention(
            embed_dim=feature_dims[0],  # Assume same dim
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(feature_dims[0])
    
    def forward(self, layer_outputs) -> Tuple[List[Tensor], Dict]:
        """
        Args:
            layer_outputs: [f_1, f_2, ..., f_L]
        
        Returns:
            refined_outputs: [f'_1, f'_2, ..., f'_L]
            debug_info: {'attention_weights': ..., 'adjacency': ...}
        """
        refined_outputs = []
        attention_maps = []
        
        for i, query_feat in enumerate(layer_outputs):
            # Attend to all layers
            attn_out, attn_weights = self.shared_attention(
                query_feat,
                torch.stack(layer_outputs, dim=0),
                torch.stack(layer_outputs, dim=0)
            )
            
            # Residual + LayerNorm
            refined = query_feat + attn_out
            refined = self.layer_norm(refined)
            
            refined_outputs.append(refined)
            attention_maps.append(attn_weights)
        
        return refined_outputs, {'attention_weights': attention_maps}
```

### Step 3: Wrap Backbone (ResNet Example)
```python
# ngn/backbones/resnet.py
class NGNResNet(nn.Module):
    """ResNet wrapped with NGN layer graph."""
    
    def __init__(self, backbone_name='resnet18', num_classes=10):
        self.backbone = torchvision.models.resnet18(pretrained=False)
        
        # Hook into intermediate layers
        self.layer_hooks = []
        self.layer_outputs = []
        
        # Register hooks to capture intermediate features
        for name, module in self.backbone.named_modules():
            if 'layer' in name:
                module.register_forward_hook(self._capture_output)
        
        # Create layer graph
        self.layer_graph = AttentionLayerGraph(
            num_layers=4,  # ResNet has 4 layer groups
            feature_dims=[64, 128, 256, 512]
        )
        
        # Classification head
        self.fc = nn.Linear(512, num_classes)
    
    def _capture_output(self, module, input, output):
        """Hook to capture intermediate layer outputs."""
        self.layer_outputs.append(output)
    
    def forward(self, x):
        # Forward through backbone
        self.layer_outputs = []
        features = self.backbone(x)
        
        # Apply layer graph
        refined_outputs, debug_info = self.layer_graph(self.layer_outputs)
        
        # Aggregate and classify
        final_features = refined_outputs[-1].mean(dim=[2, 3])  # Global avg pool
        logits = self.fc(final_features)
        
        return logits, debug_info
```

### Step 4: Training Loop with Logging
```python
# ngn/training/trainer.py
class NGNTrainer:
    def __init__(self, model, device='cuda', log_dir='logs/'):
        self.model = model.to(device)
        self.device = device
        self.writer = tb.SummaryWriter(log_dir)
    
    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward
            logits, debug_info = self.model(x)
            
            # Loss
            loss = F.cross_entropy(logits, y)
            if 'attention_weights' in debug_info:
                # Add regularization
                entropy_reg = compute_entropy_reg(debug_info['attention_weights'])
                loss = loss + 0.1 * entropy_reg
            
            # Backward with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Log
            total_loss += loss.item()
            self.writer.add_scalar('loss', loss.item(), 
                                   epoch * len(dataloader) + batch_idx)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss:.4f}")
        
        return total_loss / len(dataloader)
```

### Step 5: Visualization Tools
```python
# ngn/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attention_weights: List[Tensor], epoch: int):
    """Plot attention matrix evolution during training."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, attn in enumerate(attention_weights[:4]):
        ax = axes[i // 2, i % 2]
        
        # Flatten batch dimension for visualization
        attn_np = attn.mean(dim=0).detach().cpu().numpy()
        
        sns.heatmap(attn_np, ax=ax, cmap='viridis')
        ax.set_title(f'Layer {i} Attention (Epoch {epoch})')
    
    plt.tight_layout()
    plt.savefig(f'logs/attention_epoch_{epoch}.png')
    plt.close()

def plot_adjacency_matrix(model: nn.Module, epoch: int):
    """Plot learned adjacency matrix."""
    adj = model.layer_graph.adjacency.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(adj, annot=True, fmt='.2f', cmap='RdYlGn')
    plt.title(f'Learned Adjacency (Epoch {epoch})')
    plt.savefig(f'logs/adjacency_epoch_{epoch}.png')
    plt.close()
```

## Dependencies (requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
tensorboard>=2.10.0
wandb>=0.13.0
tqdm>=4.64.0
pyyaml>=6.0
```

## Testing Strategy

```python
# tests/test_layer_graph.py
import torch
from ngn.core.communication import AttentionLayerGraph

def test_layer_graph_output_shape():
    """Verify LayerGraph preserves feature dimensions."""
    num_layers = 4
    batch_size = 8
    
    # Create dummy layer outputs
    layer_outputs = [
        torch.randn(batch_size, 64, 32, 32),   # Layer 1
        torch.randn(batch_size, 128, 16, 16),  # Layer 2
        torch.randn(batch_size, 256, 8, 8),    # Layer 3
        torch.randn(batch_size, 512, 4, 4),    # Layer 4
    ]
    
    graph = AttentionLayerGraph(num_layers, [64, 128, 256, 512])
    refined_outputs, _ = graph(layer_outputs)
    
    # Check shapes preserved
    for i, ref_out in enumerate(refined_outputs):
        assert ref_out.shape == layer_outputs[i].shape

def test_gradient_flow():
    """Ensure gradients flow through graph."""
    graph = AttentionLayerGraph(4, [64, 128, 256, 512])
    layer_outputs = [torch.randn(8, d, 8, 8, requires_grad=True) 
                     for d in [64, 128, 256, 512]]
    
    refined_outputs, _ = graph(layer_outputs)
    loss = sum(out.sum() for out in refined_outputs)
    loss.backward()
    
    # Check gradients exist
    assert all(out.grad is not None for out in layer_outputs)
```

## Next: Let's Code!

Ready to implement? I'll start with:
1. `ngn/core/graph.py` (LayerGraph base class)
2. `ngn/core/communication.py` (AttentionLayerGraph)
3. `ngn/backbones/resnet.py` (NGN-wrapped ResNet)

Then we'll test on synthetic data!
