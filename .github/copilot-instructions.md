# NGN (Neural Graph Network) — AI Agent Instructions

## Project Overview

**NGN** is a deep learning research codebase exploring **dynamic inter-layer communication** in neural networks. It treats each network layer as a learnable node in a graph, enabling bidirectional, input-dependent information flow beyond fixed forward propagation.

**Current Status:** Dual implementations (PyTorch and TensorFlow) with working vision experiments (CIFAR-10) and modular architecture ready for NLP/sequence extensions.

## Architecture Map

The codebase has **two independent implementations**:
- `ngn_pytorch-package/ngn_pytorch/` — Core PyTorch library (primary, actively maintained)  
- `ngn-tensorflow-package/ngn_tensorflow/` — TensorFlow mirror (same interface)
- `ngn_pytorch-experiments/` and `ngn-tensorflow-experiments/` — Experiment scripts

### Core Pattern: Backbone + LayerGraph

Every NGN model follows a 2-component design:

1. **Backbone** (CNN/RNN/Transformer): Produces layer features
2. **LayerGraph** (abstract base): Learns which layers communicate and how

```python
# Real usage pattern from train_cifar10.py
layer_graph = SharedAttentionAggregator(
    num_layers=4,
    feature_dims=[64, 128, 256, 512],  # ResNet-18 layer sizes
    num_heads=8
)
model = NGNResNet(
    backbone_name='resnet18',
    num_classes=10,
    layer_graph=layer_graph  # Plug in any LayerGraph
)
```

This design lets you swap LayerGraph implementations without touching the backbone.

## Key Codebase Patterns

### 1. LayerGraph Abstraction (graph.py)

All communication strategies inherit from `LayerGraph` (abstract base):

```python
# Defined in graph.py
class LayerGraph(nn.Module, ABC):
    def __init__(self, num_layers, feature_dims, use_residual=True, use_layer_norm=True):
        self.adjacency = nn.Parameter(...)  # Learnable (num_layers, num_layers) matrix
    
    @abstractmethod
    def forward(self, layer_outputs: List[Tensor]) -> Tuple[List[Tensor], Dict]:
        """Apply learned communication; return refined features + debug info"""
```

**Concrete implementations in communication.py:**
- `SharedAttentionAggregator` — Multi-head attention (MRLA-style); **current default**
- `SharedRNNAggregator` — Recurrent aggregation (DIANet-style); less stable
- Others can be added without changing backbone code

**Key design**: Residual updates prevent gradient collapse — updates are `features + learned_aggregation(...)`, not direct replacements.

### 2. Backbone Wrapping Pattern

Backbones (CNN/RNN/Transformer) are wrapped, not modified:

```python
# In cnn_backbone.py: NGNResNet
class NGNResNet(nn.Module):
    def __init__(self, backbone_name='resnet18', layer_graph=None):
        self.backbone = models.resnet18(pretrained=False)  # Unchanged ResNet
        self.layer_graph = layer_graph or IdentityLayerGraph(...)  # Plug in graph
        self._register_hooks()  # Capture intermediate layer features
    
    def forward(self, x):
        # Run backbone, capture layer outputs via hooks
        layer_outputs = [self.intermediate_features[name] for name in self.layer_names]
        # Apply NGN communication
        refined_outputs, debug_info = self.layer_graph(layer_outputs)
        # Classify from final refined feature
        return self.classifier(refined_outputs[-1])
```

This pattern applies to Transformer (`NGNTransformer`) and RNN (`NGNRecurrent`) too — same layer hooking strategy.

### 3. Stability & Training Patterns

**Stability is non-negotiable** for feedback loops. Key mechanisms in `stability.py`:

```python
# GradientMonitor: Track health of gradient flow
monitor = GradientMonitor(model)
monitor.check_layer_gradients()  # Detects vanishing/exploding

# Training pattern in trainer.py
if gradient_clip_norm:
    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)  # Default: 1.0

# Layer normalization + residual updates prevent collapse
# See LayerGraph.apply_residual_connection() for residual pattern
```

**Common issues & fixes:**
| Issue | Cause | Fix |
|-------|-------|-----|
| Loss spikes during training | Feedback loops amplify gradients | Reduce `gradient_clip_norm`, add layer norm |
| Learned adjacency collapses (one layer dominates) | Optimization flattens attention | Add entropy regularization to loss (see `losses.py`) |
| Out-of-memory on large models | All-pairs layer attention is O(L²) in memory | Use sparse adjacency mask or restrict to nearby layers |

### 4. Import Patterns (Critical for AI agents)

The codebase uses **module-level import paths** rather than direct top-level imports. This matters for debugging:

**❌ WRONG (causes ImportError):**
```python
from ngn_pytorch import SharedAttentionAggregator  # This may fail
```

**✅ RIGHT (use these paths):**
```python
from ngn_pytorch.communication import SharedAttentionAggregator  # Correct
from ngn_pytorch.cnn_backbone import NGNResNet
from ngn_pytorch.trainer import NGNTrainer
from ngn_pytorch.visualization import NGNVisualizer
```

See [`ngn_pytorch/__init__.py`](ngn_pytorch/__init__.py#L1) for what's re-exported at top level vs. what requires submodule import.

## Developer Workflows & Commands

### Setup & First Run

```bash
# Install the library in editable mode
cd ngn_pytorch-package
pip install -e .

# Or: Install TensorFlow version
cd ngn-tensorflow-package
pip install -e .
```

### Running Experiments

**Vision (CIFAR-10):**
```bash
# Train NGN-ResNet on CIFAR-10
python ngn_pytorch-experiments/vision/train_cifar10.py

# Train baseline ResNet (for comparison)
python ngn_pytorch-experiments/vision/train_cifar10.py --model baseline

# With visualization of learned graph
python ngn_pytorch-experiments/vision/train_cifar10.py --visualize_graphs
```

**Synthetic task (for debugging):**
```bash
# Train on XOR — verifies layer graph learns meaningful connections
python ngn_pytorch-experiments/synthetic/train_xor.py --visualize_graph
```

### Monitoring & Analysis

```bash
# Check gradient health during training
python -c "from ngn_pytorch.stability import GradientMonitor; 
           monitor = GradientMonitor(model); 
           monitor.check_layer_gradients()"

# Visualize learned layer connectivity
python -c "from ngn_pytorch.visualization import NGNVisualizer; 
           viz = NGNVisualizer(); 
           viz.plot_adjacency_matrix(checkpoint_path)"

# Profile memory overhead vs. baseline
python -c "from ngn_pytorch.profiling import NGNProfiler; 
           profiler = NGNProfiler(); 
           profiler.profile_memory(model)"
```

### Key Files for Common Tasks

| Task | File | Key Class/Function |
|------|------|-------------------|
| Add new LayerGraph type | `ngn_pytorch/communication.py` | Inherit from `LayerGraph`, implement `forward()` |
| Wrap new backbone | `ngn_pytorch/cnn_backbone.py` | Copy `NGNResNet` pattern, register hooks for layers |
| Debug training instability | `ngn_pytorch/stability.py` | Use `GradientMonitor`, `StabilityCallback` |
| Analyze learned connections | `ngn_pytorch/analysis.py` | Use `GraphAnalyzer.extract_edge_list()` |
| Run vision baseline | `ngn_pytorch-experiments/vision/train_cifar10.py` | Pass `--model baseline` flag |

## Common Pitfalls & How to Avoid Them

### Pitfall 1: Direct imports fail
**❌ Wrong:**
```python
from ngn_pytorch import SharedAttentionAggregator  
# → ImportError or gets identity class
```
**✅ Right:**
```python
from ngn_pytorch.communication import SharedAttentionAggregator
```

### Pitfall 2: Forgetting to normalize intermediate features
**Problem:** Different layers may have vastly different feature scales (e.g., ResNet layer1 outputs 64 channels, layer4 outputs 512). Attention mechanism will bias toward larger-scale features.

**Solution:** Always use layer normalization in LayerGraph.__init__:
```python
self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in feature_dims])
```

### Pitfall 3: Creating feature dimension mismatches
**Problem:** `SharedAttentionAggregator` assumes all features project to same embed_dim. Mismatched dimensions cause attention errors.

**Solution:** The class includes projection layers:
```python
self.projections = nn.ModuleList([
    nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity()
    for dim in feature_dims
])
```

### Pitfall 4: Not clipping gradients
**Problem:** Feedback loops → exploding gradients → NaN loss.

**Solution:** Always set `gradient_clip_norm` in trainer:
```python
trainer = NGNTrainer(..., gradient_clip_norm=1.0)
```

## Reference & Contexts

**Research gaps NGN addresses:**
- Beyond vision (NLP, audio, RL generality)
- Hierarchical layer graphs (graph-of-graphs)
- Per-input topology adaptation (conditional routing)
- Training stability theory (convergence with feedback)
- Interpretability (what does learned graph reveal?)
- Efficiency (hardware-aware execution)

**Must-read docs** for context:
- `DOCUMENTATION/REFERENCE/CONVERSTATION.MD` — Literature review (why NGN matters)
- `DOCUMENTATION/ARCHITECTURE/ARCHITECTURE.md` — Technical design (layer graph, attention, loss)
- `DOCUMENTATION/REFERENCE/IMPLEMENTATION_GUIDE.md` — PyTorch patterns & best practices

## Inter-Component Communication Patterns

### Data Flow Through NGN

```
Input x → Backbone (ResNet/Transformer) → Layer outputs [f1, f2, ..., fL]
                                               ↓
                                          LayerGraph
                                       (SharedAttention or RNN)
                                               ↓
                              Refined outputs [f'1, f'2, ..., f'L]
                                               ↓
                                    Output head (Linear classifier)
                                               ↓
                                    Logits / Predictions
```

### Layer ↔ Layer Communication

- **Adjacency matrix** (`LayerGraph.adjacency`): Learnable (L×L) weights that gate which layers influence which
- **Residual updates**: `refined_f_i = f_i + aggregation(...)` — prevents gradient collapse
- **Shared modules**: All layers use same attention/RNN unit (reduces parameters, improves stability)

### Training with NGN

1. Forward pass: Collect all layer outputs via registered hooks
2. Apply LayerGraph: Get refined features + attention weights (for visualization/analysis)
3. Compute loss: CE loss + optional entropy regularization on adjacency matrix (prevent collapse)
4. Backward + clip gradients (1.0 default) — critical for stability
5. Optional: Monitor gradient health via `GradientMonitor`

## Integration & Dependencies

**PyTorch (primary):**
- `torch.nn`: Core modules (Linear, MultiheadAttention, LayerNorm)
- `torchvision.models`: Pre-trained ResNet/backbone loading
- `torch.optim`: Optimizers (Adam, SGD)

**TensorFlow (parallel implementation):**
- Same interface as PyTorch; mirror at `ngn-tensorflow-package/`
- Use `tf.keras.layers`, `tf.nn.multi_head_attention`

**Data & Vision (experiments):**
- `torchvision.datasets`: CIFAR-10, ImageNet (vision/train_cifar10.py)
- `numpy`, `matplotlib`: Analysis and visualization

**Analysis & Debugging:**
- `tensorboard`: Monitor training curves (logs/ directory)
- `networkx`: Graph analysis (analysis.py)
- `matplotlib`: Plot adjacency matrices, attention heatmaps (visualization.py)

## Conventions & Project-Specific Practices

1. **Paper-first development:** Changes should be motivated by literature gaps or research questions (reference `CONVERSTATION.MD`)
2. **Reproducibility:** All experiments logged with random seed, hyperparameters, and trained model checkpoints
3. **Ablation discipline:** Test hypotheses methodically (e.g., "do dynamic graphs outperform static ones?")
4. **Generalization testing:** If claiming generality beyond vision, test on multiple domains (not just CIFAR)
5. **Stability guarantees:** Theoretical or empirical evidence that training remains numerically stable (not assumption)
6. **Dual implementations:** Maintain PyTorch and TensorFlow versions in parallel—same interface, different backends

## Debugging & Troubleshooting Tips

| Issue | Likely Cause | Debug Approach |
|-------|--------------|---|
| Gradient explosion/vanishing in feedback loops | Unscaled cross-layer updates | Add layer norm, residual connections, or gradient clipping |
| Learned graph collapses (all attention on one layer) | Optimization landscape too flat | Try regularization (entropy penalty on attention), warm-start from static baseline |
| Training diverges with dynamic topology | Input-dependent routing destabilizes | Simplify first: test static learned graphs before per-input adaptation |
| Slow training/inference | Overhead from all-pairs layer connections | Sparsify graph (learned pruning), or restrict to nearby layers initially |
