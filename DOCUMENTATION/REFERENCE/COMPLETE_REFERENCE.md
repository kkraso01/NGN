# NGN Architecture Complete Reference

## Quick Reference Guide

### **What is NGN?**
A learnable meta-architecture that learns which layers in any neural network should communicate with each other, using shared multi-head attention and residual updates.

### **Key Insight**
```
Standard Networks:  Layer1 → Layer2 → Layer3 → Layer4
                    (fixed sequence, no backflow)

NGN Networks:       Layer1 ←→ Layer2 ←→ Layer3 ←→ Layer4
                    (learned routing, bidirectional)
```

---

## Core Components (Implementation Order)

### 1. **LayerGraph** (Base Class)
```python
# ngn/core/graph.py
class LayerGraph(nn.Module):
    """Base class for learned layer connectivity."""
    
    def __init__(self, num_layers: int, feature_dims: List[int]):
        # Learn which layers connect to which
        self.adjacency = nn.Parameter(torch.ones(num_layers, num_layers))
    
    def forward(self, layer_outputs: List[Tensor]) -> Tuple[List[Tensor], Dict]:
        """Route features through learned graph."""
        raise NotImplementedError
```

### 2. **SharedAttentionAggregator** (Main Communication Module)
```python
# ngn/core/communication.py
class SharedAttentionAggregator(LayerGraph):
    """Uses multi-head attention for cross-layer communication."""
    
    def __init__(self, num_layers, feature_dims, num_heads=8):
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, layer_outputs):
        refined_outputs = []
        attention_weights = []
        
        for i, query_feat in enumerate(layer_outputs):
            # Layer i attends to ALL other layers
            attn_out, weights = self.attention(
                query=query_feat,
                key=stack(layer_outputs),
                value=stack(layer_outputs)
            )
            
            # Residual + LayerNorm for stability
            refined = query_feat + attn_out
            refined = self.layer_norm(refined)
            
            refined_outputs.append(refined)
            attention_weights.append(weights)
        
        return refined_outputs, {'attention_weights': attention_weights}
```

### 3. **NGNResNet** (Backbone Wrapper)
```python
# ngn/backbones/resnet.py
class NGNResNet(nn.Module):
    """ResNet wrapped with NGN layer graph."""
    
    def __init__(self, backbone_name='resnet18', num_classes=10):
        self.backbone = torchvision.models.resnet18()
        
        # Register hooks to capture layer outputs
        self.layer_outputs = []
        self._register_hooks()
        
        # Create LayerGraph
        self.layer_graph = SharedAttentionAggregator(
            num_layers=4,
            feature_dims=[64, 128, 256, 512]
        )
        
        # Classification head
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        self.layer_outputs = []
        features = self.backbone(x)
        
        # Apply NGN
        refined, debug = self.layer_graph(self.layer_outputs)
        
        # Classify
        final = refined[-1].mean(dim=[2, 3])  # Global avg pool
        logits = self.fc(final)
        
        return logits, debug
```

### 4. **Training Loop**
```python
# ngn/training/trainer.py
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Forward
        logits, debug_info = model(x)
        
        # Loss
        loss = F.cross_entropy(logits, y)
        
        # Optional: Add regularization
        if 'attention_weights' in debug_info:
            entropy_reg = compute_entropy_regularization(
                debug_info['attention_weights']
            )
            loss = loss + 0.1 * entropy_reg
        
        # Backward with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log
        if batch_idx % 10 == 0:
            print(f"Loss: {loss:.4f}")
```

### 5. **Visualization**
```python
# ngn/utils/visualization.py
def plot_attention_heatmap(attention_weights, epoch):
    """Plot how each layer attends to others."""
    for i, attn in enumerate(attention_weights):
        attn_np = attn.mean(dim=0).detach().cpu().numpy()
        plt.figure()
        sns.heatmap(attn_np, cmap='viridis')
        plt.title(f'Layer {i} Attention (Epoch {epoch})')
        plt.savefig(f'logs/attention_layer_{i}_epoch_{epoch}.png')
        plt.close()
```

---

## Implementation Checklist

### Phase 1: Foundation
- [ ] Project structure (directories, requirements.txt, setup.py)
- [ ] LayerGraph base class (abstract forward method)
- [ ] SharedAttentionAggregator (multi-head attention module)
- [ ] NGNResNet backbone wrapper
- [ ] Basic training loop (no logging yet)
- [ ] Test on synthetic XOR task

### Phase 2: Vision Validation
- [ ] CIFAR-10 training script
- [ ] Attention visualization
- [ ] Compare vs. vanilla ResNet
- [ ] Log metrics (accuracy, loss, gradients)

### Phase 3: Interpretability
- [ ] Adjacency matrix heatmaps
- [ ] Identify layer "hubs" and communities
- [ ] Analyze attention weight distributions

### Phase 4: Generalization
- [ ] NLP task (language model)
- [ ] Sequence task (time series)
- [ ] Prove same LayerGraph works across domains

### Phase 5: Optimization (Time Permitting)
- [ ] Per-input routing (dynamic topology)
- [ ] Hierarchical graphs (graph-of-graphs)
- [ ] Memory/compute profiling

---

## Key Design Principles (Must Follow)

| Principle | Why | Implementation |
|-----------|-----|---|
| **Residual Connections** | Gradient preservation | `refined = original + α * context` |
| **Shared Aggregator** | Fewer params, more stable | One attention module for ALL layers |
| **LayerNorm** | Activation stability | Normalize after every cross-layer op |
| **Gradient Clipping** | Prevent explosion | `clip_grad_norm(..., max_norm=1.0)` |
| **Input-Dependent Attention** | Adaptive routing | Attention weights vary per input |
| **Sparse Graph** (learned) | Interpretability + efficiency | Some edges ≈ 0; learned during training |

---

## Stability Safeguards (Critical!)

```python
# In SharedAttentionAggregator.forward():

for i, query_feat in enumerate(layer_outputs):
    # 1. Multi-head attention
    attn_out, weights = self.attention(query, keys, values)
    
    # 2. Residual connection (critical!)
    refined = query_feat + attn_out
    
    # 3. Layer normalization
    refined = self.layer_norm(refined)
    
    # 4. Gradient clipping (in trainer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Result: Stable training even with bidirectional updates ✓
```

---

## Debugging Tips

| Issue | Diagnostic | Solution |
|-------|-----------|----------|
| **Loss NaN** | Check gradient norms | Add gradient clipping, reduce learning rate |
| **Loss not decreasing** | Check attention weights | Weights should show clear patterns by epoch 50 |
| **Gradient explosion** | Monitor ||grad|| per layer | Use layer norm, residual connections |
| **Attention collapses** | All weight on one layer | Add entropy regularization on attention |
| **Memory OOM** | Profile with `torch.cuda.memory_summary()` | Reduce batch size or use gradient checkpointing |

---

## Success Criteria

### Phase 1 (Synthetic XOR)
- ✓ Accuracy → 95%+
- ✓ Loss curves smooth (no spikes)
- ✓ Gradient norms bounded (<1.0 after clipping)
- ✓ Attention weights show clear patterns (not uniform)

### Phase 2 (CIFAR-10)
- ✓ NGN ResNet-18 beats vanilla ResNet-18 by 1-2%
- ✓ Attention heatmaps interpretable
- ✓ Training stable for 200+ epochs
- ✓ No overfitting (train/val gap reasonable)

### Phase 3 (Interpretability)
- ✓ Can identify layer hubs (high in/out attention)
- ✓ Adjacent layers show higher attention (expected structure)
- ✓ Different input classes show different routing patterns

### Phase 4 (Generalization)
- ✓ Same LayerGraph improves NLP baseline too
- ✓ Same architecture, different backbones, consistent gains
- ✓ Proves: "Dynamic layer graphs are universal, not vision-specific"

---

## Research Narrative (For Paper/Presentation)

**Title:** "Neural Graph Networks: Learning Dynamic Inter-Layer Communication"

**Contribution Summary:**
1. **Generic Framework:** One LayerGraph works across CNNs, Transformers, RNNs (generality)
2. **Interpretable:** Visualization tools reveal what network learns (interpretability)
3. **Stable:** Theoretical + empirical evidence of training stability with feedback (stability)
4. **Efficient:** <5% overhead vs. baseline (efficiency)
5. **Extensible:** Foundation for hierarchical & per-input routing (extensibility)

**Key Finding:** Dynamic inter-layer communication is a *fundamental architectural principle*, not just a vision trick. We prove this by validating across multiple domains, architectures, and scales.

---

## Next Immediate Steps

1. **Create project structure**
   ```bash
   mkdir -p ngn/{core,backbones,training,experiments/synthetic,experiments/vision,experiments/nlp,utils}
   touch ngn/__init__.py ngn/core/__init__.py ngn/backbones/__init__.py ...
   ```

2. **Create requirements.txt**
   ```
   torch>=2.0.0
   torchvision>=0.15.0
   numpy>=1.24.0
   matplotlib>=3.6.0
   seaborn>=0.12.0
   tensorboard>=2.10.0
   ```

3. **Start with LayerGraph & SharedAttentionAggregator**
   - These are the core, everything else builds on them

4. **Test on XOR first**
   - Fast iteration (seconds/epoch)
   - Verify correctness before CIFAR-10

5. **Then CIFAR-10**
   - Real benchmark, meaningful accuracy gains
   - Visualize attention patterns

---

## Code References

- **ARCHITECTURE.md** — Technical design details
- **IMPLEMENTATION_GUIDE.md** — PyTorch patterns & best practices
- **ARCHITECTURE_DIAGRAMS.md** — Visual representations
- **RESEARCH_HOLES.md** — Which research gaps each component addresses

All documentation is **linked, cross-referenced, and ready to guide implementation.**

---

## Final Thought

> **NGN is not a specific model; it's a *pattern* for learning dynamic connectivity in any neural network.**
>
> Once implemented, you'll have a reusable framework:
> - Swap backbones (ResNet → ViT → Transformer → RNN)
> - Swap tasks (vision → NLP → sequence → audio)
> - Same LayerGraph logic, different applications
>
> This is the power of designing for **generality first**.
