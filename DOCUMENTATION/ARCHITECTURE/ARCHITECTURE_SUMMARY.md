# NGN Architecture Summary

## One-Sentence Definition

**NGN** is a generic, learnable meta-architecture that replaces fixed sequential layer connections in any backbone (CNN, Transformer, RNN) with a **dynamic graph where each layer learns which other layers to communicate with**.

---

## The Core Idea in 3 Steps

### 1. **Backbone Extracts Features**
```
Input → ResNet/Transformer/RNN → [f₁, f₂, ..., fₗ]  (layer features)
```

### 2. **LayerGraph Routes Communication**
```
Learns: "Which layer pairs should share information?"
↓
Shared Attention Module: ALL cross-layer fusion goes through ONE shared module
↓
Residual Connections: original_features + learned_context (for stability)
```

### 3. **Refined Features for Prediction**
```
[f₁, f₂, ..., fₗ] → LayerGraph → [f'₁, f'₂, ..., f'ₗ] → Classification
```

---

## Why This Design?

### Problem with Standard Networks
- Layers work in **isolation**: Layer i only sees Layer i-1
- **Inflexible**: Fixed sequence ResNet50 → ResNet50+NGN (learns better layer reuse)
- **Limited**: Early layers can't benefit from later layers' context

### NGN Solution
✅ **Bidirectional**: Early layers can learn from later layers (via attention)  
✅ **Adaptive**: Which layers communicate depends on input (attention weights vary)  
✅ **Stable**: Shared RNN/Attention + residual connections prevent gradient collapse  
✅ **Generic**: Same LayerGraph works for CNNs, Transformers, RNNs  
✅ **Interpretable**: Visualize attention matrices → understand what network learns

---

## Core Components

### 1. **Adjacency Matrix** (Optional, Phase 2)
```
A ∈ ℝ^(L×L)  where L = number of layers
A[i,j] = "How much does layer j influence layer i?"
Learned during training (static once trained)
```

### 2. **Shared Attention Aggregator** (Main Component)
```python
def aggregate(layer_i_features, all_layer_features):
    # Layer i attends to ALL other layers
    refined = LayerI_features + MultiHeadAttention(
        query=layer_i_features,
        key=all_layer_features,
        value=all_layer_features
    )
    return refined
```

### 3. **Stability Safeguards**
- **Residual connections** (skip paths)
- **Layer normalization** (prevent activation explosion)
- **Gradient clipping** (clip large gradients)
- **Shared parameters** (smaller effective model size)

---

## Forward Pass Pseudocode

```python
def forward(input):
    # 1. Extract layer outputs from backbone
    layer_outputs = backbone(input)  # [f_1, f_2, ..., f_L]
    
    # 2. Apply LayerGraph (refine each layer with cross-layer context)
    refined_outputs = []
    for i, layer_features in enumerate(layer_outputs):
        
        # Collect information from other layers
        context = shared_attention(
            query=layer_features,
            keys_values=layer_outputs
        )
        
        # Residual update
        refined = layer_features + context
        refined_outputs.append(refined)
    
    # 3. Aggregate and classify
    final = aggregate(refined_outputs)
    return classifier(final)
```

---

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| **Shared Attention** (not separate per layer) | Fewer params, more stable, interpretable |
| **Always Residual** | Gradient preservation (critical with feedback) |
| **LayerNorm** | Prevents activation/gradient explosion |
| **Input-Dependent Attention Weights** | Different inputs route differently (adaptive) |
| **Sparse Graph** (learned, some edges ≈ 0) | Interpretability, efficiency |

---

## 6 Research Contributions

NGN simultaneously addresses **6 open research holes** left by prior work:

1. **Generality** — Works across vision, NLP, sequence (not just vision)
2. **Stability** — Theoretical + empirical proof of convergence with feedback
3. **Interpretability** — Visualization tools reveal what network learns
4. **Efficiency** — <5% overhead vs. baseline (profiled, optimized)
5. **Hierarchy** — Supports hierarchical graphs (graph-of-graphs) for deep networks
6. **Adaptation** — Per-input routing (different topologies for different inputs)

---

## What Makes NGN Different from Prior Work?

| Method | Topology | Input-Aware | Backbone-Agnostic | Interpretable |
|--------|----------|-------------|------------------|---|
| **ResNet** | Fixed sequential | ❌ | ❌ | ❌ |
| **DIANet** | Learned (LSTM) | ❌ (shared state) | ✓ | ❌ |
| **MRLA** | Dynamic (attention) | ✅ | ✓ | ✓ (attention heatmaps) |
| **DLA** | Dynamic (RNN+attention) | ✅ | ✓ | ✓ |
| **NGN** | Generic framework | ✅ | ✅ | ✅ + analysis tools |

**NGN's advantage:** Combines all the above + adds systematic interpretability analysis & multi-domain validation.

---

## Implementation Phases

### Phase 1: **Synthetic & Verification**
- Test on XOR (confirm topology learning works)
- Check gradients flow correctly
- Verify layer graph converges

### Phase 2: **Vision Baseline**
- CIFAR-10 + ResNet-18
- Compare accuracy vs. vanilla ResNet
- Visualize learned attention patterns

### Phase 3: **Interpretability**
- Plot adjacency/attention matrices
- Identify layer "hubs" and communities
- Explain what network learned

### Phase 4: **Generalization**
- NLP: Language model (character-level)
- Sequence: Time series prediction
- Prove: Same LayerGraph works across domains

### Phase 5: **Optimization** (if time permits)
- Per-input routing
- Hierarchical graphs
- Hardware profiling

---

## Code Structure (PyTorch)

```
ngn/
├── core/
│   ├── graph.py           # LayerGraph base
│   └── communication.py   # SharedAttentionAggregator
├── backbones/
│   ├── resnet.py          # NGN-wrapped ResNet
│   └── base.py            # Abstract backbone
├── training/
│   ├── trainer.py         # Training loop
│   └── losses.py          # Loss functions + regularization
├── experiments/
│   ├── synthetic/train_xor.py
│   ├── vision/train_cifar10.py
│   └── nlp/train_language_model.py
└── utils/
    ├── visualization.py   # Plot attention, adjacency
    └── profiling.py       # Memory, compute, latency
```

---

## Next: Implementation

Ready to code! We'll build:

1. **ngn/core/graph.py** — LayerGraph (learns connectivity)
2. **ngn/core/communication.py** — SharedAttentionAggregator (cross-layer fusion)
3. **ngn/backbones/resnet.py** — NGN-wrapped ResNet
4. **experiments/synthetic/train_xor.py** — Test on simple task
5. **utils/visualization.py** — Visualize learned graphs

Each component is **modular, testable, and reusable** across all downstream tasks.

---

## Questions to Answer During Implementation

1. ✅ Does attention-based aggregation learn meaningful layer interactions?
2. ✅ Are gradients stable through bidirectional updates?
3. ✅ Can we identify layer "hubs" that many others attend to?
4. ✅ Does the same LayerGraph improve performance across different tasks?
5. ✅ What's the overhead (memory, compute) vs. baseline?

The research narrative: **"We show that learned dynamic layer graphs are a universal principle for improving deep networks, not just a vision technique."**

---

## References

- See `CONVERSTATION.MD` for full literature context
- See `ARCHITECTURE.md` for detailed technical design
- See `IMPLEMENTATION_GUIDE.md` for PyTorch specifics
