# NGN (Neural Graph Network) — Complete Architecture Design

## High-Level Overview

NGN is a **meta-architecture** that wraps any backbone (CNN, Transformer, RNN) and replaces fixed sequential layer connections with a **learnable, dynamic layer graph**.

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT (x)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           BACKBONE LAYERS (Layer 1 → Layer N)              │
│  ┌───────────┐  ┌───────────┐      ┌───────────┐          │
│  │ Layer 1   │  │ Layer 2   │  … │ Layer N   │          │
│  │(Features) │  │(Features) │      │(Features) │          │
│  └─────┬─────┘  └─────┬─────┘      └─────┬─────┘          │
│        │              │                    │               │
│        └──────────────┼────────────────────┘               │
│                       ▼                                     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            LAYER GRAPH (Core of NGN)                        │
│                                                             │
│  Learns: "Which layers should communicate?"               │
│  - Adjacency Matrix (L × L): Connectivity               │
│  - Attention Weights: Input-dependent routing            │
│  - Shared RNN/Attention Unit: Handles cross-layer ops   │
│                                                             │
│  ┌─────────────────────────────────────────┐              │
│  │  Feature Aggregation & Routing          │              │
│  │  ┌────────────────────────────────────┐ │              │
│  │  │ Shared Communication Module        │ │              │
│  │  │ (RNN or Multi-Head Attention)     │ │              │
│  │  └────────────────────────────────────┘ │              │
│  │           ↓↑ (bidirectional)            │              │
│  │  ┌────────────────────────────────────┐ │              │
│  │  │ Learned Adjacency + Gating        │ │              │
│  │  │ - Which edges to activate?        │ │              │
│  │  │ - How much to communicate?        │ │              │
│  │  └────────────────────────────────────┘ │              │
│  └─────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          REFINED LAYER OUTPUTS (f'₁, f'₂, ..., f'ₙ)        │
│  Each layer's features enriched by cross-layer context     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           OUTPUT HEAD (Classification/Regression)           │
│           Aggregate refined features → Prediction          │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Architecture

### 1. **Input & Backbone Extraction**

```python
# Pseudo-code architecture
input: x ∈ ℝ^(B × C_in × H × W)  # Batch size B, channels C_in

# Backbone (CNN/Transformer/RNN) produces layer outputs
layer_outputs = [f_1, f_2, ..., f_L]  # L layers
where f_i ∈ ℝ^(B × C_i × H_i × W_i)  # Features may change size per layer
```

**Key Design Decision:**
- Extract features at **intermediate layers**, not just final layer
- Allows early layers to influence later layers directly
- Enables **hierarchical feature fusion**

---

### 2. **LayerGraph (Core NGN Component)**

The LayerGraph learns **"who talks to whom"** among layers.

#### A. **Adjacency/Connectivity Matrix**

```
A ∈ ℝ^(L × L)  # Learnable adjacency matrix
```

**Option 1: Static Learned Topology (Simpler)**
- `A` is learned once during training
- All inputs follow the same layer connectivity
- Similar to: DIANet (fixed LSTM context)

**Option 2: Dynamic (Input-Dependent) Topology**
- `A(x)` is recomputed per input
- Different inputs may route through different paths
- Similar to: MRLA, DLA (attention weights vary per input)

For **initial implementation**, start with **Option 1** (static), then extend to Option 2.

#### B. **Routing Mechanism**

```
# For each layer i, determine which previous layers contribute:
incoming_edges = {j : A[j, i] > threshold}

# Aggregate features from incoming layers
aggregated_features = Σ(A[j, i] * f_j)  for j in incoming_edges
```

**Key Property:**
- Not all pairs communicate (sparse graph)
- Learned sparsity: Network learns which connections matter

---

### 3. **Shared Communication Module**

All cross-layer communication goes through **one shared module** (critical for training stability).

#### Option A: **Shared RNN (like DIANet)**

```python
class SharedRNNAggregator(nn.Module):
    def __init__(self, feature_dim):
        self.rnn = nn.LSTMCell(feature_dim, feature_dim)
        self.hidden_state = None  # Carries context across layers
    
    def forward(self, layer_features, layer_idx):
        # layer_features: [B, C, H, W] or flattened
        
        # Update hidden state with current layer info
        h_new, c_new = self.rnn(layer_features, (self.hidden_state, self.cell_state))
        
        # Use hidden state as "context" for all layers
        context = h_new
        
        # Influence current layer
        refined_features = layer_features + context  # Residual
        
        return refined_features
```

**Advantages:**
- Shared params across all layers → smaller model
- Sequential dependency captures layer interactions
- Similar to DIANet (AAAI 2020)

#### Option B: **Multi-Head Attention (like MRLA)**

```python
class SharedAttentionAggregator(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
    
    def forward(self, layer_features, all_layer_features, layer_idx):
        # layer_features: query (current layer)
        # all_layer_features: keys & values (all previous layers)
        
        # Attend to all previous layers
        refined_features, attention_weights = self.attention(
            query=layer_features,
            key=all_layer_features,
            value=all_layer_features
        )
        
        # Residual connection for stability
        refined_features = layer_features + refined_features
        
        return refined_features, attention_weights
```

**Advantages:**
- Input-dependent routing (attention = soft routing)
- Interpretable (can visualize which layers attend to which)
- Similar to MRLA (ICLR 2023)

**Recommendation for initial NGN:** Start with **Attention** (Option B) because:
- More interpretable (we can visualize attention matrices)
- Input-dependent by design (better learning)
- Slightly more complex → more research novelty

---

### 4. **Dynamic Routing (Optional, Advanced)**

Add a **gating/routing network** that learns **per-input topology adaptation**.

```python
class RoutingGate(nn.Module):
    def __init__(self, feature_dim, num_layers):
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Spatial pooling
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_layers)  # Score each layer's importance
        )
    
    def forward(self, layer_features):
        # Score how much this input needs each layer
        importance_scores = self.router(layer_features)  # [B, L]
        
        # Soft gating
        gate_weights = torch.softmax(importance_scores, dim=1)  # [B, L]
        
        return gate_weights
```

**Use Case:**
- Simple inputs: gate down some layers (skip computation)
- Complex inputs: use all layers
- Reduces compute for easy cases

---

## Complete Forward Pass

Here's how data flows through NGN:

```python
def ngn_forward(input_x):
    """
    Complete forward pass of NGN
    """
    
    # Step 1: Extract layer-wise features from backbone
    layer_outputs = backbone(input_x)  # [f_1, f_2, ..., f_L]
    
    # Step 2: Create layer graph
    refined_outputs = [None] * L
    
    # Step 3: For each layer (in sequence or in parallel)
    for i in range(L):
        
        # Step 3a: Collect features from incoming layers
        incoming_layers = []
        for j in range(L):
            if adjacency_matrix[j, i] > threshold:
                incoming_layers.append(layer_outputs[j])
        
        # Step 3b: Apply shared communication module
        context = shared_aggregator(
            query=layer_outputs[i],
            keys_values=incoming_layers,
            layer_idx=i
        )
        
        # Step 3c: Refine features with residual connection
        refined_outputs[i] = layer_outputs[i] + context
    
    # Step 4: Aggregate refined outputs for prediction
    final_features = aggregate(refined_outputs)  # e.g., avg pool
    
    # Step 5: Classification head
    logits = classifier(final_features)
    
    return logits, {
        'layer_outputs': layer_outputs,
        'refined_outputs': refined_outputs,
        'attention_weights': attention_weights  # for visualization
    }
```

---

## Key Architectural Decisions & Trade-offs

| Decision | Choice | Trade-off |
|----------|--------|-----------|
| **Aggregation** | Attention | More interpretable, but higher compute than RNN |
| **Topology** | Static learned (Phase 1) → Dynamic (Phase 2) | Simple first, then complex |
| **Residual Updates** | Yes (always) | Stability & gradient flow over pure communication |
| **Shared Params** | Yes (one aggregator) | Fewer params, faster training, more stable |
| **Graph Sparsity** | Learned (some edges = 0) | Reduces compute & reveals structure |
| **Bidirectional** | Unidirectional initially | Later layers can attend to earlier, but not vice versa |

---

## Stability Mechanisms (Critical!)

Since we have feedback loops (later layers influence earlier), we need **safeguards**:

### 1. **Residual Connections**
```python
refined_features = original_features + small_weight * cross_layer_input
```
Ensures gradient can flow even if aggregator fails.

### 2. **Layer Normalization**
```python
cross_layer_features = LayerNorm(cross_layer_features)
refined_features = original_features + aggregator(cross_layer_features)
```
Prevents activation explosion.

### 3. **Gradient Clipping**
```python
for param in model.parameters():
    torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
```
Stops gradient explosion in feedback paths.

### 4. **Shared RNN Hidden State**
```python
# Bounded hidden state ensures stability
h_new = tanh(linear(h_old + x))  # tanh ∈ [-1, 1]
```

---

## Loss Function & Training

```python
total_loss = classification_loss + regularization_terms

# Classification loss
classification_loss = CrossEntropyLoss(logits, labels)

# Regularization 1: Encourage sparse graphs
sparsity_loss = L1(adjacency_matrix)  # Sparsity

# Regularization 2: Prevent collapsed attention (all mass on one layer)
entropy_loss = -Entropy(attention_weights)  # Encourage diversity

# Regularization 3: Stability monitoring (optional)
gradient_penalty = mean(gradient_norms_per_layer)

total_loss = classification_loss + λ₁*sparsity_loss + λ₂*entropy_loss + λ₃*gradient_penalty
```

---

## Best Python Libraries for This

| Component | Library | Why |
|-----------|---------|-----|
| **Core framework** | **PyTorch** | Industry standard; flexible; great debugging |
| **Backbones** | **torchvision** | Pre-trained ResNets, easy to extract intermediate layers |
| **Graph operations** | **torch_geometric** | Optional; for advanced graph algorithms |
| **Visualization** | **matplotlib, networkx** | Plot adjacency matrices, network graphs |
| **Logging** | **tensorboard, wandb** | Track attention weights, gradients, topology per epoch |
| **Data** | **torchvision.datasets, HuggingFace** | Vision & NLP datasets |

---

## NGN Implementation Phases

### Phase 1: **Backbone-Agnostic Foundation** ✓
- Static learned adjacency matrix
- Shared attention aggregator
- Residual connections + layer norm
- Test on synthetic data

### Phase 2: **Vision Validation**
- CIFAR-10 + ResNet-18 baseline
- Log attention weights per epoch
- Compare accuracy vs. vanilla ResNet

### Phase 3: **Interpretability & Analysis**
- Visualization tools (adjacency heatmaps)
- Identify layer communities/hubs
- Prove: "Network learns meaningful structure"

### Phase 4: **Generalization to Other Domains**
- NLP: Character-level language model
- Sequence: Time series prediction
- Show: Same LayerGraph works across domains

### Phase 5: **Optimizations (if time permits)**
- Per-input routing (dynamic topology)
- Hierarchical graphs (graph-of-graphs)
- Hardware profiling & efficiency

---

## Summary: NGN as a Single Diagram

```
INPUT
  ↓
BACKBONE (L layers) → [f₁, f₂, ..., fₗ]
  ↓
LAYER GRAPH:
  • Learned Adjacency Matrix (L × L)
  • Shared Attention Aggregator
  • Residual Updates + Layer Norm
  ↓
REFINED OUTPUTS → [f'₁, f'₂, ..., f'ₗ]
  ↓
AGGREGATE & CLASSIFY
  ↓
OUTPUT (logits, loss, metrics)
```

---

## Next Steps

1. **Implement in PyTorch:**
   - `ngn/core/graph.py` → LayerGraph class
   - `ngn/core/attention.py` → SharedAttentionAggregator class
   - `ngn/backbones/wrapped_resnet.py` → NGN-wrapped ResNet

2. **Test on synthetic data** (XOR task) before CIFAR-10

3. **Visualize learned graphs** to verify correctness

4. **Benchmark vs. baselines** (vanilla ResNet, DIANet, MRLA if available)

Ready to start coding? I'll generate the PyTorch implementation!
