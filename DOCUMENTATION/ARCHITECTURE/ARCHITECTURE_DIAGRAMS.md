# NGN Architecture Diagrams

## 1. High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT (Image/Text/Audio)                  │
└─────────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              BACKBONE (ResNet/Transformer/RNN)                   │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Layer1  │→ │  Layer2  │→ │  Layer3  │→ │  Layer4  │        │
│  │[64,32,32]│  │[128,16,16]  │[256,8,8]│  │[512,4,4]│        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       └─────────────┼─────────────┼─────────────┘               │
│                     │(extract)    │                              │
└─────────────────────┼─────────────┼──────────────────────────────┘
                      │             │
        ┌─────────────▼─────────────▼──────────────┐
        │   Layer Outputs: [f1, f2, f3, f4]       │
        │   Raw features from backbone            │
        └─────────────┬─────────────┬──────────────┘
                      │             │
                      ▼             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     LAYER GRAPH (NGN Core)                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Question: Which layers should communicate?                │ │
│  │ Answer: Learned via multi-head attention                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Layer 1 ←─→ Layer 2 ←─→ Layer 3 ←─→ Layer 4                 │
│    ↓         ↓         ↓         ↓                              │
│    └─────────┴─────────┴─────────┘ (bidirectional)            │
│                   ↓                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Shared Attention Aggregator (one module for ALL pairs)   │ │
│  │                                                             │ │
│  │  For each layer i:                                        │ │
│  │    refined[i] = layer[i] + attention(                     │ │
│  │                  query=layer[i],                          │ │
│  │                  keys_values=[all layers]                 │ │
│  │                )                                           │ │
│  │                                                             │ │
│  │  Key design: Residual connection + LayerNorm             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Refined Layer Outputs: [f'1, f'2, f'3, f'4]                 │
│  (Each layer enriched with cross-layer context)               │
│                                                                  │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              AGGREGATION HEAD (per-task)                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Global Average Pool: combine all refined layer outputs   │  │
│  │ final_features = mean([f'1, f'2, f'3, f'4])            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└───────────────────┬──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│           TASK-SPECIFIC HEAD (Classification/Regression)        │
│                                                                  │
│  Linear(512 → num_classes)                                      │
│  → Softmax → Predictions                                        │
│                                                                  │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
            ┌───────────────┐
            │  OUTPUT       │
            │  (logits)     │
            └───────────────┘
```

---

## 2. Shared Attention Aggregator (Detailed)

```
                    Layer Features
                 [f_1, f_2, f_3, f_4]
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
      Layer 1       Layer 2       Layer 3      ...
        [B,64]       [B,128]       [B,256]
           │             │             │
           │             │             │
           └─────────────┼─────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │  Shared Attention Module   │
            │  (One for ALL layers)      │
            │                            │
            │  Forward Pass:             │
            │  • Layer i = Query         │
            │  • All layers = Keys+Values│
            │  • Multi-head attention    │
            │  • Output = context        │
            └────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │  Refinement (per layer i): │
            │                            │
            │  refined[i] =              │
            │    LayerNorm(              │
            │      layer[i] +            │
            │      attention_output      │
            │    )                       │
            │                            │
            │  (Residual connection)    │
            └────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │   Refined Outputs          │
            │  [f'_1, f'_2, f'_3, f'_4] │
            │                            │
            │  Each layer now knows      │
            │  about ALL other layers    │
            └────────────────────────────┘
```

---

## 3. Attention Mechanism (Multi-Head)

```
For each layer i attending to all layers:

                Query (Layer i)
                      │
                ┌─────┴─────┐
                │           │
                ▼           ▼
         Linear(Wq)   Linear(Wk)
                │           │
                ▼           ▼
            Q_i         K_all
                │           │
                └─────┬─────┘
                      │
            Attention Weights = softmax(Q_i · K_all^T / √d)
                      │
            ┌─────────┴─────────┐
            │                   │
      Which layers?        How much?
            │                   │
      Layer 1: 15%        Layer 2: 60%
      Layer 2: 60%        Layer 3: 20%
      Layer 3: 20%        Layer 4: 5%
      Layer 4: 5%
            │
            ▼
      Weighted Sum: 0.15*f1 + 0.60*f2 + 0.20*f3 + 0.05*f4
            │
            ▼
      Add to original Layer i (Residual)
            │
            ▼
      Refined Layer i
```

---

## 4. Training Data Flow (with Gradients)

```
                        Loss
                         │
                    ┌────┴────┐
                    │          │
            Classification   Regularization
            Loss              Loss
              │                │
              ├───────┬────────┤
                      │
                      ▼
                  Backprop
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
    Gradient through       Gradient through
    Attention Weights      Residual Connections
         │                         │
         ├─────────────┬───────────┤
                       │
                       ▼
                ┌─────────────────┐
                │ Gradient Clip   │
                │ (max_norm=1.0)  │
                └────────┬────────┘
                         │
                         ▼
                    Parameter Update
                 (optimizer.step())
```

---

## 5. Attention Weight Evolution (Training Progress)

```
EPOCH 1 (Random Initialization):
  Layer 1 → 2: 0.25    Layer 1 → 3: 0.25    Layer 1 → 4: 0.25
  Layer 2 → 1: 0.25    Layer 2 → 3: 0.25    Layer 2 → 4: 0.25
  ...
  (Almost uniform, random)

EPOCH 50 (Learning Structure):
  Layer 1 → 2: 0.60    Layer 1 → 3: 0.25    Layer 1 → 4: 0.05
  Layer 2 → 1: 0.10    Layer 2 → 3: 0.70    Layer 2 → 4: 0.10
  ...
  (Some structure emerges)

EPOCH 200 (Converged):
  Layer 1 → 2: 0.85    Layer 1 → 3: 0.10    Layer 1 → 4: 0.01
  Layer 2 → 1: 0.02    Layer 2 → 3: 0.92    Layer 2 → 4: 0.05
  Layer 3 → 1: 0.05    Layer 3 → 2: 0.05    Layer 3 → 4: 0.80
  Layer 4 → ALL: 0.50  (Hub: broadcast to all)
  
  Pattern:
  - Layer 1 → 2 (strong)
  - Layer 2 → 3 (strong, feedforward pattern emerges)
  - Layer 3 → 4 (strong)
  - Layer 4 → all others (hub, bidirectional feedback)
```

---

## 6. Comparison: NGN vs. ResNet Forward Pass

```
RESNET (Standard):

Input → Conv → Layer1 → Layer2 → Layer3 → Layer4 → FC → Output
                 │           │
                 └─ skip ────┘ (fixed, not learnable)

Problem: Layer1 output frozen once it leaves Layer1


NGN (Neural Graph Network):

Input → Conv → Layer1 ─┐
                ├─ Layer2 ─┐
                ├─ Layer3 ─┤ NGN LayerGraph  ┌─ Refined Layer1
                └─ Layer4 ─┘ (learns which    ├─ Refined Layer2
                              communicate)    ├─ Refined Layer3
                                              └─ Refined Layer4
                              ↓
                           FC → Output

Benefit:
  • Layer1 receives context from ALL other layers
  • Connections learned, not fixed
  • Attention weights different per input (adaptive routing)
```

---

## 7. Stability Mechanisms

```
┌────────────────────────────────────────────────────────────────┐
│              Risk: Feedback Loops → Instability                │
│                                                                │
│  Problem: If Layer 4 updates Layer 1, and then Layer 1        │
│           updates Layer 4 again, gradients can explode/vanish  │
└────────────────────────────────────────────────────────────────┘

Solution 1: Residual Connections
  refined = original + α * new_signal  (α is small, e.g., 0.1)
            ↑
            Ensures gradient can flow even if new_signal fails

Solution 2: Layer Normalization
  norm_out = LayerNorm(output)
  Keeps activation magnitudes bounded (±1 range typically)

Solution 3: Gradient Clipping
  if ||gradient|| > 1.0:
      gradient = gradient / ||gradient||
  Prevents explosion

Solution 4: Shared Parameters
  One attention module for ALL layers
  Fewer parameters → more stable learning
  (vs. independent modules per layer)

Result:
  Training curves stay smooth, gradients don't explode
  ✓ Can train for 200+ epochs without divergence
```

---

## 8. Phase 1 Implementation: Synthetic XOR Task

```
XOR Problem (2D → 1D):

Input:           Network:          Output:
(0,0) → 0     ┌─────────────┐     Prediction
(0,1) → 1     │  Backbone   │     vs.
(1,0) → 1  → │  (2 layers) │  →  Target
(1,1) → 0     │             │     Loss = MSE
              │ NGN Layer   │
              │ Graph       │
              └─────────────┘

Simple enough to verify:
  ✓ Topology converges (not random)
  ✓ Gradients stable
  ✓ Network learns function
  ✓ Fast training (seconds, not hours)

BUT: Complex enough to test
  ✓ Bidirectional updates
  ✓ Attention mechanisms
  ✓ Residual connections
  ✓ All core mechanics

Success Criteria:
  • Accuracy → 95%+
  • Attention weights show clear pattern (not uniform)
  • No gradient explosion/vanishing
  • Training loss monotonically decreases
```

---

## Summary Diagram: NGN in One Picture

```
                    ┏━━━━━━━━━━━━━┓
                    ┃   INPUT     ┃
                    ┗━━━━┳━━━━━━━┛
                         │
                    ┌────▼────┐
                    │ BACKBONE │ ← ResNet/Transformer/RNN
                    │ (4 layers)│   (or any architecture)
                    └─┬──┬──┬──┬┘
                      │  │  │  │
        ┌─────────────┴──┴──┴──┴─────────────┐
        │   [f1, f2, f3, f4]                 │
        │   (raw layer outputs)              │
        └─────┬────────────────────────────┬─┘
              │                            │
        ╔═════▼════════════════════════════▼═════╗
        ║     NGN: Shared Attention Layer Graph  ║
        ║     (learns which layers communicate) ║
        ║                                        ║
        ║  Bidirectional Attention Connections  ║
        ║  + Residual Updates                   ║
        ║  + Layer Normalization                ║
        ║  = STABILITY ✓                        ║
        ╚═════┬────────────────────────────┬════╝
              │                            │
        ┌─────▼────────────────────────────▼─┐
        │ [f'1, f'2, f'3, f'4]               │
        │ (refined, context-aware features) │
        └─────┬────────────────────────────┬─┘
              │                            │
              └─────────────┬──────────────┘
                            │
                        ┌───▼───┐
                        │AGGREGATE
                        │ (avg pool)
                        └───┬───┘
                            │
                        ┌───▼──────┐
                        │  FC Head  │
                        │ (classify)│
                        └───┬──────┘
                            │
                      ┏━━━━━▼━━━━━┓
                      ┃  PREDICTION┃
                      ┗━━━━━━━━━━━┛
```

---

All diagrams are **conceptual**. Actual implementation in PyTorch will follow this structure exactly.
