"""
NGN Experiments - Synthetic XOR Task

This is the first experiment: Test NGN on a simple synthetic task
to verify the implementation works and can learn meaningful layer connections.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional

from ngn_pytorch.core.graph import LayerGraph
from ngn_pytorch.core.communication import SharedAttentionAggregator
from ngn_pytorch.training.trainer import NGNTrainer
from ngn_pytorch.utils.visualization import NGNVisualizer

logger = logging.getLogger(__name__)


class XORDataset(Dataset):
    """Simple XOR dataset for testing NGN."""

    def __init__(self, num_samples: int = 1000):
        # Generate XOR data: (x1, x2) -> x1 XOR x2
        x1 = np.random.randint(0, 2, num_samples)
        x2 = np.random.randint(0, 2, num_samples)
        y = x1 ^ x2  # XOR operation

        # Convert to one-hot encoding for richer representation
        self.x = torch.zeros(num_samples, 4)  # 2 bits * 2 inputs = 4 features
        self.y = torch.zeros(num_samples, 2)  # 2 classes

        for i in range(num_samples):
            # Encode inputs as one-hot
            self.x[i, x1[i]] = 1
            self.x[i, 2 + x2[i]] = 1
            # Encode output
            self.y[i, y[i]] = 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleMLP(nn.Module):
    """
    Simple MLP backbone for XOR task.

    This creates a 3-layer network that NGN can learn to connect.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ])

        self.layer_outputs = []

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward pass, collecting intermediate outputs."""
        self.layer_outputs = []

        # Layer 1
        out = F.relu(self.layers[0](x))
        self.layer_outputs.append(out)

        # Layer 2
        out = F.relu(self.layers[1](out))
        self.layer_outputs.append(out)

        # Layer 3 (output layer, no activation)
        out = self.layers[2](out)
        self.layer_outputs.append(out)

        return self.layer_outputs


class NGNXORModel(nn.Module):
    """
    NGN model for XOR task.

    Combines MLP backbone with NGN layer graph.
    """

    def __init__(self, layer_graph: Optional[LayerGraph] = None):
        super().__init__()

        self.backbone = SimpleMLP()
        self.layer_graph = layer_graph

        if self.layer_graph is None:
            # Default: identity (no communication)
            self.layer_graph = IdentityLayerGraph(
                num_layers=3,
                feature_dims=[32, 32, 2]
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with NGN communication."""
        # Get backbone layer outputs
        layer_outputs = self.backbone(x)

        # Apply NGN layer graph
        refined_outputs, debug_info = self.layer_graph(layer_outputs)

        # Use final refined output as prediction
        logits = refined_outputs[-1]

        return logits, debug_info


class SimpleAttentionAggregator(LayerGraph):
    """
    Simplified attention aggregator for 2D features (XOR task).
    """

    def __init__(self, num_layers: int, feature_dims: List[int], embed_dim: int = 64):
        super().__init__(num_layers, feature_dims, use_residual=True, use_layer_norm=True)

        # Simple linear projections for 2D features
        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim) for dim in feature_dims
        ])
        
        # Output projections: from embed_dim back to feature_dim
        self.output_projections = nn.ModuleList([
            nn.Linear(embed_dim, dim) for dim in feature_dims
        ])

        # Shared attention mechanism
        self.shared_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, layer_outputs: List[Tensor]) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """Apply attention across layers for 2D features."""
        # Project all layer outputs
        projected_outputs = []
        for i, feat in enumerate(layer_outputs):
            # feat is (batch, feature_dim)
            proj_feat = self.projections[i](feat)  # (batch, embed_dim)
            projected_outputs.append(proj_feat)

        # Stack for attention: (batch, num_layers, embed_dim)
        stacked_features = torch.stack(projected_outputs, dim=1)

        refined_outputs = []
        attention_weights_list = []

        for i in range(self.num_layers):
            # Query: current layer
            query = projected_outputs[i].unsqueeze(1)  # (batch, 1, embed_dim)

            # Keys/Values: all layers
            keys_values = stacked_features  # (batch, num_layers, embed_dim)

            # Apply attention
            attn_output, attn_weights = self.shared_attention(
                query=query,
                key=keys_values,
                value=keys_values
            )

            # Remove singleton dimension
            attn_output = attn_output.squeeze(1)  # (batch, embed_dim)

            # Project back to original dimension
            update = self.output_projections[i](attn_output)

            # Apply residual connection
            refined = self.apply_residual_connection(
                original=layer_outputs[i],
                update=update,
                layer_idx=i
            )

            refined_outputs.append(refined)
            attention_weights_list.append(attn_weights)

        debug_info = {
            'attention_weights': attention_weights_list,
            'adjacency_matrix': self.get_adjacency_matrix()
        }

        return refined_outputs, debug_info


class IdentityLayerGraph(LayerGraph):
    """Identity layer graph for baseline comparison."""

    def __init__(self, num_layers: int, feature_dims: List[int]):
        super().__init__(num_layers, feature_dims, use_residual=False, use_layer_norm=False)

    def forward(self, layer_outputs: List[Tensor]) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'communication_type': 'identity'
        }
        return layer_outputs, debug_info


def create_ngn_xor_model() -> NGNXORModel:
    """Create NGN model with SimpleAttentionAggregator."""
    layer_graph = SimpleAttentionAggregator(
        num_layers=3,
        feature_dims=[32, 32, 2],
        embed_dim=32
    )

    model = NGNXORModel(layer_graph=layer_graph)
    return model


def train_xor_experiment(
    model_name: str = 'ngn_xor',
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    log_dir: str = 'logs/xor',
    ablation: str = 'dynamic',
    num_heads: int = 4,
    use_residual: bool = True
):
    """
    Run XOR training experiment.

    Args:
        model_name: Name of the model ('ngn_xor' or 'baseline')
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_dir: Logging directory
    """
    # Create datasets
    train_dataset = XORDataset(num_samples=1000)
    val_dataset = XORDataset(num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    if ablation == 'static':
        from ngn_pytorch.core.graph import LayerGraph
        layer_graph = LayerGraph(
            num_layers=3,
            feature_dims=[32, 32, 2],
            use_residual=use_residual,
            use_layer_norm=True
        )
        model = NGNXORModel(layer_graph=layer_graph)
        logger.info("Created static LayerGraph XOR model")
    elif ablation == 'dynamic':
        from ngn_pytorch.core.communication import SharedAttentionAggregator
        layer_graph = SharedAttentionAggregator(
            num_layers=3,
            feature_dims=[32, 32, 2],
            num_heads=num_heads,
            dropout=0.1
        )
        model = NGNXORModel(layer_graph=layer_graph)
        logger.info(f"Created NGN-XOR model (dynamic, heads={num_heads}, residual={use_residual})")
    elif ablation == 'residual_off':
        from ngn_pytorch.core.communication import SharedAttentionAggregator
        layer_graph = SharedAttentionAggregator(
            num_layers=3,
            feature_dims=[32, 32, 2],
            num_heads=num_heads,
            dropout=0.1
        )
        if hasattr(layer_graph, 'use_residual'):
            layer_graph.use_residual = False
        model = NGNXORModel(layer_graph=layer_graph)
        logger.info(f"Created NGN-XOR model (dynamic, heads={num_heads}, residual=OFF)")
    else:
        model = NGNXORModel()  # Baseline with identity layer graph
        logger.info("Created baseline XOR model")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = NGNTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        log_dir=log_dir
    )

    # Visualizer
    visualizer = NGNVisualizer(output_dir='visualizations/xor')

    logger.info(f"Starting {model_name} training on {device}")

    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_metrics = trainer.validate(val_loader, epoch)

        # Log
        logger.info(
            f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}"
        )

        # Visualize every 10 epochs
        if epoch % 10 == 0 and hasattr(model, 'layer_graph'):
            # Get a batch for visualization
            x, y = next(iter(val_loader))
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                _, debug_info = model(x)

            if 'attention_weights' in debug_info:
                visualizer.plot_attention_heatmap(
                    debug_info['attention_weights'],
                    epoch=epoch,
                    save=True
                )

            if 'adjacency_matrix' in debug_info:
                visualizer.plot_adjacency_matrix(
                    debug_info['adjacency_matrix'],
                    epoch=epoch,
                    save=True
                )

    # Final evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            targets = y.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    final_accuracy = correct / total
    logger.info(f"Final validation accuracy: {final_accuracy:.4f}")

    # Save model
    checkpoint_path = f'checkpoints/{model_name}_final.pth'
    trainer.save_checkpoint(checkpoint_path, num_epochs, {
        'final_accuracy': final_accuracy,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    })

    return {
        'model': model,
        'final_accuracy': final_accuracy,
        'training_history': trainer.metrics_history
    }


def run_xor_comparison():
    """Compare NGN vs baseline on XOR task."""
    logger.info("Running XOR comparison experiment")

    # Train baseline
    logger.info("Training baseline model...")
    baseline_results = train_xor_experiment(
        model_name='baseline_xor',
        num_epochs=30,
        log_dir='logs/xor_baseline'
    )

    # Train NGN
    logger.info("Training NGN model...")
    ngn_results = train_xor_experiment(
        model_name='ngn_xor',
        num_epochs=30,
        log_dir='logs/xor_ngn'
    )

    # Compare results
    baseline_acc = baseline_results['final_accuracy']
    ngn_acc = ngn_results['final_accuracy']

    logger.info(f"Baseline accuracy: {baseline_acc:.4f}")
    logger.info(f"NGN accuracy: {ngn_acc:.4f}")
    logger.info(f"Improvement: {(ngn_acc - baseline_acc) * 100:.2f}%")

    return {
        'baseline_accuracy': baseline_acc,
        'ngn_accuracy': ngn_acc,
        'improvement': ngn_acc - baseline_acc
    }


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    Path('logs').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="XOR NGN Ablation Experiments")
    parser.add_argument('--ablation', type=str, default='dynamic', choices=['dynamic', 'static', 'residual_off', 'baseline'], help='Ablation type')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (if dynamic)')
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    use_residual = not args.no_residual
    results = train_xor_experiment(
        model_name='ngn_xor' if args.ablation != 'baseline' else 'baseline_xor',
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        ablation=args.ablation,
        num_heads=args.num_heads,
        use_residual=use_residual
    )
    print("XOR Experiment Results:")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")