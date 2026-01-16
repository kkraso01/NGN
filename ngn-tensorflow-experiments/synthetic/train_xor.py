"""
NGN Experiments - Synthetic XOR Task

This is the first experiment: Test NGN on a simple synthetic task
to verify the implementation works and can learn meaningful layer connections.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional

from ngn_tensorflow.core.graph import LayerGraph
from ngn_tensorflow.core.communication import SharedAttentionAggregator
from ngn_tensorflow.training.trainer import NGNTrainer
from ngn_tensorflow.utils.visualization import NGNVisualizer

logger = logging.getLogger(__name__)


def create_xor_dataset(num_samples: int = 1000) -> tf.data.Dataset:
    """Create XOR dataset for testing NGN."""
    # Generate XOR data: (x1, x2) -> x1 XOR x2
    x1 = np.random.randint(0, 2, num_samples)
    x2 = np.random.randint(0, 2, num_samples)
    y = x1 ^ x2  # XOR operation

    # Convert to one-hot encoding for richer representation
    x = np.zeros((num_samples, 4))  # 2 bits * 2 inputs = 4 features
    targets = np.zeros((num_samples,))  # Integer labels for sparse categorical

    for i in range(num_samples):
        # Encode inputs as one-hot
        x[i, x1[i]] = 1
        x[i, 2 + x2[i]] = 1
        # Encode output as integer
        targets[i] = y[i]

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), targets.astype(np.int32)))
    return dataset


class SimpleMLP(keras.Model):
    """
    Simple MLP backbone for XOR task.

    This creates a 3-layer network that NGN can learn to connect.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 2, **kwargs):
        super().__init__(**kwargs)

        self.layers_list = [
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(output_dim)
        ]

    def call(self, x: tf.Tensor) -> List[tf.Tensor]:
        """Forward pass, collecting intermediate outputs."""
        layer_outputs = []

        # Layer 1
        out = self.layers_list[0](x)
        layer_outputs.append(out)

        # Layer 2
        out = self.layers_list[1](out)
        layer_outputs.append(out)

        # Layer 3 (output layer, no activation)
        out = self.layers_list[2](out)
        layer_outputs.append(out)

        return layer_outputs


class NGNXORModel(keras.Model):
    """
    NGN model for XOR task.

    Combines MLP backbone with NGN layer graph.
    """

    def __init__(self, layer_graph: Optional[LayerGraph] = None, **kwargs):
        super().__init__(**kwargs)

        self.backbone = SimpleMLP()
        self.layer_graph = layer_graph

        if self.layer_graph is None:
            # Default: identity (no communication)
            self.layer_graph = IdentityLayerGraph(
                num_layers=3,
                feature_dims=[32, 32, 2]
            )

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Forward pass with NGN communication."""
        # Get backbone layer outputs
        layer_outputs = self.backbone(x)

        # Apply NGN layer graph
        refined_outputs, debug_info = self.layer_graph(layer_outputs)

        # Use final refined output as prediction
        logits = refined_outputs[-1]

        return logits, debug_info


class IdentityLayerGraph(LayerGraph):
    """Identity layer graph for baseline comparison."""

    def __init__(self, num_layers: int, feature_dims: List[int]):
        super().__init__(num_layers, feature_dims, use_residual=False, use_layer_norm=False)

    def call(self, layer_outputs: List[tf.Tensor]) -> Tuple[List[tf.Tensor], Dict[str, tf.Tensor]]:
        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'communication_type': 'identity'
        }
        return layer_outputs, debug_info


def create_ngn_xor_model() -> NGNXORModel:
    """Create NGN model with SharedAttentionAggregator."""
    layer_graph = SharedAttentionAggregator(
        num_layers=3,
        feature_dims=[32, 32, 2],
        num_heads=4,
        dropout=0.1
    )

    model = NGNXORModel(layer_graph=layer_graph)
    return model


def train_xor_experiment(
    model_name: str = 'ngn_xor',
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    log_dir: str = 'logs/xor'
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
    train_dataset = create_xor_dataset(num_samples=1000).batch(batch_size).shuffle(1000)
    val_dataset = create_xor_dataset(num_samples=200).batch(batch_size)

    # Create model
    if model_name == 'ngn_xor':
        model = create_ngn_xor_model()
        logger.info("Created NGN-XOR model")
    else:
        model = NGNXORModel()  # Baseline with identity layer graph
        logger.info("Created baseline XOR model")

    # Optimizer
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    # Trainer
    trainer = NGNTrainer(
        model=model,
        optimizer=optimizer,
        log_dir=log_dir
    )

    # Visualizer
    visualizer = NGNVisualizer(output_dir='visualizations/xor')

    logger.info(f"Starting {model_name} training")

    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        train_metrics = trainer.train_epoch(train_dataset, epoch)

        # Validate
        val_metrics = trainer.validate(val_dataset, epoch)

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
            for x, y in val_dataset.take(1):
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
    total_correct = 0
    total_samples = 0

    for x, y in val_dataset:
        logits, _ = model(x)
        preds = tf.argmax(logits, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(tf.cast(preds, y.dtype), y), tf.int32))
        total_correct += int(correct.numpy())
        total_samples += int(tf.shape(y)[0].numpy())

    final_accuracy = total_correct / total_samples
    logger.info(f"Final validation accuracy: {final_accuracy:.4f}")

    # Save model
    checkpoint_path = f'checkpoints/{model_name}_final'
    trainer.save_checkpoint(checkpoint_path, num_epochs, {
        'final_accuracy': final_accuracy,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    })

    return {
        'model': model,
        'final_accuracy': final_accuracy
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
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create directories
    Path('logs').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)

    # Run experiment
    results = run_xor_comparison()

    print("XOR Experiment Results:")
    print(f"Baseline Accuracy: {results['baseline_accuracy']:.4f}")
    print(f"NGN Accuracy: {results['ngn_accuracy']:.4f}")
    print(f"Improvement: {results['improvement'] * 100:.2f}%")