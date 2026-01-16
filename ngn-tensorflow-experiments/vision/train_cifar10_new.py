"""
NGN Experiments - Vision (CIFAR-10)

Train NGN-ResNet on CIFAR-10 to validate the approach on real vision data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras import mixed_precision
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

from ngn_tensorflow.backbones.cnn_backbone import NGNResNet
from ngn_tensorflow.core.communication import SharedAttentionAggregator
from ngn_tensorflow.training.trainer import NGNTrainer
from ngn_tensorflow.utils.visualization import NGNVisualizer

logger = logging.getLogger(__name__)

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')
logger.info("Mixed precision training enabled (float16)")
from ngn_tensorflow.core.communication import SharedAttentionAggregator
from ngn_tensorflow.training.trainer import NGNTrainer
from ngn_tensorflow.utils.visualization import NGNVisualizer

logger = logging.getLogger(__name__)

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')
logger.info("Mixed precision training enabled (float16)")


def get_cifar10_datasets(
    batch_size: int = 128,
    data_dir: str = 'data'
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Get CIFAR-10 train and validation datasets."""

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Data augmentation for training
    data_augmentation = keras.Sequential([
        layers.RandomCrop(32, 32),
        layers.RandomFlip("horizontal"),
    ])

    # Normalize
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    def normalize(image):
        return (image - mean) / std

    def preprocess_train(x, y):
        x = data_augmentation(x)
        x = normalize(x)
        return x, y

    def preprocess_test(x, y):
        x = normalize(x)
        return x, y

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(50000)
    train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def create_ngn_resnet_model(num_classes: int = 10) -> NGNResNet:
    """Create NGN-ResNet model for CIFAR-10."""

    # Create layer graph
    layer_graph = SharedAttentionAggregator(
        num_layers=4,  # ResNet-50 has 4 layer groups
        feature_dims=[256, 512, 1024, 2048],
        num_heads=8,
        dropout=0.1
    )

    # Create NGN-ResNet
    model = NGNResNet(
        backbone_name='resnet50',
        num_classes=num_classes,
        pretrained=False,  # Train from scratch for CIFAR-10
        layer_graph=layer_graph
    )

    return model


def create_baseline_resnet_model(num_classes: int = 10) -> NGNResNet:
    """Create baseline ResNet model (no NGN)."""

    # Identity layer graph (no communication)
    from ngn_tensorflow.backbones.cnn_backbone import IdentityLayerGraph

    layer_graph = IdentityLayerGraph(
        num_layers=4,
        feature_dims=[256, 512, 1024, 2048]
    )

    model = NGNResNet(
        backbone_name='resnet50',
        num_classes=num_classes,
        pretrained=False,
        layer_graph=layer_graph
    )

    return model


def train_cifar10_experiment(
    model_name: str = 'ngn_resnet50',
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.1,
    weight_decay: float = 5e-4,
    log_dir: str = 'logs/cifar10'
):
    """
    Train CIFAR-10 experiment.

    Args:
        model_name: Model name ('ngn_resnet50' or 'baseline_resnet50')
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        log_dir: Logging directory
    """

    # Get data
    train_dataset, val_dataset = get_cifar10_datasets(batch_size)

    # Create model
    if model_name == 'ngn_resnet50':
        model = create_ngn_resnet_model()
        logger.info("Created NGN-ResNet50 model")
    else:
        model = create_baseline_resnet_model()
        logger.info("Created baseline ResNet50 model")

    # Optimizer with momentum and weight decay
    optimizer = optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    def lr_schedule(epoch):
        return learning_rate * 0.5 * (1 + np.cos(np.pi * epoch / num_epochs))

    lr_callback = callbacks.LearningRateScheduler(lr_schedule)

    # Trainer
    trainer = NGNTrainer(
        model=model,
        optimizer=optimizer,
        log_dir=log_dir,
        gradient_clip_norm=1.0
    )

    # Visualizer
    visualizer = NGNVisualizer(output_dir='visualizations/cifar10')

    logger.info(f"Starting {model_name} training")
    logger.info(f"Training data: {50000} samples")
    logger.info(f"Validation data: {10000} samples")

    best_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        train_metrics = trainer.train_epoch(train_dataset, epoch)

        # Validate
        val_metrics = trainer.validate(val_dataset, epoch)

        # Log current learning rate
        current_lr = lr_schedule(epoch)

        # Log
        logger.info(
            f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}, "
            f"LR={current_lr:.6f}"
        )

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            checkpoint_path = f'checkpoints/{model_name}_best'
            trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
            logger.info(f"New best accuracy: {best_accuracy:.4f}")

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

    # Final evaluation on best model
    trainer.load_checkpoint(f'checkpoints/{model_name}_best')

    total_correct = 0
    total_samples = 0

    for x, y in val_dataset:
        logits, _ = model(x)
        preds = tf.argmax(logits, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.squeeze(y)), tf.int32))
        total_correct += int(correct.numpy())
        total_samples += int(tf.shape(y)[0].numpy())

    final_accuracy = total_correct / total_samples
    logger.info(f"Final best accuracy: {final_accuracy:.4f}")

    return {
        'model': model,
        'best_accuracy': best_accuracy,
        'final_accuracy': final_accuracy
    }


def run_cifar10_comparison():
    """Compare NGN vs baseline on CIFAR-10."""
    logger.info("Running CIFAR-10 comparison experiment")

    # Train baseline
    logger.info("Training baseline ResNet50...")
    baseline_results = train_cifar10_experiment(
        model_name='baseline_resnet50',
        num_epochs=5,  # Reduced for faster testing
        log_dir='logs/cifar10_baseline'
    )

    # Train NGN
    logger.info("Training NGN-ResNet50...")
    ngn_results = train_cifar10_experiment(
        model_name='ngn_resnet50',
        num_epochs=5,  # Reduced for faster testing
        log_dir='logs/cifar10_ngn'
    )

    # Compare results
    baseline_acc = baseline_results['best_accuracy']
    ngn_acc = ngn_results['best_accuracy']

    logger.info(f"Baseline best accuracy: {baseline_acc:.4f}")
    logger.info(f"NGN best accuracy: {ngn_acc:.4f}")
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
    for dir_name in ['logs', 'checkpoints', 'visualizations', 'data']:
        Path(dir_name).mkdir(exist_ok=True)

    # Run experiment
    results = run_cifar10_comparison()

    print("CIFAR-10 Experiment Results:")
    print(f"Baseline Accuracy: {results['baseline_accuracy']:.4f}")
    print(f"NGN Accuracy: {results['ngn_accuracy']:.4f}")
    print(f"Improvement: {results['improvement'] * 100:.2f}%")
