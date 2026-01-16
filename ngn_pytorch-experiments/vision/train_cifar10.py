"""
NGN Experiments - Vision (CIFAR-10)

Train NGN-ResNet on CIFAR-10 to validate the approach on real vision data.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging
from pathlib import Path
from typing import Tuple

from ngn_pytorch.backbones.cnn_backbone import NGNResNet
from ngn_pytorch.core.communication import SharedAttentionAggregator
from ngn_pytorch.training.trainer import NGNTrainer
from ngn_pytorch.utils.visualization import NGNVisualizer

logger = logging.getLogger(__name__)


def get_cifar10_dataloaders(
    batch_size: int = 128,
    data_dir: str = 'data'
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and validation dataloaders."""

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader


def create_ngn_resnet_model(num_classes: int = 10) -> NGNResNet:
    """Create NGN-ResNet model for CIFAR-10."""

    # Create layer graph
    layer_graph = SharedAttentionAggregator(
        num_layers=4,  # ResNet-18 has 4 layer groups
        feature_dims=[64, 128, 256, 512],
        num_heads=8,
        dropout=0.1
    )

    # Create NGN-ResNet
    model = NGNResNet(
        backbone_name='resnet18',
        num_classes=num_classes,
        pretrained=False,  # Train from scratch for CIFAR-10
        layer_graph=layer_graph
    )

    return model


def create_baseline_resnet_model(num_classes: int = 10) -> NGNResNet:
    """Create baseline ResNet model (no NGN)."""

    # Identity layer graph (no communication)
    from ngn_pytorch.backbones.cnn_backbone import IdentityLayerGraph

    layer_graph = IdentityLayerGraph(
        num_layers=4,
        feature_dims=[64, 128, 256, 512]
    )

    model = NGNResNet(
        backbone_name='resnet18',
        num_classes=num_classes,
        pretrained=False,
        layer_graph=layer_graph
    )

    return model


def train_cifar10_experiment(
    model_name: str = 'ngn_resnet18',
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.1,
    weight_decay: float = 5e-4,
    log_dir: str = 'logs/cifar10',
    ablation: str = 'dynamic',
    num_heads: int = 8,
    use_residual: bool = True
):
    """
    Train CIFAR-10 experiment.

    Args:
        model_name: Model name ('ngn_resnet18' or 'baseline_resnet18')
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        log_dir: Logging directory
    """

    # Get data
    train_loader, val_loader = get_cifar10_dataloaders(batch_size)

    # Create model
    if ablation == 'static':
        # Static learned graph (no input-dependent attention)
        from ngn_pytorch.core.graph import LayerGraph
        layer_graph = LayerGraph(
            num_layers=4,
            feature_dims=[64, 128, 256, 512],
            use_residual=use_residual,
            use_layer_norm=True
        )
        model = NGNResNet(
            backbone_name='resnet18',
            num_classes=10,
            pretrained=False,
            layer_graph=layer_graph
        )
        logger.info("Created static LayerGraph ResNet18 model")
    elif ablation == 'dynamic':
        # Dynamic attention-based graph
        from ngn_pytorch.core.communication import SharedAttentionAggregator
        layer_graph = SharedAttentionAggregator(
            num_layers=4,
            feature_dims=[64, 128, 256, 512],
            num_heads=num_heads,
            dropout=0.1
        )
        model = NGNResNet(
            backbone_name='resnet18',
            num_classes=10,
            pretrained=False,
            layer_graph=layer_graph
        )
        logger.info(f"Created NGN-ResNet18 model (dynamic, heads={num_heads}, residual={use_residual})")
    elif ablation == 'residual_off':
        # Dynamic attention, no residual
        from ngn_pytorch.core.communication import SharedAttentionAggregator
        layer_graph = SharedAttentionAggregator(
            num_layers=4,
            feature_dims=[64, 128, 256, 512],
            num_heads=num_heads,
            dropout=0.1
        )
        # Patch to disable residuals if supported
        if hasattr(layer_graph, 'use_residual'):
            layer_graph.use_residual = False
        model = NGNResNet(
            backbone_name='resnet18',
            num_classes=10,
            pretrained=False,
            layer_graph=layer_graph
        )
        logger.info(f"Created NGN-ResNet18 model (dynamic, heads={num_heads}, residual=OFF)")
    else:
        # Baseline
        model = create_baseline_resnet_model()
        logger.info("Created baseline ResNet18 model")

    # Optimizer with momentum and weight decay
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = NGNTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        log_dir=log_dir,
        gradient_clip_norm=1.0
    )

    # Visualizer
    visualizer = NGNVisualizer(output_dir='visualizations/cifar10')

    logger.info(f"Starting {model_name} training on {device}")
    logger.info(f"Training data: {len(train_loader.dataset)} samples")
    logger.info(f"Validation data: {len(val_loader.dataset)} samples")

    best_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_metrics = trainer.validate(val_loader, epoch)

        # Step scheduler
        scheduler.step()

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']

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
            checkpoint_path = f'checkpoints/{model_name}_best.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
            logger.info(f"New best accuracy: {best_accuracy:.4f}")

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

            # Plot gradient flow
            visualizer.plot_gradient_flow(
                dict(model.named_parameters()),
                epoch=epoch,
                save=True
            )

    # Final evaluation on best model
    checkpoint = torch.load(f'checkpoints/{model_name}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    final_accuracy = correct / total
    logger.info(f"Final best accuracy: {final_accuracy:.4f}")

    return {
        'model': model,
        'best_accuracy': best_accuracy,
        'final_accuracy': final_accuracy,
        'training_history': trainer.metrics_history
    }


def run_cifar10_comparison():
    """Compare NGN vs baseline on CIFAR-10."""
    logger.info("Running CIFAR-10 comparison experiment")

    # Train baseline
    logger.info("Training baseline ResNet18...")
    baseline_results = train_cifar10_experiment(
        model_name='baseline_resnet18',
        num_epochs=2,  # Very short demo
        batch_size=64,  # Smaller batch size for CPU
        log_dir='logs/cifar10_baseline'
    )

    # Train NGN
    logger.info("Training NGN-ResNet18...")
    ngn_results = train_cifar10_experiment(
        model_name='ngn_resnet18',
        num_epochs=2,  # Very short demo
        batch_size=64,  # Smaller batch size for CPU
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
    import argparse
    logging.basicConfig(level=logging.INFO)
    for dir_name in ['logs', 'checkpoints', 'visualizations', 'data']:
        Path(dir_name).mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="CIFAR-10 NGN Ablation Experiments")
    parser.add_argument('--ablation', type=str, default='dynamic', choices=['dynamic', 'static', 'residual_off', 'baseline'], help='Ablation type')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (if dynamic)')
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    use_residual = not args.no_residual
    results = train_cifar10_experiment(
        model_name='ngn_resnet18' if args.ablation != 'baseline' else 'baseline_resnet18',
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        ablation=args.ablation,
        num_heads=args.num_heads,
        use_residual=use_residual
    )
    print("CIFAR-10 Experiment Results:")
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")