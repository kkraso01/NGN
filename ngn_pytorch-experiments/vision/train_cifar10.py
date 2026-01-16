"""
NGN Experiments - Vision (CIFAR-10)

Train NGN-ResNet on CIFAR-10 to validate the approach on real vision data.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging
from pathlib import Path
from typing import Tuple
import random
import numpy as np
import json

from ngn_pytorch.cnn_backbone import NGNResNet, IdentityLayerGraph
from ngn_pytorch.communication import SharedAttentionAggregator
from ngn_pytorch.graph import StaticLayerGraph
from ngn_pytorch.losses import NGNLoss
from ngn_pytorch.trainer import NGNTrainer
from ngn_pytorch.visualization import NGNVisualizer

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


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_ngn_resnet_model(
    num_classes: int = 10,
    num_heads: int = 8,
    use_residual: bool = True
) -> NGNResNet:
    """Create NGN-ResNet model for CIFAR-10."""

    # Create layer graph
    layer_graph = SharedAttentionAggregator(
        num_layers=4,  # ResNet-18 has 4 layer groups
        feature_dims=[64, 128, 256, 512],
        num_heads=num_heads,
        dropout=0.1,
        use_residual=use_residual
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
    use_residual: bool = True,
    attention_entropy_weight: float = 0.05,
    adjacency_sparsity_weight: float = 0.005,
    adjacency_entropy_weight: float = 0.0,
    gradient_clip_norm: float = 1.0,
    gradient_monitor_interval: int = 50,
    seed: int = 42,
    deterministic: bool = False
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

    set_seed(seed, deterministic)

    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'ablation': ablation,
        'num_heads': num_heads,
        'use_residual': use_residual,
        'attention_entropy_weight': attention_entropy_weight,
        'adjacency_sparsity_weight': adjacency_sparsity_weight,
        'adjacency_entropy_weight': adjacency_entropy_weight,
        'gradient_clip_norm': gradient_clip_norm,
        'gradient_monitor_interval': gradient_monitor_interval,
        'seed': seed,
        'deterministic': deterministic
    }
    logger.info("Run configuration: %s", config)
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    with (log_dir_path / 'config.json').open('w') as config_file:
        json.dump(config, config_file, indent=2)

    # Get data
    train_loader, val_loader = get_cifar10_dataloaders(batch_size)

    # Create model
    if ablation == 'static':
        layer_graph = StaticLayerGraph(
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
        layer_graph = SharedAttentionAggregator(
            num_layers=4,
            feature_dims=[64, 128, 256, 512],
            num_heads=num_heads,
            dropout=0.1,
            use_residual=use_residual
        )
        model = NGNResNet(
            backbone_name='resnet18',
            num_classes=10,
            pretrained=False,
            layer_graph=layer_graph
        )
        logger.info(f"Created NGN-ResNet18 model (dynamic, heads={num_heads}, residual={use_residual})")
    elif ablation == 'identity':
        layer_graph = IdentityLayerGraph(
            num_layers=4,
            feature_dims=[64, 128, 256, 512]
        )
        model = NGNResNet(
            backbone_name='resnet18',
            num_classes=10,
            pretrained=False,
            layer_graph=layer_graph
        )
        logger.info("Created identity LayerGraph ResNet18 model")
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
        gradient_clip_norm=gradient_clip_norm,
        loss_fn=NGNLoss(
            attention_entropy_weight=attention_entropy_weight,
            sparsity_weight=adjacency_sparsity_weight,
            adjacency_entropy_weight=adjacency_entropy_weight
        ),
        gradient_monitor_interval=gradient_monitor_interval
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
    parser.add_argument('--ablation', type=str, default='dynamic', choices=['baseline', 'identity', 'static', 'dynamic'], help='Ablation type')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (if dynamic)')
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--log_dir', type=str, default='logs/cifar10', help='Log directory')
    parser.add_argument('--attention_entropy_weight', type=float, default=0.05, help='Attention entropy regularization weight')
    parser.add_argument('--adjacency_sparsity_weight', type=float, default=0.005, help='Adjacency sparsity regularization weight')
    parser.add_argument('--adjacency_entropy_weight', type=float, default=0.0, help='Adjacency entropy regularization weight')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--gradient_monitor_interval', type=int, default=50, help='Gradient monitoring interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic CuDNN')
    args = parser.parse_args()

    use_residual = not args.no_residual
    results = train_cifar10_experiment(
        model_name='ngn_resnet18' if args.ablation != 'baseline' else 'baseline_resnet18',
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        ablation=args.ablation,
        num_heads=args.num_heads,
        use_residual=use_residual,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir=args.log_dir,
        attention_entropy_weight=args.attention_entropy_weight,
        adjacency_sparsity_weight=args.adjacency_sparsity_weight,
        adjacency_entropy_weight=args.adjacency_entropy_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        gradient_monitor_interval=args.gradient_monitor_interval,
        seed=args.seed,
        deterministic=args.deterministic
    )
    print("CIFAR-10 Experiment Results:")
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
