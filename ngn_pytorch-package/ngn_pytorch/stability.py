"""
NGN Training - Stability Monitoring and Callbacks

This module provides utilities for monitoring training stability,
detecting potential issues, and implementing training callbacks.
"""

from typing import Dict, List, Optional, Any, Callable
import torch
from torch import nn, Tensor
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GradientMonitor:
    """
    Monitors gradient statistics during training for stability analysis.

    Tracks gradient norms, NaN/inf values, and provides early warning
    for potential training issues.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = []
        self.layer_gradient_history = []

    def record_gradients(self) -> Dict[str, float]:
        """
        Record current gradient statistics.

        Returns:
            stats: Dictionary with gradient statistics
        """
        total_norm = 0.0
        max_norm = 0.0
        num_params = 0
        layer_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                num_params += 1

                # Record per-layer norms
                layer_name = name.split('.')[0]  # e.g., 'backbone' or 'layer_graph'
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param_norm)

        total_norm = np.sqrt(total_norm) if num_params > 0 else 0.0

        # Average layer norms
        for layer in layer_norms:
            layer_norms[layer] = np.mean(layer_norms[layer])

        stats = {
            'total_norm': total_norm,
            'max_norm': max_norm,
            'num_params': num_params,
            'layer_norms': layer_norms
        }

        self.gradient_history.append(stats)
        self.layer_gradient_history.append(layer_norms)

        return stats

    def check_stability(self, threshold: float = 10.0) -> List[str]:
        """
        Check for potential stability issues.

        Args:
            threshold: Maximum acceptable gradient norm

        Returns:
            warnings: List of warning messages
        """
        warnings = []

        if len(self.gradient_history) == 0:
            return warnings

        current_stats = self.gradient_history[-1]

        # Check total gradient norm
        if current_stats['total_norm'] > threshold:
            warnings.append(
                f"High gradient norm: {current_stats['total_norm']:.4f} > {threshold}"
            )

        # Check for NaN gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                warnings.append(f"NaN gradients in parameter: {name}")

        # Check for exploding gradients (compared to recent history)
        if len(self.gradient_history) > 5:
            recent_norms = [s['total_norm'] for s in self.gradient_history[-5:]]
            if current_stats['total_norm'] > 2 * np.mean(recent_norms):
                warnings.append("Gradient explosion detected")

        return warnings

    def get_gradient_trends(self) -> Dict[str, List[float]]:
        """Get gradient norm trends over training."""
        if len(self.gradient_history) == 0:
            return {}

        trends = {
            'total_norm': [s['total_norm'] for s in self.gradient_history],
            'max_norm': [s['max_norm'] for s in self.gradient_history]
        }

        # Layer-specific trends
        layer_names = set()
        for history in self.layer_gradient_history:
            layer_names.update(history.keys())

        for layer in layer_names:
            trends[f'layer_{layer}'] = [
                h.get(layer, 0.0) for h in self.layer_gradient_history
            ]

        return trends


class TrainingCallback:
    """
    Base class for training callbacks.

    Callbacks can be used to implement custom logging, checkpointing,
    early stopping, etc.
    """

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of each batch."""
        pass

    def on_train_start(self, logs: Dict[str, Any] = None):
        """Called at the start of training."""
        pass

    def on_train_end(self, logs: Dict[str, Any] = None):
        """Called at the end of training."""
        pass


class StabilityCallback(TrainingCallback):
    """
    Callback that monitors training stability and logs warnings.
    """

    def __init__(self, gradient_monitor: GradientMonitor, log_every: int = 10):
        self.monitor = gradient_monitor
        self.log_every = log_every
        self.warnings = []

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Monitor gradients after each batch."""
        if batch % self.log_every == 0:
            stats = self.monitor.record_gradients()
            batch_warnings = self.monitor.check_stability()

            if batch_warnings:
                self.warnings.extend(batch_warnings)
                for warning in batch_warnings:
                    logger.warning(f"Batch {batch}: {warning}")

            # Add stats to logs
            if logs is not None:
                logs.update(stats)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Summarize epoch stability."""
        if self.warnings:
            logger.info(f"Epoch {epoch} stability summary: {len(self.warnings)} warnings")
            self.warnings = []  # Reset for next epoch


class EarlyStoppingCallback(TrainingCallback):
    """
    Early stopping based on validation metric.
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Check if training should stop."""
        if logs is None:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if self._is_better(current_value):
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                # Store best weights (simplified - in practice, store model state)
                pass
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping at epoch {epoch}")

    def _is_better(self, current_value: float) -> bool:
        """Check if current value is better than best."""
        if self.mode == 'min':
            return current_value < self.best_value
        else:
            return current_value > self.best_value

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self.stopped_epoch > 0


class LearningRateSchedulerCallback(TrainingCallback):
    """
    Learning rate scheduling callback.
    """

    def __init__(self, scheduler: Any, monitor: str = None):
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Step the scheduler."""
        if hasattr(self.scheduler, 'step'):
            if self.monitor and logs:
                # Validation-based scheduling
                val_metric = logs.get(self.monitor)
                if val_metric is not None:
                    self.scheduler.step(val_metric)
            else:
                # Epoch-based scheduling
                self.scheduler.step()


def create_stability_callbacks(
    model: nn.Module,
    log_every: int = 10
) -> List[TrainingCallback]:
    """
    Create a standard set of stability monitoring callbacks.

    Args:
        model: Model to monitor
        log_every: How often to log gradient stats

    Returns:
        callbacks: List of training callbacks
    """
    monitor = GradientMonitor(model)
    stability_callback = StabilityCallback(monitor, log_every)

    return [stability_callback]