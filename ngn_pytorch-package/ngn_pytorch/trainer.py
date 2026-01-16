"""
NGN Training Module - Training Loop and Utilities

This module provides training utilities, loss functions, and
stability monitoring for NGN models.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
from contextlib import nullcontext
from pathlib import Path

from .losses import NGNLoss
from .stability import GradientMonitor

logger = logging.getLogger(__name__)


class NGNTrainer:
    """
    Trainer class for NGN models with stability monitoring and logging.

    Args:
        model: NGN model to train
        optimizer: Optimizer (Adam, SGD, etc.)
        device: Device to train on ('cuda' or 'cpu')
        log_dir: Directory for TensorBoard logs
        gradient_clip_norm: Maximum gradient norm for clipping
        use_mixed_precision: Whether to use automatic mixed precision
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        log_dir: str = 'logs/',
        gradient_clip_norm: float = 1.0,
        use_mixed_precision: bool = False,
        loss_fn: Optional[NGNLoss] = None,
        gradient_monitor_interval: int = 50,
        gradient_monitor_threshold: float = 10.0
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = use_mixed_precision
        self.loss_fn = loss_fn or NGNLoss()
        self.gradient_monitor_interval = gradient_monitor_interval
        self.gradient_monitor_threshold = gradient_monitor_threshold

        # Logging setup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_file = self.log_dir / 'metrics.json'
        self.metrics_history = []
        self.global_step = 0

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

        # Stability monitoring
        self.gradient_norms = []
        self.layer_gradient_norms = []
        self.gradient_monitor = GradientMonitor(self.model)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_every: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            log_every: Log frequency (batches)

        Returns:
            Metrics dictionary
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        component_totals: Dict[str, float] = {}
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            with torch.cuda.amp.autocast() if self.scaler else nullcontext():
                logits, debug_info = self.model(x)
                loss, loss_components = self.loss_fn(
                    logits=logits,
                    targets=y,
                    debug_info=debug_info,
                    model=self.model
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )

                self._check_gradients()

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )

                self._check_gradients()

                self.optimizer.step()

            # Compute accuracy
            if y.dim() > 1 and y.size(1) > 1:
                # One-hot encoded labels
                targets = y.argmax(dim=1)
            else:
                # Class indices
                targets = y

            preds = logits.argmax(dim=1)
            accuracy = (preds == targets).float().mean().item()

            # Update totals
            total_loss += loss.item()
            total_accuracy += accuracy
            for key, value in loss_components.items():
                component_totals[key] = component_totals.get(key, 0.0) + value

            # Log batch metrics
            if batch_idx % log_every == 0:
                self._log_batch_metrics(
                    loss.item(), accuracy, loss_components, debug_info, epoch, batch_idx
                )

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}"
            })

            self.global_step += 1

        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_accuracy = total_accuracy / num_batches

        # Log epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'step': self.global_step
        }
        for key, value in component_totals.items():
            epoch_metrics[key] = value / num_batches
        self.metrics_history.append(epoch_metrics)
        self._save_metrics()

        logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")

        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }

    def validate(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader
            epoch: Current epoch number

        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                logits, _ = self.model(x)
                loss, _ = self.loss_fn(logits=logits, targets=y, debug_info=None, model=None)

                preds = logits.argmax(dim=1)
                if y.dim() > 1 and y.size(1) > 1:
                    # One-hot encoded labels
                    targets = y.argmax(dim=1)
                else:
                    # Class indices
                    targets = y
                accuracy = (preds == targets).float().mean().item()

                total_loss += loss.item()
                total_accuracy += accuracy

        val_loss = total_loss / num_batches
        val_accuracy = total_accuracy / num_batches

        # Log validation metrics
        val_metrics = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'step': self.global_step
        }
        self.metrics_history.append(val_metrics)
        self._save_metrics()

        logger.info(f"Validation {epoch}: Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")

        return {
            'loss': val_loss,
            'accuracy': val_accuracy
        }

    def _log_batch_metrics(
        self,
        loss: float,
        accuracy: float,
        loss_components: Dict[str, float],
        debug_info: Dict[str, Any],
        epoch: int,
        batch_idx: int
    ):
        """Log detailed batch metrics."""
        # Log basic metrics
        if batch_idx % 50 == 0:  # Log less frequently for batch metrics
            logger.debug(
                "Batch %s: Loss=%.4f, Acc=%.4f, Components=%s",
                batch_idx,
                loss,
                accuracy,
                {k: f"{v:.4f}" for k, v in loss_components.items()}
            )

        # Log gradient norms if available
        if hasattr(self.model, 'parameters'):
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Monitor for stability
            if total_norm > 10.0:
                logger.warning(f"Large gradient norm: {total_norm:.4f} at step {self.global_step}")

        # Log attention weights statistics if available
        if 'attention_weights' in debug_info and batch_idx % 50 == 0:
            attn_weights = debug_info['attention_weights']
            if isinstance(attn_weights, list):
                sample = attn_weights[0]
            else:
                sample = attn_weights
            if isinstance(sample, Tensor):
                entropy = -torch.sum(sample * torch.log(sample + 1e-8), dim=-1).mean().item()
                logger.debug("Attention entropy: %.4f", entropy)

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)

        logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch

    def _save_metrics(self):
        """Save metrics history to JSON file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    def _check_gradients(self) -> None:
        if self.gradient_monitor_interval <= 0:
            return
        if self.global_step % self.gradient_monitor_interval != 0:
            return
        stats = self.gradient_monitor.record_gradients()
        self.gradient_norms.append(stats['total_norm'])
        self.layer_gradient_norms.append(stats['layer_norms'])
        warnings = self.gradient_monitor.check_stability(self.gradient_monitor_threshold)
        for warning in warnings:
            logger.warning("Gradient warning at step %s: %s", self.global_step, warning)
