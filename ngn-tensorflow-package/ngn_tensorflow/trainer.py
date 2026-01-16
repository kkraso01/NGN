"""
NGN Training Module - Training Loop and Utilities

This module provides training utilities, loss functions, and
stability monitoring for NGN models.
"""

from typing import Dict, List, Optional, Tuple, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics
import numpy as np
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)


class NGNTrainer:
    """
    Trainer class for NGN models with stability monitoring and logging.

    Args:
        model: NGN model to train
        optimizer: Optimizer (Adam, SGD, etc.)
        log_dir: Directory for TensorBoard logs
        gradient_clip_norm: Maximum gradient norm for clipping
        use_mixed_precision: Whether to use automatic mixed precision
    """

    def __init__(
        self,
        model: keras.Model,
        optimizer: optimizers.Optimizer,
        log_dir: str = 'logs/',
        gradient_clip_norm: float = 1.0,
        use_mixed_precision: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_dir = log_dir
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = use_mixed_precision

        # Logging
        self.writer = tf.summary.create_file_writer(log_dir)
        self.global_step = tf.Variable(0, dtype=tf.int64)

        # Mixed precision
        if use_mixed_precision:
            self.policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(self.policy)

        # Stability monitoring
        self.gradient_norms = []
        self.layer_gradient_norms = []

    def train_epoch(
        self,
        dataset: tf.data.Dataset,
        epoch: int,
        log_every: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataset: Training dataset
            epoch: Current epoch number
            log_every: Log frequency (batches)

        Returns:
            Metrics dictionary
        """
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(dataset, desc=f"Epoch {epoch}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            with tf.GradientTape() as tape:
                logits, debug_info = self.model(x, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

                # Add regularization if attention weights available
                if 'attention_weights' in debug_info:
                    entropy_reg = self._compute_attention_entropy_regularization(
                        debug_info['attention_weights']
                    )
                    loss = loss + 0.1 * entropy_reg

            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)

            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Compute accuracy
            preds = tf.argmax(logits, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(preds, y.dtype), y), tf.float32))

            # Update totals
            total_loss += float(loss.numpy())
            total_accuracy += float(accuracy.numpy())
            num_batches += 1

            # Log batch metrics
            if batch_idx % log_every == 0:
                self._log_batch_metrics(
                    float(loss.numpy()), float(accuracy.numpy()), debug_info, epoch, batch_idx
                )

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{float(loss.numpy()):.4f}",
                'acc': f"{float(accuracy.numpy()):.4f}"
            })

            self.global_step.assign_add(1)

        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_accuracy = total_accuracy / num_batches

        # Log epoch metrics
        with self.writer.as_default():
            tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
            tf.summary.scalar('epoch/accuracy', epoch_accuracy, step=epoch)

        logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")

        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }

    def validate(
        self,
        dataset: tf.data.Dataset,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            dataset: Validation dataset
            epoch: Current epoch number

        Returns:
            Validation metrics
        """
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for x, y in dataset:
            logits, _ = self.model(x, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

            preds = tf.argmax(logits, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(preds, y.dtype), y), tf.float32))

            total_loss += float(loss.numpy())
            total_accuracy += float(accuracy.numpy())
            num_batches += 1

        val_loss = total_loss / num_batches
        val_accuracy = total_accuracy / num_batches

        # Log validation metrics
        with self.writer.as_default():
            tf.summary.scalar('validation/loss', val_loss, step=epoch)
            tf.summary.scalar('validation/accuracy', val_accuracy, step=epoch)

        logger.info(f"Validation {epoch}: Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")

        return {
            'loss': val_loss,
            'accuracy': val_accuracy
        }

    def _log_batch_metrics(
        self,
        loss: float,
        accuracy: float,
        debug_info: Dict[str, Any],
        epoch: int,
        batch_idx: int
    ):
        """Log detailed batch metrics."""
        with self.writer.as_default():
            tf.summary.scalar('batch/loss', loss, step=self.global_step.numpy())
            tf.summary.scalar('batch/accuracy', accuracy, step=self.global_step.numpy())

            # Log gradient norms if available (simplified)
            # In TensorFlow, gradients are computed per step, so we'd need to store them

            # Log attention weights statistics if available
            if 'attention_weights' in debug_info:
                attn_weights = debug_info['attention_weights']
                if isinstance(attn_weights, list) and len(attn_weights) > 0:
                    # Compute entropy of attention distributions
                    entropy = self._compute_attention_entropy(attn_weights[0])
                    tf.summary.scalar('batch/attention_entropy', entropy, step=self.global_step.numpy())

    def _compute_attention_entropy_regularization(self, attention_weights: List[tf.Tensor]) -> tf.Tensor:
        """Compute entropy regularization for attention weights."""
        total_entropy = 0.0

        for attn in attention_weights:
            if isinstance(attn, list):
                attn = attn[0]  # Take first head if multi-head

            # Compute entropy: -sum(p * log(p))
            entropy = -tf.reduce_sum(attn * tf.math.log(attn + 1e-8), axis=-1)
            total_entropy += tf.reduce_mean(entropy)

        return total_entropy / len(attention_weights)

    def _compute_attention_entropy(self, attention_weights: tf.Tensor) -> float:
        """Compute average entropy of attention weights."""
        # attention_weights shape: (batch, num_heads, seq_len, seq_len)
        entropy = -tf.reduce_sum(
            attention_weights * tf.math.log(attention_weights + 1e-8),
            axis=-1
        )
        return float(tf.reduce_mean(entropy).numpy())

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_weights': self.model.get_weights(),
            # 'optimizer_weights': self.optimizer.get_weights(),  # Removed due to TF version issue
            'metrics': metrics,
            'global_step': self.global_step.numpy()
        }

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint."""
        import pickle
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.model.set_weights(checkpoint['model_weights'])
        # if 'optimizer_weights' in checkpoint:
        #     self.optimizer.set_weights(checkpoint['optimizer_weights'])  # Removed

        epoch = checkpoint['epoch']
        self.global_step.assign(checkpoint.get('global_step', 0))

        logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch