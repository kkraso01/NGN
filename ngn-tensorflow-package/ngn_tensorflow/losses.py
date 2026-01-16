"""
NGN Training - Loss Functions and Regularization

This module defines loss functions and regularization terms
specific to NGN training.
"""

from typing import List, Dict, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses


class NGNLoss(keras.layers.Layer):
    """
    Combined loss function for NGN training.

    Includes classification loss plus regularization terms for:
    - Attention entropy (encourage diverse attention)
    - Graph sparsity (encourage selective connections)
    - Gradient stability (monitor training dynamics)
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        attention_entropy_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        gradient_penalty_weight: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.classification_weight = classification_weight
        self.attention_entropy_weight = attention_entropy_weight
        self.sparsity_weight = sparsity_weight
        self.gradient_penalty_weight = gradient_penalty_weight

        self.classification_loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(
        self,
        logits: tf.Tensor,
        targets: tf.Tensor,
        debug_info: Optional[Dict[str, tf.Tensor]] = None,
        model: Optional[keras.Model] = None
    ) -> Tuple[tf.Tensor, Dict[str, float]]:
        """
        Compute total NGN loss.

        Args:
            logits: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
            debug_info: Debug information from model forward pass
            model: Model instance (for gradient penalty)

        Returns:
            total_loss: Combined loss tensor
            loss_components: Dictionary of individual loss components
        """

        # Classification loss
        cls_loss = self.classification_loss(targets, logits)
        total_loss = self.classification_weight * cls_loss

        loss_components = {
            'classification': float(cls_loss.numpy())
        }

        # Attention entropy regularization
        if debug_info and 'attention_weights' in debug_info:
            entropy_reg = self._compute_attention_entropy_regularization(
                debug_info['attention_weights']
            )
            total_loss += self.attention_entropy_weight * entropy_reg
            loss_components['attention_entropy'] = float(entropy_reg.numpy())

        # Graph sparsity regularization
        if debug_info and 'adjacency_matrix' in debug_info:
            sparsity_reg = self._compute_sparsity_regularization(
                debug_info['adjacency_matrix']
            )
            total_loss += self.sparsity_weight * sparsity_reg
            loss_components['sparsity'] = float(sparsity_reg.numpy())

        # Gradient penalty (if model provided)
        if model is not None and self.gradient_penalty_weight > 0:
            grad_penalty = self._compute_gradient_penalty(model)
            total_loss += self.gradient_penalty_weight * grad_penalty
            loss_components['gradient_penalty'] = float(grad_penalty.numpy())

        loss_components['total'] = float(total_loss.numpy())

        return total_loss, loss_components

    def _compute_attention_entropy_regularization(self, attention_weights: List[tf.Tensor]) -> tf.Tensor:
        """
        Encourage diverse attention distributions.

        High entropy = attention spread across many layers
        Low entropy = attention focused on few layers
        """
        total_entropy = 0.0
        num_weights = 0

        for attn in attention_weights:
            if isinstance(attn, list):
                # Multi-head attention
                for head_attn in attn:
                    entropy = -tf.reduce_sum(
                        head_attn * tf.math.log(head_attn + 1e-8),
                        axis=-1
                    )
                    total_entropy += tf.reduce_mean(entropy)
                    num_weights += 1
            else:
                # Single attention matrix
                entropy = -tf.reduce_sum(
                    attn * tf.math.log(attn + 1e-8),
                    axis=-1
                )
                total_entropy += tf.reduce_mean(entropy)
                num_weights += 1

        return total_entropy / max(num_weights, 1)

    def _compute_sparsity_regularization(self, adjacency_matrix: tf.Tensor) -> tf.Tensor:
        """
        Encourage sparse adjacency matrices.

        L1 penalty on adjacency weights promotes selective connections.
        """
        return tf.reduce_sum(tf.abs(adjacency_matrix))

    def _compute_gradient_penalty(self, model: keras.Model) -> tf.Tensor:
        """
        Monitor gradient magnitudes for stability.

        Penalizes large gradients that might cause training instability.
        """
        total_norm = 0.0
        num_params = 0

        for var in model.trainable_variables:
            if var.grad is not None:
                param_norm = tf.norm(var.grad, ord=2)
                total_norm += param_norm.numpy() ** 2
                num_params += 1

        if num_params == 0:
            return tf.constant(0.0)

        total_norm = total_norm / num_params
        return tf.constant(total_norm)


def compute_accuracy(logits: tf.Tensor, targets: tf.Tensor) -> float:
    """Compute classification accuracy."""
    preds = tf.argmax(logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(preds, targets), tf.float32))
    total = tf.cast(tf.shape(targets)[0], tf.float32)
    return float((correct / total).numpy())


def compute_per_class_accuracy(logits: tf.Tensor, targets: tf.Tensor, num_classes: int) -> List[float]:
    """Compute per-class accuracy."""
    preds = tf.argmax(logits, axis=1)

    per_class_correct = [0.0] * num_classes
    per_class_total = [0.0] * num_classes

    for c in range(num_classes):
        mask = tf.equal(targets, c)
        if tf.reduce_sum(tf.cast(mask, tf.int32)) > 0:
            correct_mask = tf.logical_and(tf.equal(preds, c), mask)
            per_class_correct[c] = float(tf.reduce_sum(tf.cast(correct_mask, tf.int32)).numpy())
            per_class_total[c] = float(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy())

    per_class_acc = []
    for c in range(num_classes):
        if per_class_total[c] > 0:
            per_class_acc.append(per_class_correct[c] / per_class_total[c])
        else:
            per_class_acc.append(0.0)

    return per_class_acc

    def __init__(
        self,
        classification_weight: float = 1.0,
        attention_entropy_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        gradient_penalty_weight: float = 0.0
    ):
        super().__init__()

        self.classification_weight = classification_weight
        self.attention_entropy_weight = attention_entropy_weight
        self.sparsity_weight = sparsity_weight
        self.gradient_penalty_weight = gradient_penalty_weight

        self.classification_loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(
        self,
        logits: tf.Tensor,
        targets: tf.Tensor,
        debug_info: Optional[Dict[str, tf.Tensor]] = None,
        model: Optional[keras.Model] = None
    ) -> Tuple[tf.Tensor, Dict[str, float]]:
        """
        Compute total NGN loss.

        Args:
            logits: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
            debug_info: Debug information from model forward pass
            model: Model instance (for gradient penalty)

        Returns:
            total_loss: Combined loss tensor
            loss_components: Dictionary of individual loss components
        """

        # Classification loss
        cls_loss = self.classification_loss(targets, logits)
        total_loss = self.classification_weight * cls_loss

        loss_components = {
            'classification': float(cls_loss.numpy())
        }

        # Attention entropy regularization
        if debug_info and 'attention_weights' in debug_info:
            entropy_reg = self._compute_attention_entropy_regularization(
                debug_info['attention_weights']
            )
            total_loss += self.attention_entropy_weight * entropy_reg
            loss_components['attention_entropy'] = float(entropy_reg.numpy())

        # Graph sparsity regularization
        if debug_info and 'adjacency_matrix' in debug_info:
            sparsity_reg = self._compute_sparsity_regularization(
                debug_info['adjacency_matrix']
            )
            total_loss += self.sparsity_weight * sparsity_reg
            loss_components['sparsity'] = float(sparsity_reg.numpy())

        # Gradient penalty (if model provided)
        if model is not None and self.gradient_penalty_weight > 0:
            grad_penalty = self._compute_gradient_penalty(model)
            total_loss += self.gradient_penalty_weight * grad_penalty
            loss_components['gradient_penalty'] = float(grad_penalty.numpy())

        loss_components['total'] = float(total_loss.numpy())

        return total_loss, loss_components

    def _compute_attention_entropy_regularization(self, attention_weights: List[tf.Tensor]) -> tf.Tensor:
        """
        Encourage diverse attention distributions.

        High entropy = attention spread across many layers
        Low entropy = attention focused on few layers
        """
        total_entropy = 0.0
        num_weights = 0

        for attn in attention_weights:
            if isinstance(attn, list):
                # Multi-head attention
                for head_attn in attn:
                    entropy = -tf.reduce_sum(
                        head_attn * tf.math.log(head_attn + 1e-8),
                        axis=-1
                    )
                    total_entropy += tf.reduce_mean(entropy)
                    num_weights += 1
            else:
                # Single attention matrix
                entropy = -tf.reduce_sum(
                    attn * tf.math.log(attn + 1e-8),
                    axis=-1
                )
                total_entropy += tf.reduce_mean(entropy)
                num_weights += 1

        return total_entropy / max(num_weights, 1)

    def _compute_sparsity_regularization(self, adjacency_matrix: tf.Tensor) -> tf.Tensor:
        """
        Encourage sparse adjacency matrices.

        L1 penalty on adjacency weights promotes selective connections.
        """
        return tf.reduce_sum(tf.abs(adjacency_matrix))

    def _compute_gradient_penalty(self, model: keras.Model) -> tf.Tensor:
        """
        Monitor gradient magnitudes for stability.

        Penalizes large gradients that might cause training instability.
        """
        total_norm = 0.0
        num_params = 0

        for var in model.trainable_variables:
            if hasattr(var, 'grad') and var.grad is not None:
                param_norm = tf.norm(var.grad, ord=2)
                total_norm += float(param_norm.numpy()) ** 2
                num_params += 1

        if num_params == 0:
            return tf.constant(0.0)

        total_norm = total_norm / num_params
        return tf.constant(total_norm)


def compute_accuracy(logits: tf.Tensor, targets: tf.Tensor) -> float:
    """Compute classification accuracy."""
    preds = tf.argmax(logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(preds, targets), tf.float32))
    total = tf.cast(tf.shape(targets)[0], tf.float32)
    return float((correct / total).numpy())


def compute_per_class_accuracy(logits: tf.Tensor, targets: tf.Tensor, num_classes: int) -> List[float]:
    """Compute per-class accuracy."""
    preds = tf.argmax(logits, axis=1)

    per_class_correct = [0.0] * num_classes
    per_class_total = [0.0] * num_classes

    for c in range(num_classes):
        mask = tf.equal(targets, c)
        mask_sum = tf.reduce_sum(tf.cast(mask, tf.int32))
        if mask_sum > 0:
            correct_mask = tf.logical_and(tf.equal(preds, c), mask)
            per_class_correct[c] = float(tf.reduce_sum(tf.cast(correct_mask, tf.int32)).numpy())
            per_class_total[c] = float(mask_sum.numpy())

    per_class_acc = []
    for c in range(num_classes):
        if per_class_total[c] > 0:
            per_class_acc.append(per_class_correct[c] / per_class_total[c])
        else:
            per_class_acc.append(0.0)

    return per_class_acc