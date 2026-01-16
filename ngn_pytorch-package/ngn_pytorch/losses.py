"""
NGN Training - Loss Functions and Regularization

This module defines loss functions and regularization terms
specific to NGN training.
"""

from typing import List, Dict, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class NGNLoss(nn.Module):
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
        attention_entropy_weight: float = 0.05,
        sparsity_weight: float = 0.005,
        adjacency_entropy_weight: float = 0.0,
        gradient_penalty_weight: float = 0.0
    ):
        super().__init__()

        self.classification_weight = classification_weight
        self.attention_entropy_weight = attention_entropy_weight
        self.sparsity_weight = sparsity_weight
        self.adjacency_entropy_weight = adjacency_entropy_weight
        self.gradient_penalty_weight = gradient_penalty_weight

        self.classification_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        debug_info: Optional[Dict[str, Tensor]] = None,
        model: Optional[nn.Module] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
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
        cls_loss = self.classification_loss(logits, targets)
        total_loss = self.classification_weight * cls_loss

        loss_components = {
            'classification': cls_loss.item()
        }

        # Attention entropy regularization
        if debug_info and ('gated_attention_weights' in debug_info or 'attention_weights' in debug_info):
            attention_source = debug_info.get('gated_attention_weights', debug_info.get('attention_weights'))
            entropy_reg = self._compute_attention_entropy_regularization(attention_source)
            total_loss += self.attention_entropy_weight * entropy_reg
            loss_components['attention_entropy'] = entropy_reg.item()

        # Graph sparsity regularization
        if debug_info and ('adjacency_gate' in debug_info or 'adjacency_matrix' in debug_info):
            adjacency_source = debug_info.get('adjacency_gate', debug_info.get('adjacency_matrix'))
            sparsity_reg = self._compute_sparsity_regularization(adjacency_source)
            total_loss += self.sparsity_weight * sparsity_reg
            loss_components['sparsity'] = sparsity_reg.item()

        if debug_info and self.adjacency_entropy_weight > 0:
            adjacency_source = debug_info.get('adjacency_gate', debug_info.get('adjacency_matrix'))
            if adjacency_source is not None:
                adjacency_entropy = self._compute_adjacency_entropy(adjacency_source)
                total_loss += self.adjacency_entropy_weight * adjacency_entropy
                loss_components['adjacency_entropy'] = adjacency_entropy.item()

        # Gradient penalty (if model provided)
        if model is not None and self.gradient_penalty_weight > 0:
            grad_penalty = self._compute_gradient_penalty(model)
            total_loss += self.gradient_penalty_weight * grad_penalty
            loss_components['gradient_penalty'] = grad_penalty.item()

        loss_components['total'] = total_loss.item()

        return total_loss, loss_components

    def _compute_attention_entropy_regularization(self, attention_weights: Tensor | List[Tensor]) -> Tensor:
        """
        Encourage diverse attention distributions.

        High entropy = attention spread across many layers
        Low entropy = attention focused on few layers
        """
        total_entropy = 0.0
        num_weights = 0

        if isinstance(attention_weights, list):
            weights_list = attention_weights
        else:
            weights_list = [attention_weights]

        for attn in weights_list:
            if attn.dim() == 4:
                # (batch, heads, target, source)
                entropy = -torch.sum(
                    attn * torch.log(attn + 1e-8),
                    dim=-1
                ).mean()
            else:
                entropy = -torch.sum(
                    attn * torch.log(attn + 1e-8),
                    dim=-1
                ).mean()
            total_entropy += entropy
            num_weights += 1

        return total_entropy / max(num_weights, 1)

    def _compute_sparsity_regularization(self, adjacency_matrix: Tensor) -> Tensor:
        """
        Encourage sparse adjacency matrices.

        L1 penalty on adjacency weights promotes selective connections.
        """
        return torch.abs(adjacency_matrix).sum()

    def _compute_adjacency_entropy(self, adjacency_matrix: Tensor) -> Tensor:
        gate = adjacency_matrix
        if gate.dim() != 2:
            raise ValueError("Adjacency matrix must be 2D.")
        probs = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        return entropy

    def _compute_gradient_penalty(self, model: nn.Module) -> Tensor:
        """
        Monitor gradient magnitudes for stability.

        Penalizes large gradients that might cause training instability.
        """
        total_norm = 0
        num_params = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                num_params += 1

        if num_params == 0:
            return torch.tensor(0.0)

        total_norm = total_norm / num_params
        return torch.tensor(total_norm)


def compute_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def compute_per_class_accuracy(logits: Tensor, targets: Tensor, num_classes: int) -> List[float]:
    """Compute per-class accuracy."""
    preds = logits.argmax(dim=1)

    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            per_class_correct[c] = ((preds == c) & mask).sum().item()
            per_class_total[c] = mask.sum().item()

    per_class_acc = []
    for c in range(num_classes):
        if per_class_total[c] > 0:
            per_class_acc.append(per_class_correct[c] / per_class_total[c])
        else:
            per_class_acc.append(0.0)

    return per_class_acc
