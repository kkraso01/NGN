"""
NGN Topology Module - Dynamic Routing and Graph Learning

This module handles advanced topology learning features like
per-input routing and hierarchical graphs.
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch import nn, Tensor

from .graph import LayerGraph


class DynamicRoutingGate(nn.Module):
    """
    Learns per-input routing decisions.

    This gate network determines which layers should be activated
    or emphasized for different inputs, enabling conditional computation.
    """

    def __init__(self, feature_dim: int, num_layers: int):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pool
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_layers),
            nn.Sigmoid()  # Output routing probabilities
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Compute routing weights for each layer.

        Args:
            features: Input features (batch, channels, height, width)

        Returns:
            routing_weights: (batch, num_layers) - how much to activate each layer
        """
        return self.gate_network(features)


class HierarchicalLayerGraph(LayerGraph):
    """
    Hierarchical graph with intra-group and inter-group connections.

    Organizes layers into groups and learns connectivity at two levels:
    - Dense connections within groups
    - Sparse connections between groups
    """

    def __init__(
        self,
        num_layers: int,
        feature_dims: List[int],
        groups: List[int],  # e.g., [2, 2, 2, 2] for 4 groups of 2 layers each
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__(num_layers, feature_dims, use_residual, use_layer_norm)

        self.groups = groups
        self.num_groups = len(groups)

        # Validate groups sum to num_layers
        assert sum(groups) == num_layers, "Groups must sum to total number of layers"

        # Inter-group adjacency (sparse)
        self.inter_adjacency = nn.Parameter(
            torch.ones(self.num_groups, self.num_groups) / self.num_groups
        )

        # Intra-group connections are dense (all-to-all within group)
        # We'll handle this in forward pass

    def forward(
        self,
        layer_outputs: List[Tensor]
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Apply hierarchical communication.
        """
        # For now, implement simple version
        # TODO: Implement full hierarchical logic

        # Placeholder: just return original outputs with adjacency info
        debug_info = {
            'adjacency_matrix': self.get_adjacency_matrix(),
            'inter_adjacency': self.inter_adjacency.detach(),
            'groups': self.groups
        }

        return layer_outputs, debug_info

    def get_adjacency_matrix(self) -> Tensor:
        """Return hierarchical adjacency matrix."""
        # Build full adjacency from inter and intra connections
        full_adj = torch.zeros(self.num_layers, self.num_layers)

        group_start = 0
        for i, group_size in enumerate(self.groups):
            group_end = group_start + group_size

            # Intra-group: dense connections
            full_adj[group_start:group_end, group_start:group_end] = 1.0

            # Inter-group: sparse connections
            for j, other_size in enumerate(self.groups):
                if i != j:
                    other_start = sum(self.groups[:j])
                    other_end = other_start + other_size
                    full_adj[group_start:group_end, other_start:other_end] = \
                        self.inter_adjacency[i, j]

            group_start = group_end

        return full_adj.detach()