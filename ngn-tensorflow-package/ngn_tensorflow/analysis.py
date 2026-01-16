"""
NGN Utils - Analysis Tools

This module provides tools for analyzing learned NGN graphs,
interpreting layer communication patterns, and extracting insights.
"""

from typing import List, Dict, Optional, Tuple, Any
import tensorflow as tf
import numpy as np
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """
    Analyzes learned layer graphs and communication patterns.

    Provides methods to:
    - Identify hub layers
    - Detect communities
    - Measure graph properties
    - Compare graphs across training
    """

    def __init__(self):
        self.graph_cache = {}

    def analyze_adjacency_matrix(
        self,
        adjacency_matrix: tf.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze properties of the learned adjacency matrix.

        Args:
            adjacency_matrix: (num_layers, num_layers) adjacency matrix
            threshold: Threshold for considering connections

        Returns:
            analysis: Dictionary with graph analysis results
        """
        adj_np = adjacency_matrix.numpy()
        num_layers = adj_np.shape[0]

        # Basic statistics
        sparsity = np.mean(adj_np < threshold)
        avg_strength = np.mean(adj_np)
        max_strength = np.max(adj_np)

        # Per-layer statistics
        incoming_connections = np.sum(adj_np > threshold, axis=0)  # To each layer
        outgoing_connections = np.sum(adj_np > threshold, axis=1)  # From each layer

        # Identify hubs
        hub_threshold = np.percentile(outgoing_connections, 75)
        hubs = np.where(outgoing_connections >= hub_threshold)[0]

        # Self-connections
        self_connections = np.diag(adj_np)

        analysis = {
            'num_layers': num_layers,
            'sparsity': sparsity,
            'avg_connection_strength': avg_strength,
            'max_connection_strength': max_strength,
            'incoming_connections': incoming_connections.tolist(),
            'outgoing_connections': outgoing_connections.tolist(),
            'hub_layers': hubs.tolist(),
            'self_connections': self_connections.tolist(),
            'is_symmetric': np.allclose(adj_np, adj_np.T),
            'is_fully_connected': np.sum(adj_np > threshold) == num_layers ** 2
        }

        return analysis

    def detect_communities(
        self,
        adjacency_matrix: tf.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect communities in the layer graph using network analysis.

        Args:
            adjacency_matrix: Learned adjacency matrix
            threshold: Connection threshold

        Returns:
            communities: Dictionary with community detection results
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX not available for community detection")
            return {'error': 'NetworkX not installed'}

        adj_np = adjacency_matrix.numpy()

        # Create networkx graph
        G = nx.from_numpy_array((adj_np > threshold).astype(int))

        # Detect communities using Louvain method (if available)
        try:
            communities = list(nx.community.louvain_communities(G))
            community_labels = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    community_labels[node] = comm_id

            num_communities = len(communities)
            modularity = nx.community.modularity(G, communities)

        except AttributeError:
            # Fallback to simpler community detection
            communities = list(nx.community.greedy_modularity_communities(G))
            community_labels = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    community_labels[node] = comm_id

            num_communities = len(communities)
            modularity = nx.community.modularity(G, communities)

        # Community sizes
        community_sizes = [len(comm) for comm in communities]

        # Intra-community density
        intra_density = []
        for comm in communities:
            subgraph = G.subgraph(comm)
            if len(comm) > 1:
                density = nx.density(subgraph)
            else:
                density = 0.0
            intra_density.append(density)

        return {
            'num_communities': num_communities,
            'community_sizes': community_sizes,
            'community_labels': community_labels,
            'modularity': modularity,
            'intra_community_density': intra_density,
            'communities': [list(comm) for comm in communities]
        }

    def analyze_attention_patterns(
        self,
        attention_weights: List[tf.Tensor]
    ) -> Dict[str, Any]:
        """
        Analyze attention weight patterns across layers.

        Args:
            attention_weights: List of attention weight tensors

        Returns:
            patterns: Dictionary with attention pattern analysis
        """
        if not attention_weights:
            return {'error': 'No attention weights provided'}

        patterns = {
            'num_layers': len(attention_weights),
            'layer_patterns': []
        }

        for i, attn in enumerate(attention_weights):
            if isinstance(attn, list):
                # Multi-head attention - analyze each head
                head_patterns = []
                for head_attn in attn:
                    head_patterns.append(self._analyze_single_attention(head_attn))
                layer_pattern = self._aggregate_head_patterns(head_patterns)
            else:
                layer_pattern = self._analyze_single_attention(attn)

            patterns['layer_patterns'].append(layer_pattern)

        # Cross-layer patterns
        patterns.update(self._analyze_cross_layer_patterns(patterns['layer_patterns']))

        return patterns

    def _analyze_single_attention(self, attention: tf.Tensor) -> Dict[str, float]:
        """Analyze a single attention matrix."""
        attn_np = attention.numpy()

        # Average over batch dimension if present
        if attn_np.ndim == 3:
            attn_np = np.mean(attn_np, axis=0)

        # Basic statistics
        entropy = -np.sum(attn_np * np.log(attn_np + 1e-8), axis=-1).mean()
        concentration = np.max(attn_np, axis=-1).mean()  # Max attention per query
        uniformity = np.std(attn_np, axis=-1).mean()  # Variation across keys

        # Self-attention (diagonal)
        if attn_np.shape[0] == attn_np.shape[1]:
            self_attention = np.diag(attn_np).mean()
        else:
            self_attention = 0.0

        return {
            'entropy': entropy,
            'concentration': concentration,
            'uniformity': uniformity,
            'self_attention': self_attention,
            'shape': attn_np.shape
        }

    def _aggregate_head_patterns(self, head_patterns: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate patterns across attention heads."""
        if not head_patterns:
            return {}

        aggregated = {}
        keys = head_patterns[0].keys()

        for key in keys:
            if key == 'shape':
                aggregated[key] = head_patterns[0][key]  # Keep first shape
            else:
                values = [p[key] for p in head_patterns]
                aggregated[key] = np.mean(values)

        return aggregated

    def _analyze_cross_layer_patterns(self, layer_patterns: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze patterns across layers."""
        if not layer_patterns:
            return {}

        entropies = [p['entropy'] for p in layer_patterns]
        concentrations = [p['concentration'] for p in layer_patterns]

        return {
            'avg_entropy': np.mean(entropies),
            'entropy_std': np.std(entropies),
            'avg_concentration': np.mean(concentrations),
            'concentration_std': np.std(concentrations),
            'most_focused_layer': np.argmax(concentrations),
            'most_diverse_layer': np.argmax(entropies)
        }

    def compare_graphs(
        self,
        graph1: tf.Tensor,
        graph2: tf.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Compare two adjacency matrices.

        Args:
            graph1: First adjacency matrix
            graph2: Second adjacency matrix
            threshold: Connection threshold

        Returns:
            comparison: Dictionary with comparison metrics
        """
        g1_binary = tf.cast(graph1 > threshold, tf.float32)
        g2_binary = tf.cast(graph2 > threshold, tf.float32)

        # Jaccard similarity for connections
        intersection = tf.reduce_sum(g1_binary * g2_binary)
        union = tf.reduce_sum(tf.cast((g1_binary + g2_binary) > 0, tf.float32))
        jaccard = intersection / (union + 1e-8)

        # Frobenius norm of difference
        frobenius_diff = tf.norm(graph1 - graph2, ord='euclidean')

        # Sparsity difference
        sparsity1 = tf.reduce_mean(tf.cast(graph1 < threshold, tf.float32))
        sparsity2 = tf.reduce_mean(tf.cast(graph2 < threshold, tf.float32))
        sparsity_diff = tf.abs(sparsity1 - sparsity2)

        return {
            'jaccard_similarity': jaccard.numpy(),
            'frobenius_difference': frobenius_diff.numpy(),
            'sparsity_difference': sparsity_diff.numpy(),
            'connection_overlap': intersection.numpy(),
            'total_connections_1': tf.reduce_sum(g1_binary).numpy(),
            'total_connections_2': tf.reduce_sum(g2_binary).numpy()
        }

    def extract_interpretations(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Extract human-readable interpretations from analysis results.

        Args:
            analysis_results: Results from analyze_adjacency_matrix or analyze_attention_patterns

        Returns:
            interpretations: List of interpretation strings
        """
        interpretations = []

        # Graph structure interpretations
        if 'sparsity' in analysis_results:
            sparsity = analysis_results['sparsity']
            if sparsity > 0.8:
                interpretations.append(".1f")
            elif sparsity < 0.3:
                interpretations.append(".1f")

        if 'hub_layers' in analysis_results:
            hubs = analysis_results['hub_layers']
            if hubs:
                interpretations.append(f"Layers {hubs} act as communication hubs, connecting to many other layers")

        if 'num_communities' in analysis_results:
            num_comm = analysis_results['num_communities']
            if num_comm == 1:
                interpretations.append("All layers form a single connected community")
            elif num_comm > analysis_results.get('num_layers', 0) // 2:
                interpretations.append("Layers are highly modular with many small communities")

        # Attention pattern interpretations
        if 'avg_entropy' in analysis_results:
            avg_entropy = analysis_results['avg_entropy']
            if avg_entropy > 2.0:
                interpretations.append("Attention is highly diverse across layers")
            elif avg_entropy < 1.0:
                interpretations.append("Attention is concentrated on specific layers")

        if 'most_focused_layer' in analysis_results:
            focused = analysis_results['most_focused_layer']
            interpretations.append(f"Layer {focused} receives the most focused attention from other layers")

        return interpretations