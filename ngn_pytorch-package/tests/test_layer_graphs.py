import torch

from ngn_pytorch.communication import SharedAttentionAggregator
from ngn_pytorch.graph import StaticLayerGraph


def _make_layer_outputs(batch_size: int = 2):
    return [
        torch.randn(batch_size, 64, 8, 8),
        torch.randn(batch_size, 128, 4, 6),
        torch.randn(batch_size, 256, 2, 3),
        torch.randn(batch_size, 512, 1, 2),
    ]


def test_shared_attention_shapes():
    torch.manual_seed(0)
    layer_outputs = _make_layer_outputs()
    layer_graph = SharedAttentionAggregator(
        num_layers=4,
        feature_dims=[64, 128, 256, 512],
        num_heads=4,
        dropout=0.0
    )

    refined_outputs, debug_info = layer_graph(layer_outputs)

    assert len(refined_outputs) == len(layer_outputs)
    for original, refined in zip(layer_outputs, refined_outputs):
        assert original.shape == refined.shape

    gated = debug_info['gated_attention_weights']
    assert gated.shape[-2:] == (4, 4)


def test_shared_attention_adjacency_gating():
    torch.manual_seed(1)
    layer_outputs = [
        torch.randn(2, 16, 4, 5),
        torch.randn(2, 32, 3, 4),
        torch.randn(2, 64, 2, 3),
    ]
    layer_graph = SharedAttentionAggregator(
        num_layers=3,
        feature_dims=[16, 32, 64],
        num_heads=2,
        dropout=0.0
    )

    with torch.no_grad():
        layer_graph.adjacency.fill_(-10.0)
        layer_graph.adjacency.fill_diagonal_(10.0)

    _, debug_info = layer_graph(layer_outputs)
    gated = debug_info['gated_attention_weights']
    eye = torch.eye(3, device=gated.device).bool()
    off_diag = gated[..., ~eye]
    assert off_diag.mean().item() < 1e-2


def test_static_layer_graph_shapes():
    torch.manual_seed(2)
    layer_outputs = _make_layer_outputs()
    layer_graph = StaticLayerGraph(
        num_layers=4,
        feature_dims=[64, 128, 256, 512]
    )

    refined_outputs, debug_info = layer_graph(layer_outputs)
    assert debug_info['communication_type'] == 'static'
    for original, refined in zip(layer_outputs, refined_outputs):
        assert original.shape == refined.shape
