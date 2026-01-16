"""
NGN Utils - Profiling Tools

This module provides profiling utilities for NGN models,
measuring memory usage, compute overhead, and latency.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import nn, Tensor
import time
import psutil
from contextlib import contextmanager
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import importlib
    GPUtil = importlib.import_module('GPUtil')
    HAS_GPUTIL = True
except Exception:
    GPUtil = None
    HAS_GPUTIL = False
    logger.warning("GPUtil not available. GPU profiling will be limited.")


class NGNProfiler:

    def profile_per_layer_timing(self, model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 5) -> Dict[str, Any]:
        """
        Profile per-layer forward timing for the model.
        Placeholder for future implementation.
        """
        logger.info("profile_per_layer_timing is a placeholder (not implemented)")
        return {}

    def profile_memory_breakdown(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Profile memory usage breakdown by layer/module.
        Placeholder for future implementation.
        """
        logger.info("profile_memory_breakdown is a placeholder (not implemented)")
        return {}

    def advanced_flop_analysis(self, model: nn.Module) -> Dict[str, Any]:
        """
        Advanced FLOP analysis (e.g., per-layer, per-operation).
        Placeholder for future implementation.
        """
        logger.info("advanced_flop_analysis is a placeholder (not implemented)")
        return {}

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.baseline_stats = {}

    @contextmanager
    def profile_context(self, name: str):
        """
        Context manager for profiling code blocks.

        Usage:
            with profiler.profile_context('forward_pass'):
                output = model(input)
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.info(f"Profile '{name}': {duration:.4f}s, {memory_delta:.2f}MB memory")

            if name not in self.baseline_stats:
                self.baseline_stats[name] = []

            self.baseline_stats[name].append({
                'duration': duration,
                'memory_delta': memory_delta,
                'timestamp': time.time()
            })

    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Comprehensive model profiling.

        Args:
            model: Model to profile
            input_shape: Input tensor shape (without batch dim)
            num_runs: Number of profiling runs for averaging

        Returns:
            profile: Dictionary with profiling results
        """
        model.eval()
        device = next(model.parameters()).device

        # Warmup
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape).to(device)
            for _ in range(3):
                _ = model(dummy_input)

        # Profile runs
        latencies = []
        memory_usages = []
        gpu_usages = []

        with torch.no_grad():
            for _ in range(num_runs):
                dummy_input = torch.randn(1, *input_shape).to(device)

                # Memory before
                mem_before = self._get_memory_usage()

                # Time forward pass
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()

                output, debug_info = model(dummy_input)

                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()

                # Memory after
                mem_after = self._get_memory_usage()

                latencies.append(end_time - start_time)
                memory_usages.append(mem_after - mem_before)

                # GPU utilization
                if device.type == 'cuda':
                    if HAS_GPUTIL:
                        gpu_usage = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
                    else:
                        gpu_usage = 0.0  # GPU monitoring not available
                    gpu_usages.append(gpu_usage)

        # Calculate statistics
        profile = {
            'avg_latency_ms': np.mean(latencies) * 1000,
            'std_latency_ms': np.std(latencies) * 1000,
            'min_latency_ms': np.min(latencies) * 1000,
            'max_latency_ms': np.max(latencies) * 1000,
            'avg_memory_mb': np.mean(memory_usages),
            'peak_memory_mb': np.max(memory_usages),
            'throughput_samples_per_sec': 1.0 / np.mean(latencies),
        }

        if gpu_usages:
            profile['avg_gpu_utilization'] = np.mean(gpu_usages)

        # Model statistics
        profile.update(self._get_model_stats(model))

        return profile

    def profile_ngn_overhead(
        self,
        baseline_model: nn.Module,
        ngn_model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Profile NGN overhead compared to baseline.

        Args:
            baseline_model: Baseline model without NGN
            ngn_model: NGN-enhanced model
            input_shape: Input tensor shape
            num_runs: Number of profiling runs

        Returns:
            overhead: Dictionary with overhead analysis
        """
        baseline_profile = self.profile_model(baseline_model, input_shape, num_runs)
        ngn_profile = self.profile_model(ngn_model, input_shape, num_runs)

        overhead = {
            'latency_overhead_percent': (
                (ngn_profile['avg_latency_ms'] - baseline_profile['avg_latency_ms']) /
                baseline_profile['avg_latency_ms'] * 100
            ),
            'memory_overhead_percent': (
                (ngn_profile['avg_memory_mb'] - baseline_profile['avg_memory_mb']) /
                baseline_profile['avg_memory_mb'] * 100
            ),
            'parameter_overhead_percent': (
                (ngn_profile['total_params'] - baseline_profile['total_params']) /
                baseline_profile['total_params'] * 100
            ),
            'throughput_overhead_percent': (
                (baseline_profile['throughput_samples_per_sec'] - ngn_profile['throughput_samples_per_sec']) /
                baseline_profile['throughput_samples_per_sec'] * 100
            ),
            'baseline_profile': baseline_profile,
            'ngn_profile': ngn_profile
        }

        # Assessment
        if overhead['latency_overhead_percent'] < 5.0:
            overhead['assessment'] = 'Excellent: <5% overhead'
        elif overhead['latency_overhead_percent'] < 15.0:
            overhead['assessment'] = 'Good: <15% overhead'
        elif overhead['latency_overhead_percent'] < 30.0:
            overhead['assessment'] = 'Acceptable: <30% overhead'
        else:
            overhead['assessment'] = 'High overhead: >30%'

        return overhead

    def profile_layer_communication(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        Profile communication overhead in layer graph.

        Args:
            model: NGN model to profile
            input_shape: Input tensor shape

        Returns:
            comm_profile: Communication profiling results
        """
        if not hasattr(model, 'layer_graph'):
            return {'error': 'Model does not have layer_graph attribute'}

        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape).to(device)

        # Hook to measure communication time
        communication_times = []

        def communication_hook(module, input, output):
            start = time.time()
            result = module(*input)
            end = time.time()
            communication_times.append(end - start)
            return result

        # Register hook on layer graph
        hook_handle = model.layer_graph.register_forward_hook(communication_hook)

        try:
            with torch.no_grad():
                _ = model(dummy_input)
        finally:
            hook_handle.remove()

        if communication_times:
            avg_comm_time = np.mean(communication_times) * 1000  # ms
            total_comm_time = np.sum(communication_times) * 1000  # ms
        else:
            avg_comm_time = 0.0
            total_comm_time = 0.0

        return {
            'avg_communication_time_ms': avg_comm_time,
            'total_communication_time_ms': total_comm_time,
            'num_communication_calls': len(communication_times)
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def _get_model_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get model statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate FLOPs (rough approximation)
        flops = self._estimate_flops(model)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'estimated_flops': flops,
            'model_size_mb': total_params * 4 / 1024 / 1024  # Rough estimate
        }

    def _estimate_flops(self, model: nn.Module) -> int:
        """Rough FLOP estimation for the model."""
        # This is a very rough approximation
        total_flops = 0

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Conv FLOPs: output_elements * kernel_size
                output_elements = (
                    module.out_channels *
                    (224 // module.stride[0]) *  # Assuming 224x224 input
                    (224 // module.stride[1])
                )
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                total_flops += output_elements * kernel_flops

            elif isinstance(module, nn.Linear):
                # Linear FLOPs: in_features * out_features
                total_flops += module.in_features * module.out_features

        return total_flops

    def create_profiling_report(
        self,
        profile_results: Dict[str, Any],
        output_file: str = 'profiling_report.txt'
    ):
        """
        Create a human-readable profiling report.

        Args:
            profile_results: Results from profiling functions
            output_file: Output file path
        """
        report_lines = [
            "NGN Profiling Report",
            "=" * 50,
            "",
        ]

        if 'avg_latency_ms' in profile_results:
            report_lines.extend([
                "Performance Metrics:",
                ".2f",
                ".2f",
                ".2f",
                ".2f",
                "",
            ])

        if 'avg_memory_mb' in profile_results:
            report_lines.extend([
                "Memory Usage:",
                ".2f",
                ".2f",
                "",
            ])

        if 'total_params' in profile_results:
            report_lines.extend([
                "Model Statistics:",
                f"Total Parameters: {profile_results['total_params']:,}",
                f"Trainable Parameters: {profile_results['trainable_params']:,}",
                ".2e",
                ".2f",
                "",
            ])

        if 'latency_overhead_percent' in profile_results:
            report_lines.extend([
                "NGN Overhead Analysis:",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
                "",
                f"Assessment: {profile_results.get('assessment', 'Unknown')}",
            ])

        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Profiling report saved to {output_file}")


# Utility functions
def benchmark_models(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...],
    num_runs: int = 10,
    device: str = 'cuda'
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark multiple models.

    Args:
        models: Dictionary of model names to models
        input_shape: Input tensor shape
        num_runs: Number of benchmarking runs
        device: Device to run on

    Returns:
        benchmarks: Dictionary of benchmark results per model
    """
    profiler = NGNProfiler(device)
    benchmarks = {}

    for name, model in models.items():
        logger.info(f"Benchmarking {name}...")
        benchmarks[name] = profiler.profile_model(model, input_shape, num_runs)

    return benchmarks


def compare_ngn_to_baseline(
    baseline_model: nn.Module,
    ngn_model: nn.Module,
    input_shape: Tuple[int, ...],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare NGN model to baseline.

    Args:
        baseline_model: Baseline model
        ngn_model: NGN-enhanced model
        input_shape: Input tensor shape
        output_file: Optional output file for report

    Returns:
        comparison: Comparison results
    """
    profiler = NGNProfiler()

    comparison = profiler.profile_ngn_overhead(
        baseline_model, ngn_model, input_shape
    )

    if output_file:
        profiler.create_profiling_report(comparison, output_file)

    return comparison