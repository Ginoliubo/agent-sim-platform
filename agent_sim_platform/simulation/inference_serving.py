"""Inference serving simulation engine with continuous batching.

Uses a discrete-event simulation for performance:
- request arrivals
- prefill batch completions
- decode token completions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .. import config as sim_config
from ..config import FEASIBILITY_MAX_MEMORY_UTIL
from ..data_models import (
    AgentHarnessSpec,
    HardwareSpec,
    InferenceServiceConfig,
    ModelSpec,
    OptimizationConfig,
    SimulationConfig,
    SimulationResult,
    WorkloadSpec,
)
from ..utils.units import gb_to_bytes


@dataclass
class Request:
    """A single inference request."""

    id: int
    arrival_time: float
    input_len: int
    output_len: int
    start_prefill_time: float = -1.0
    first_token_time: float = -1.0
    completion_time: float = -1.0
    generated_tokens: int = 0
    dropped: bool = False

    @property
    def ttft(self) -> float:
        if self.first_token_time < 0:
            return -1.0
        return self.first_token_time - self.arrival_time

    @property
    def total_latency(self) -> float:
        if self.completion_time < 0:
            return -1.0
        return self.completion_time - self.arrival_time


class InferenceServingEngine:
    """Simulate an inference serving system with continuous batching."""

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareSpec,
        service_config: InferenceServiceConfig,
        precision: str = "FP8",
        kv_precision: str = "FP8",
        optimization: OptimizationConfig = None,
        gpu_count: int = 1,
        seed: int = 42,
    ):
        self.model = model
        self.hardware = hardware
        self.service_config = service_config
        self.precision = precision.upper()
        self.kv_precision = kv_precision.upper()
        self.optimization = optimization or OptimizationConfig()
        self.gpu_count = gpu_count
        self.rng = np.random.default_rng(seed)

        self.bytes_per_param = model.bytes_per_param(self.precision)
        self.kv_bytes_per_token = model.kv_bytes_per_token(self.kv_precision)
        self.weight_memory_gb = model.weight_memory_gb(self.precision)

        self.prefill_time_per_token = self._compute_prefill_time_per_token()
        self.decode_time_per_token = self._compute_decode_time_per_token()

    def _compute_prefill_time_per_token(self) -> float:
        """Time to process one token during prefill (compute-bound)."""
        flops_per_token = self.model.flops_per_token_forward()
        effective_flops = (
            self.hardware.effective_flops(self.precision, sim_config.DEFAULT_PREFILL_UTILIZATION)
            * self.gpu_count
        )
        return flops_per_token / effective_flops

    def _compute_decode_time_per_token(self) -> float:
        """Time to generate one token during decode (memory-bound)."""
        flops_per_token = self.model.flops_per_token_forward()
        effective_compute = (
            self.hardware.effective_flops(self.precision, sim_config.DEFAULT_DECODE_UTILIZATION)
            * self.gpu_count
        )
        compute_time = flops_per_token / effective_compute

        active_params = self.model.active_params_b * 1e9
        bytes_per_token = (
            active_params * self.bytes_per_param + self.kv_bytes_per_token
        )
        effective_bw = self.hardware.memory_bw_bytes_s() * sim_config.DEFAULT_DECODE_UTILIZATION * self.gpu_count
        memory_time = bytes_per_token / effective_bw

        return max(compute_time, memory_time)

    def _generate_requests(self) -> List[Request]:
        """Generate a sequence of requests based on arrival distribution."""
        cfg = self.service_config
        duration = cfg.simulation_duration_seconds

        if cfg.arrival_distribution == "poisson":
            n_requests = int(cfg.arrival_rate_per_sec * duration * 2)
            inter_arrivals = self.rng.exponential(1.0 / cfg.arrival_rate_per_sec, size=n_requests)
            arrival_times = np.cumsum(inter_arrivals)
            arrival_times = arrival_times[arrival_times <= duration]
        elif cfg.arrival_distribution == "fixed":
            n_requests = int(cfg.arrival_rate_per_sec * duration)
            arrival_times = np.linspace(0, duration, n_requests, endpoint=False)
        else:
            n_requests = int(cfg.arrival_rate_per_sec * duration)
            inter_arrivals = self.rng.exponential(1.0 / cfg.arrival_rate_per_sec, size=n_requests)
            arrival_times = np.cumsum(inter_arrivals)
            arrival_times = arrival_times[arrival_times <= duration]

        requests = []
        for i, t in enumerate(arrival_times):
            input_len = max(1, int(self.rng.normal(cfg.request_length_mean, cfg.request_length_std)))
            output_len = max(1, int(self.rng.normal(cfg.output_length_mean, cfg.output_length_std)))
            requests.append(Request(id=i, arrival_time=t, input_len=input_len, output_len=output_len))
        return requests

    def _memory_for_batch(self, requests: List[Request]) -> float:
        """Estimate memory (GB) required for a set of active requests."""
        total_tokens = sum(r.input_len + r.generated_tokens for r in requests)
        kv_gb = total_tokens * self.kv_bytes_per_token / gb_to_bytes(1.0)
        return self.weight_memory_gb + kv_gb

    def _next_arrival_time(self, arrivals: List[Request], idx: int) -> float:
        if idx < len(arrivals):
            return arrivals[idx].arrival_time
        return float("inf")

    def _next_prefill_completion(self, prefill_batch: List[Request]) -> float:
        if not prefill_batch:
            return float("inf")
        # All requests in prefill batch share the accelerator; total prefill time
        total_tokens = sum(r.input_len for r in prefill_batch)
        return total_tokens * self.prefill_time_per_token

    def _next_decode_completion(self, decode_batch: List[Request]) -> float:
        if not decode_batch:
            return float("inf")
        # Time until next token is generated for the whole batch
        return len(decode_batch) * self.decode_time_per_token

    def run(self) -> SimulationResult:
        """Run event-driven inference serving simulation."""
        cfg = self.service_config
        arrivals = self._generate_requests()
        arrival_idx = 0

        queue: List[Request] = []
        prefill_batch: List[Request] = []
        decode_batch: List[Request] = []
        completed: List[Request] = []
        dropped = 0

        current_time = 0.0
        prefill_remaining_time = 0.0  # time left to finish current prefill batch
        decode_remaining_time = 0.0   # time left to generate next token for decode batch

        max_queue_len = cfg.max_queue_len
        max_batch_size = cfg.max_batch_size

        while (
            arrival_idx < len(arrivals)
            or queue
            or prefill_batch
            or decode_batch
        ):
            # Admit new arrivals
            while arrival_idx < len(arrivals) and arrivals[arrival_idx].arrival_time <= current_time:
                req = arrivals[arrival_idx]
                if len(queue) >= max_queue_len:
                    req.dropped = True
                    dropped += 1
                else:
                    queue.append(req)
                arrival_idx += 1

            # Fill prefill batch from queue
            while (
                queue
                and len(prefill_batch) + len(decode_batch) < max_batch_size
            ):
                candidate = queue[0]
                test_batch = prefill_batch + decode_batch + [candidate]
                if (
                    self._memory_for_batch(test_batch)
                    <= self.hardware.memory_gb * self.gpu_count * FEASIBILITY_MAX_MEMORY_UTIL
                ):
                    req = queue.pop(0)
                    req.start_prefill_time = current_time
                    prefill_batch.append(req)
                else:
                    break

            # Determine next event time
            next_arrival = self._next_arrival_time(arrivals, arrival_idx)
            next_prefill = prefill_remaining_time if prefill_batch else float("inf")
            next_decode = decode_remaining_time if decode_batch else float("inf")

            next_event_time = min(next_arrival, next_prefill, next_decode)
            if next_event_time == float("inf"):
                break

            # Advance time
            advance = next_event_time
            current_time += advance

            if prefill_batch:
                prefill_remaining_time -= advance
            if decode_batch:
                decode_remaining_time -= advance

            # Handle decode completion
            if decode_batch and decode_remaining_time <= 1e-9:
                for req in decode_batch:
                    req.generated_tokens += 1
                # Move completed requests out
                still_decoding = []
                for req in decode_batch:
                    if req.generated_tokens >= req.output_len:
                        req.completion_time = current_time
                        completed.append(req)
                    else:
                        still_decoding.append(req)
                decode_batch = still_decoding
                if decode_batch:
                    decode_remaining_time = self._next_decode_completion(decode_batch)

            # Handle prefill completion
            if prefill_batch and prefill_remaining_time <= 1e-9:
                for req in prefill_batch:
                    req.first_token_time = current_time
                    decode_batch.append(req)
                prefill_batch = []
                if decode_batch:
                    decode_remaining_time = self._next_decode_completion(decode_batch)

            # Reset prefill_remaining_time if batch changed
            if prefill_batch and prefill_remaining_time <= 1e-9:
                prefill_remaining_time = self._next_prefill_completion(prefill_batch)

        # Compute metrics
        completed_reqs = [r for r in completed if not r.dropped]
        ttfts = [r.ttft for r in completed_reqs if r.ttft >= 0]
        tpots = [
            (r.total_latency - r.ttft) / r.output_len
            for r in completed_reqs
            if r.total_latency >= 0 and r.output_len > 0
        ]
        total_latencies = [r.total_latency for r in completed_reqs if r.total_latency >= 0]

        total_tokens = sum(r.input_len + r.output_len for r in completed_reqs)
        duration = cfg.simulation_duration_seconds

        throughput_req = len(completed_reqs) / duration if duration > 0 else 0.0
        throughput_tok = total_tokens / duration if duration > 0 else 0.0

        # Approximate utilization
        total_decode_tokens = sum(r.output_len for r in completed_reqs)
        busy_time = (
            sum(r.input_len for r in completed_reqs) * self.prefill_time_per_token
            + total_decode_tokens * self.decode_time_per_token
        )
        utilization = min(1.0, busy_time / (duration * self.gpu_count)) if duration > 0 else 0.0

        cost_usd = (duration / 3600.0) * self.hardware.cost_per_hour * self.gpu_count

        sim_config = SimulationConfig(
            hardware=self.hardware,
            model=self.model,
            workload=WorkloadSpec(
                name="inference-serving",
                max_steps=1,
                avg_steps=1.0,
                step_std=0.0,
                context_limit=cfg.request_length_mean * 2,
            ),
            harness=AgentHarnessSpec(name="inference-serving", concurrency=1),
            target_context_tokens=cfg.request_length_mean,
            precision=self.precision,
            kv_precision=self.kv_precision,
            optimization=self.optimization,
        )

        def _p(values, p):
            if not values:
                return 0.0
            return float(np.percentile(values, p))

        return SimulationResult(
            config=sim_config,
            latency_seconds=duration,
            wall_time_seconds=duration,
            tokens_total=total_tokens,
            tokens_input=sum(r.input_len for r in completed_reqs),
            tokens_output=sum(r.output_len for r in completed_reqs),
            peak_kv_gb=max(
                (self._memory_for_batch(completed_reqs) - self.weight_memory_gb), 0.0
            ),
            memory_required_gb=self.hardware.memory_gb * self.gpu_count,
            gpu_count=self.gpu_count,
            feasible=True,
            bottleneck="",
            cost_usd=cost_usd,
            utilization_gpu=utilization,
            metadata={
                "requests_total": len(arrivals),
                "requests_completed": len(completed_reqs),
                "requests_dropped": dropped,
                "throughput_req_per_sec": throughput_req,
                "throughput_tok_per_sec": throughput_tok,
                "ttft_p50_ms": _p(ttfts, 50) * 1000,
                "ttft_p99_ms": _p(ttfts, 99) * 1000,
                "tpot_p50_ms": _p(tpots, 50) * 1000,
                "tpot_p99_ms": _p(tpots, 99) * 1000,
                "e2e_latency_p99_ms": _p(total_latencies, 99) * 1000,
                "prefill_time_per_token_ms": self.prefill_time_per_token * 1000,
                "decode_time_per_token_ms": self.decode_time_per_token * 1000,
            },
        )


def run_serving(
    model: ModelSpec,
    hardware: HardwareSpec,
    service_config: InferenceServiceConfig,
    precision: str = "FP8",
    kv_precision: str = "FP8",
    optimization: OptimizationConfig = None,
    gpu_count: int = 1,
    seed: int = 42,
) -> SimulationResult:
    """Convenience function to run inference serving simulation."""
    return InferenceServingEngine(
        model, hardware, service_config, precision, kv_precision, optimization, gpu_count, seed
    ).run()
