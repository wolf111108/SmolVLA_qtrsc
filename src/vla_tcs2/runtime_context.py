"""
Unified runtime context for phase-aware statistics (sparsity / workload).

Thread-safe via ContextVars; nothing here mutates model behavior. The
context is set automatically by monkey-patched hooks installed in
model_wrapper.py (sample_actions / denoise_step / attention layers), so
runners never need to call set_phase manually.

Canonical labels (research manual §5/§11):
    phase          : "prefill" | "denoise"   (fallback "unknown")
    flow_step      : -1 for prefill, 0..num_steps-1 for denoise steps
    attention_kind : "self" | "cross"        (attention layers only)
    generation_id  : monotonic id of predict_action_chunk calls
"""

from contextvars import ContextVar
from typing import Any, Dict

CURRENT_PHASE: ContextVar = ContextVar("vla_phase", default="unknown")
CURRENT_FLOW_STEP: ContextVar = ContextVar("vla_flow_step", default=-1)
CURRENT_ATTN_KIND: ContextVar = ContextVar("vla_attn_kind", default="unknown")
CURRENT_GENERATION_ID: ContextVar = ContextVar("vla_generation_id", default=-1)


def get_runtime_context() -> Dict[str, Any]:
    """Snapshot of the current runtime labels (cheap; call per collection)."""
    return {
        "phase": CURRENT_PHASE.get(),
        "flow_step": CURRENT_FLOW_STEP.get(),
        "attention_kind": CURRENT_ATTN_KIND.get(),
        "generation_id": CURRENT_GENERATION_ID.get(),
    }
