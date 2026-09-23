"""Opt-in SmolVLM vision attention adapter (eager/SDPA, inference PTQ).

Matches transformers 4.52.4 SmolVLMVisionAttention's eager computation.
QK output scales describe unscaled QK; head scaling and masking follow it.
No global transformers attention registry or shared config is modified.
"""
from types import MethodType

import torch
import torch.nn.functional as F

from vla_tcs2.runtime_context import CURRENT_ATTN_KIND


def vision_attention_forward(self, hidden_states, attention_mask=None,
                             output_attentions=False):
    batch, length, width = hidden_states.shape
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool or attention_mask.ndim != 4:
            raise ValueError("Vision attention requires a 4D additive float mask")
        if attention_mask.shape[-2:] != (length, length):
            raise ValueError("Vision attention mask must match the token sequence")
    token = CURRENT_ATTN_KIND.set("self")
    try:
        def heads(proj):
            return proj(hidden_states).view(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)
        query, key, value = heads(self.q_proj), heads(self.k_proj), heads(self.v_proj)
        scores = self.quant_qk(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        output = self.quant_pv(probs, value).transpose(1, 2).contiguous()
        output = self.out_proj(output.reshape(batch, length, width))
        return output, probs if output_attentions else None
    finally:
        CURRENT_ATTN_KIND.reset(token)


def attach_vision_attention(attention, qk, pv):
    """Register two real child modules and replace only this instance's forward."""
    required = ("q_proj", "k_proj", "v_proj", "out_proj", "num_heads",
                "head_dim", "scale", "dropout", "config")
    if any(not hasattr(attention, name) for name in required):
        raise TypeError("Unsupported SmolVLM vision attention layout")
    backend = attention.config._attn_implementation
    if backend not in ("eager", "sdpa") or getattr(attention, "is_causal", False):
        raise ValueError(f"Unsupported vision attention backend: {backend}")
    if hasattr(attention, "quant_qk") or hasattr(attention, "quant_pv"):
        raise RuntimeError("Vision attention has already been quantized")
    attention._original_vision_forward = attention.forward
    attention.quant_qk = qk
    attention.quant_pv = pv
    attention.forward = MethodType(vision_attention_forward, attention)
