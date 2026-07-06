from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=None)
def _make_gdn_tape_capture_kernel(Dk: int, Dv: int, Hk: int, Hv: int,
                                  input_dtype: mx.Dtype, state_dtype: mx.Dtype):
    if not mx.metal.is_available():
        return None

    source = r"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // Inputs are pre-normalized by the Python caller.
        // k is shared across dv threads in a threadgroup — load it once.
        threadgroup float k_shared[Dk];
        auto local_dv_idx = thread_position_in_threadgroup.y;

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(state_ptr[n_per_t * dk_idx + i]);
        }

        for (int t = 0; t < T; ++t) {
            auto q_t    = q    + ((b_idx * T + t) * Hk + hk_idx) * Dk;
            auto k_t    = k    + ((b_idx * T + t) * Hk + hk_idx) * Dk;
            auto v_t    = v    + ((b_idx * T + t) * Hv + hv_idx) * Dv;
            auto g_t    = g    + (b_idx * T + t) * Hv;
            auto beta_t = beta + (b_idx * T + t) * Hv;

            if (local_dv_idx == 0) {
                for (int i = 0; i < n_per_t; ++i) {
                    k_shared[n_per_t * dk_idx + i] = static_cast<float>(k_t[n_per_t * dk_idx + i]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i]  = state[i] * g_t[hv_idx];
                kv_mem   += state[i] * k_shared[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx  = n_per_t * dk_idx + i;
                state[i]   = state[i] + k_shared[s_idx] * delta;
                out        += state[i] * static_cast<float>(q_t[s_idx]);
            }
            out = simd_sum(out);

            auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
            if (thread_index_in_simdgroup == 0) {
                y_t[dv_idx] = static_cast<InT>(out);
                tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx] = delta;
            }

            for (int i = 0; i < n_per_t; ++i) {
                state[i] = static_cast<float>(static_cast<StT>(state[i]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto state_out_ptr = final_state + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
            state_out_ptr[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name=f"gdn_tape_capture_v2_Dk{Dk}_Dv{Dv}_Hk{Hk}_Hv{Hv}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "final_state", "tape"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _make_gdn_tape_replay_kernel(Dk: int, Dv: int, Hk: int, Hv: int,
                                  input_dtype: mx.Dtype, state_dtype: mx.Dtype):
    if not mx.metal.is_available():
        return None

    source = r"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx       = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx       = thread_position_in_grid.y;

        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(state_ptr[n_per_t * dk_idx + i]);
        }

        for (int t = 0; t < Steps; ++t) {
            auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
            auto g_t = g + (b_idx * T + t) * Hv;

            if (local_dv_idx == 0) {
                for (int i = 0; i < n_per_t; ++i) {
                    k_shared[n_per_t * dk_idx + i] = static_cast<float>(k_t[n_per_t * dk_idx + i]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto delta = tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx];
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i]  = state[i] * g_t[hv_idx];
                state[i]  = state[i] + k_shared[s_idx] * delta;
                state[i]  = static_cast<float>(static_cast<StT>(state[i]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto out_ptr = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
            out_ptr[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name=f"gdn_tape_replay_v2_Dk{Dk}_Dv{Dv}_Hk{Hk}_Hv{Hv}",
        input_names=["tape", "k", "g", "state_in", "T", "Steps"],
        output_names=["state_out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _gdn_tape_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> Optional[Tuple[mx.array, mx.array, mx.array]]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0 or Dv % 4 != 0:
        return None
    kernel = _make_gdn_tape_capture_kernel(Dk, Dv, Hk, Hv, q.dtype, state.dtype)
    if kernel is None:
        return None
    tgy = _pick_tgy(Dv)
    return kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, T, Hv, Dv)],
        output_dtypes=[q.dtype, state.dtype, mx.float32],
    )


def _gdn_tape_replay(
    tape: mx.array,
    k: mx.array,
    g: mx.array,
    state_in: mx.array,
    steps: int,
) -> Optional[mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = state_in.shape[1], state_in.shape[2]
    if Dk % 32 != 0 or steps <= 0 or steps > T:
        return None
    kernel = _make_gdn_tape_replay_kernel(Dk, Dv, Hk, Hv, k.dtype, state_in.dtype)
    if kernel is None:
        return None
    tgy = _pick_tgy(Dv)
    (state_out,) = kernel(
        inputs=[tape, k, g, state_in, T, steps],
        template=[
            ("InT", k.dtype),
            ("StT", state_in.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[state_in.shape],
        output_dtypes=[state_in.dtype],
    )
    return state_out


def _pick_tgy(Dv: int) -> int:
    for tgy in (4, 8, 16, 32):
        if Dv % tgy == 0:
            return tgy
    return 4


def backbone_forward_with_gdn_tape(
    model: nn.Module,
    y: mx.array,
    model_cache: list,
    *,
    return_hidden: bool = False,
    n_confirmed: int = 0,
) -> Tuple:
    from .gated_delta import compute_g
    from .base import create_attention_mask, create_ssm_mask

    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        out = model(y, cache=model_cache, return_hidden=return_hidden,
                    n_confirmed=n_confirmed)
        if return_hidden:
            return out[0], out[1], {}
        return out, {}

    B, S = y.shape[:2]
    hidden_states = inner.embed_tokens(y)
    cache = model_cache

    fa_idx = getattr(inner, "fa_idx", None)
    ssm_idx = getattr(inner, "ssm_idx", None)
    fa_mask = create_attention_mask(hidden_states, cache[fa_idx] if fa_idx is not None else None)
    ssm_mask = create_ssm_mask(hidden_states, cache[ssm_idx] if ssm_idx is not None else None)

    captures: Dict[int, dict] = {}

    for layer_idx, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        normed = layer.input_layernorm(hidden_states)

        if layer.is_linear:
            r, cap = _gdn_forward_capturing(
                layer.linear_attn, normed,
                mask=mask, cache=layer_cache, n_confirmed=n_confirmed,
            )
            if cap is not None:
                captures[layer_idx] = cap
        else:
            r = layer.self_attn(normed, mask=mask, cache=layer_cache)

        h = hidden_states + r
        hidden_states = h + layer.mlp(layer.post_attention_layernorm(h))

    normed_out = inner.norm(hidden_states)
    if lm.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(normed_out)
    else:
        logits = lm.lm_head(normed_out)


    if return_hidden:
        return logits, hidden_states, captures
    return logits, captures


def _gdn_forward_capturing(gdn, inputs: mx.array, *, mask, cache, n_confirmed: int = 0):
    from .gated_delta import compute_g, gated_delta_update

    B, S, _ = inputs.shape
    T_draft = S - n_confirmed
    n_keep = gdn.conv_kernel_size - 1

    if hasattr(gdn, "fix_query_key_value_ordering"):
        q_raw, k_raw, v_full, z_full, b_full, a_full = gdn.fix_query_key_value_ordering(
            gdn.in_proj_qkvz(inputs), gdn.in_proj_ba(inputs)
        )
        mixed_qkv_full = mx.concatenate(
            [q_raw.reshape(B, S, -1), k_raw.reshape(B, S, -1), v_full.reshape(B, S, -1)],
            axis=-1,
        )
    else:
        qkv_full   = gdn.in_proj_qkv(inputs)
        z_full     = gdn.in_proj_z(inputs).reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
        b_full     = gdn.in_proj_b(inputs)
        a_full     = gdn.in_proj_a(inputs)
        mixed_qkv_full = qkv_full

    if mask is not None:
        mixed_qkv_full = mx.where(mask[..., None], mixed_qkv_full, 0)

    conv_state = cache[0] if (cache is not None and cache[0] is not None) else mx.zeros(
        (B, n_keep, gdn.conv_dim), dtype=inputs.dtype
    )
    init_ssm = cache[1] if cache else None
    if init_ssm is None:
        init_ssm = mx.zeros(
            (B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32
        )

    if n_confirmed > 0:
        conv_in1 = mx.concatenate([conv_state, mixed_qkv_full[:, :n_confirmed]], axis=1)
        conv_snap = mx.contiguous(conv_in1[:, -n_keep:, :])

        conv_out1 = nn.silu(gdn.conv1d(conv_in1))
        q1, k1, v1 = [
            t.reshape(B, n_confirmed, h, d)
            for t, h, d in zip(
                mx.split(conv_out1, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k1.shape[-1] ** -0.5
        q1n = (inv_scale ** 2) * mx.fast.rms_norm(q1, None, 1e-6)
        k1n = inv_scale * mx.fast.rms_norm(k1, None, 1e-6)
        mask1 = mask[:, :n_confirmed] if mask is not None else None

        out1, ssm_snap = gated_delta_update(
            q1n, k1n, v1, a_full[:, :n_confirmed], b_full[:, :n_confirmed],
            gdn.A_log, gdn.dt_bias, init_ssm, mask1,
            use_kernel=not gdn.training,
        )
        if cache is not None:
            cache.rollback_state = (conv_snap, ssm_snap)

        conv_state2 = conv_snap
        ssm_state2  = ssm_snap
    else:
        out1        = None
        conv_state2 = conv_state
        ssm_state2  = init_ssm

    if T_draft > 0:
        conv_in2 = mx.concatenate([conv_state2, mixed_qkv_full[:, n_confirmed:]], axis=1)

        if cache is not None:
            if cache.lengths is not None:
                full_in = mx.concatenate([conv_state, mixed_qkv_full], axis=1)
                ends = mx.clip(cache.lengths, 0, S)
                pos = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(full_in, pos, axis=1)
            else:
                cache[0] = mx.contiguous(conv_in2[:, -n_keep:, :])

        conv_out2 = nn.silu(gdn.conv1d(conv_in2))
        q2, k2, v2 = [
            t.reshape(B, T_draft, h, d)
            for t, h, d in zip(
                mx.split(conv_out2, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k2.shape[-1] ** -0.5
        q2n = (inv_scale ** 2) * mx.fast.rms_norm(q2, None, 1e-6)
        k2n = inv_scale * mx.fast.rms_norm(k2, None, 1e-6)
        g2    = compute_g(gdn.A_log, a_full[:, n_confirmed:], gdn.dt_bias)
        beta2 = mx.sigmoid(b_full[:, n_confirmed:])
        mask2 = mask[:, n_confirmed:] if mask is not None else None

        capture_result = _gdn_tape_capture(q2n, k2n, v2, g2, beta2, ssm_state2)
        if capture_result is not None:
            out2, new_state, tape = capture_result
            capture_data = {
                "k": k2n,
                "g": g2,
                "state_in": ssm_state2,
                "tape": tape,
                "conv_in2": conv_in2,
                "n_keep": n_keep,
            }
        else:
            out2, new_state = gated_delta_update(
                q2n, k2n, v2, a_full[:, n_confirmed:], b_full[:, n_confirmed:],
                gdn.A_log, gdn.dt_bias, ssm_state2, mask2,
                use_kernel=not gdn.training,
            )
            capture_data = None
    else:
        out2        = None
        new_state   = ssm_state2
        capture_data = None

    if out1 is not None and out2 is not None:
        out_raw = mx.concatenate([out1, out2], axis=1)
    elif out1 is not None:
        out_raw = out1
    else:
        out_raw = out2

    if cache is not None:
        cache[1] = new_state
        cache.advance(S)

    out = gdn.norm(out_raw, z_full)
    return gdn.out_proj(out.reshape(B, S, -1)), capture_data


def commit_gdn_state_at(
    model_cache: list,
    captures: Dict[int, dict],
    k_accepted: int,
    N: int,
) -> bool:
    if not captures or k_accepted <= 0 or k_accepted >= N:
        return False

    for layer_idx, cap in captures.items():
        if layer_idx >= len(model_cache):
            return False
        layer_cache = model_cache[layer_idx]
        if not hasattr(layer_cache, "state"):
            return False

        new_state = _gdn_tape_replay(
            cap["tape"], cap["k"], cap["g"], cap["state_in"], steps=k_accepted,
        )
        if new_state is None:
            return False

        layer_cache[1] = mx.contiguous(new_state)

        conv_in2 = cap.get("conv_in2")
        n_keep   = cap.get("n_keep")
        if conv_in2 is not None and n_keep is not None:
            layer_cache[0] = mx.contiguous(
                conv_in2[:, k_accepted : k_accepted + n_keep, :]
            )

    return True

