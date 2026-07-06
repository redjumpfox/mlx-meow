from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

MAX_M = 8

_HEADER = r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE          = 32;
    constant constexpr int PACK_FACTOR        = 8;       // nibbles per uint32
    constant constexpr int PACKS_PER_THREAD   = 2;
    constant constexpr int VALUES_PER_THREAD  = PACK_FACTOR * PACKS_PER_THREAD; // 16
    constant constexpr int BYTES_PER_PACK     = 4;
    constant constexpr int BLOCK_SIZE         = VALUES_PER_THREAD * SIMD_SIZE;  // 512
    constant constexpr int RESULTS_PER_SG     = 4;
    constant constexpr int NUM_SG             = 2;
    constant constexpr int BN                 = RESULTS_PER_SG * NUM_SG;        // 8

    template <typename T>
    inline float load_x4(const device T* x, thread float* xt) {
        float s = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            s += float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]);
            xt[i]   = float(x[i]);
            xt[i+1] = float(x[i+1]) / 16.0f;
            xt[i+2] = float(x[i+2]) / 256.0f;
            xt[i+3] = float(x[i+3]) / 4096.0f;
        }
        return s;
    }

    inline float qdot4(const device uint8_t* w,
                       const thread float*   xt,
                       float scale, float bias, float xsum) {
        const device uint16_t* ws = (const device uint16_t*)w;
        float acc = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
            uint16_t p = ws[i];
            acc += xt[4*i]   * float(p & 0x000f)
                 + xt[4*i+1] * float(p & 0x00f0)
                 + xt[4*i+2] * float(p & 0x0f00)
                 + xt[4*i+3] * float(p & 0xf000);
        }
        return scale * acc + xsum * bias;
    }

    template <typename T>
    inline T sigmoid_stable(T x) {
        T y = 1 / (1 + metal::exp(metal::abs(x)));
        return (x < T(0)) ? y : 1 - y;
    }

    template <typename T>
    inline float load_swiglu4(const device T* gate, const device T* up,
                               thread float* xt) {
        float s = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            float g0 = float(gate[i]   * sigmoid_stable(gate[i]));
            float g1 = float(gate[i+1] * sigmoid_stable(gate[i+1]));
            float g2 = float(gate[i+2] * sigmoid_stable(gate[i+2]));
            float g3 = float(gate[i+3] * sigmoid_stable(gate[i+3]));
            T sw0 = T(g0 * float(up[i]));
            T sw1 = T(g1 * float(up[i+1]));
            T sw2 = T(g2 * float(up[i+2]));
            T sw3 = T(g3 * float(up[i+3]));
            s += sw0 + sw1 + sw2 + sw3;
            xt[i]   = float(sw0);
            xt[i+1] = float(sw1) / 16.0f;
            xt[i+2] = float(sw2) / 256.0f;
            xt[i+3] = float(sw3) / 4096.0f;
        }
        return s;
    }
"""


@lru_cache(maxsize=None)
def _gate_up_dual_kernel(M: int, group_size: int, dtype: mx.Dtype):
    if not mx.metal.is_available():
        return None

    source = f"""
        uint n_tile   = threadgroup_position_in_grid.y;
        uint sg_id    = simdgroup_index_in_threadgroup;
        uint sg_lid   = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SS = GS / VALUES_PER_THREAD; // scale step per thread
        int out_row = int(n_tile) * BN + int(sg_id) * RESULTS_PER_SG;
        int stride_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int stride_g = K / GS;

        const device uint8_t* wg = (const device uint8_t*)w_gate
            + out_row * stride_w + int(sg_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sg0 = scales_gate + out_row * stride_g + int(sg_lid) / SS;
        const device T* bg0 = biases_gate + out_row * stride_g + int(sg_lid) / SS;

        const device uint8_t* wu = (const device uint8_t*)w_up
            + out_row * stride_w + int(sg_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* su0 = scales_up + out_row * stride_g + int(sg_lid) / SS;
        const device T* bu0 = biases_up + out_row * stride_g + int(sg_lid) / SS;

        float gate_res[{M}][RESULTS_PER_SG] = {{}};
        float up_res  [{M}][RESULTS_PER_SG] = {{}};
        float xt[{M}][VALUES_PER_THREAD];

        for (int k = 0; k < K; k += BLOCK_SIZE) {{
            float xsum[{M}];
            for (int m = 0; m < {M}; m++) {{
                const device T* xr = x + m * K + k + int(sg_lid) * VALUES_PER_THREAD;
                xsum[m] = load_x4(xr, xt[m]);
            }}
            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                int n = out_row + row;
                if (n < N) {{
                    float gs = float(sg0[row * stride_g]);
                    float gb = float(bg0[row * stride_g]);
                    float us = float(su0[row * stride_g]);
                    float ub = float(bu0[row * stride_g]);
                    const device uint8_t* wgl = wg + row * stride_w;
                    const device uint8_t* wul = wu + row * stride_w;
                    const device uint16_t* wgp = (const device uint16_t*)wgl;
                    const device uint16_t* wup = (const device uint16_t*)wul;
                    for (int m = 0; m < {M}; m++) {{
                        float ga = 0.0f, ua = 0.0f;
                        for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {{
                            uint16_t pg = wgp[i], pu = wup[i];
                            ga += xt[m][4*i]   * float(pg & 0x000f)
                                + xt[m][4*i+1] * float(pg & 0x00f0)
                                + xt[m][4*i+2] * float(pg & 0x0f00)
                                + xt[m][4*i+3] * float(pg & 0xf000);
                            ua += xt[m][4*i]   * float(pu & 0x000f)
                                + xt[m][4*i+1] * float(pu & 0x00f0)
                                + xt[m][4*i+2] * float(pu & 0x0f00)
                                + xt[m][4*i+3] * float(pu & 0xf000);
                        }}
                        gate_res[m][row] += gs * ga + xsum[m] * gb;
                        up_res  [m][row] += us * ua + xsum[m] * ub;
                    }}
                }}
            }}
            wg  += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
            wu  += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
            sg0 += BLOCK_SIZE / GS;
            bg0 += BLOCK_SIZE / GS;
            su0 += BLOCK_SIZE / GS;
            bu0 += BLOCK_SIZE / GS;
        }}

        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            int n = out_row + row;
            if (n < N) {{
                for (int m = 0; m < {M}; m++) {{
                    float gr = simd_sum(gate_res[m][row]);
                    float ur = simd_sum(up_res  [m][row]);
                    if (sg_lid == 0) {{
                        y_gate[m * N + n] = T(gr);
                        y_up  [m * N + n] = T(ur);
                    }}
                }}
            }}
        }}
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"gate_up_dual_v2_M{M}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_gate", "scales_gate", "biases_gate",
                     "w_up", "scales_up", "biases_up", "K_size", "N_size"],
        output_names=["y_gate", "y_up"],
        source=source,
        header=_HEADER,
    )


@lru_cache(maxsize=None)
def _swiglu_down_kernel(M: int, group_size: int, dtype: mx.Dtype):
    if not mx.metal.is_available():
        return None

    source = f"""
        uint n_tile = threadgroup_position_in_grid.y;
        uint sg_id  = simdgroup_index_in_threadgroup;
        uint sg_lid = thread_index_in_simdgroup;

        int K = int(K_size);
        int N = int(N_size);
        constexpr int SS = GS / VALUES_PER_THREAD;
        int out_row  = int(n_tile) * BN + int(sg_id) * RESULTS_PER_SG;
        int stride_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int stride_g = K / GS;

        const device uint8_t* wd  = (const device uint8_t*)w_down
            + out_row * stride_w + int(sg_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sd0 = scales_down + out_row * stride_g + int(sg_lid) / SS;
        const device T* bd0 = biases_down + out_row * stride_g + int(sg_lid) / SS;

        float res[{M}][RESULTS_PER_SG] = {{}};
        float xt[{M}][VALUES_PER_THREAD];

        for (int k = 0; k < K; k += BLOCK_SIZE) {{
            float xsum[{M}];
            for (int m = 0; m < {M}; m++) {{
                const device T* gp = gate_act + m * K + k + int(sg_lid) * VALUES_PER_THREAD;
                const device T* up = up_act   + m * K + k + int(sg_lid) * VALUES_PER_THREAD;
                xsum[m] = load_swiglu4(gp, up, xt[m]);
            }}
            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                int n = out_row + row;
                if (n < N) {{
                    float sc = float(sd0[row * stride_g]);
                    float bc = float(bd0[row * stride_g]);
                    const device uint8_t* wdl = wd + row * stride_w;
                    const device uint16_t* wdp = (const device uint16_t*)wdl;
                    for (int m = 0; m < {M}; m++) {{
                        float acc = 0.0f;
                        for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {{
                            uint16_t p = wdp[i];
                            acc += xt[m][4*i]   * float(p & 0x000f)
                                 + xt[m][4*i+1] * float(p & 0x00f0)
                                 + xt[m][4*i+2] * float(p & 0x0f00)
                                 + xt[m][4*i+3] * float(p & 0xf000);
                        }}
                        res[m][row] += sc * acc + xsum[m] * bc;
                    }}
                }}
            }}
            wd  += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
            sd0 += BLOCK_SIZE / GS;
            bd0 += BLOCK_SIZE / GS;
        }}

        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            int n = out_row + row;
            if (n < N) {{
                for (int m = 0; m < {M}; m++) {{
                    float r = simd_sum(res[m][row]);
                    if (sg_lid == 0) {{
                        y[m * N + n] = T(r);
                    }}
                }}
            }}
        }}
    """
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"swiglu_down_v2_M{M}_gs{group_size}_{dtype_tag}",
        input_names=["gate_act", "up_act",
                     "w_down", "scales_down", "biases_down", "K_size", "N_size"],
        output_names=["y"],
        source=source,
        header=_HEADER,
    )


def _is_eligible_q4_linear(m: Any) -> bool:
    return (
        isinstance(m, nn.QuantizedLinear)
        and int(getattr(m, "bits", 0)) == 4
        and int(getattr(m, "group_size", 0)) in {32, 64, 128}
        and str(getattr(m, "mode", "affine")) == "affine"
        and getattr(m, "biases", None) is not None
        and "bias" not in m
    )


def _mlp_eligible(gate_proj, up_proj, down_proj) -> bool:
    if not all(_is_eligible_q4_linear(p) for p in (gate_proj, up_proj, down_proj)):
        return False
    if gate_proj.weight.shape != up_proj.weight.shape:
        return False
    if gate_proj.group_size != up_proj.group_size:
        return False
    K_packed = int(gate_proj.weight.shape[1])
    K = K_packed * 8
    N = int(gate_proj.weight.shape[0])
    N_down = int(down_proj.weight.shape[0])
    return K % 512 == 0 and N % 8 == 0 and N_down % 8 == 0


def fused_mlp_forward(gate_proj, up_proj, down_proj, x: mx.array) -> mx.array:
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    M = x_flat.shape[0]

    if (
        not mx.metal.is_available()
        or M < 2 or M > MAX_M
        or x_flat.dtype not in (mx.bfloat16, mx.float16)
        or not _mlp_eligible(gate_proj, up_proj, down_proj)
    ):
        return _stock_mlp(gate_proj, up_proj, down_proj, x)

    K = x_flat.shape[1]
    N_int = int(gate_proj.weight.shape[0])
    N_hid = int(down_proj.weight.shape[0])
    GS = int(gate_proj.group_size)
    dtype = x_flat.dtype

    if gate_proj.scales.dtype != dtype or gate_proj.biases.dtype != dtype:
        return _stock_mlp(gate_proj, up_proj, down_proj, x)

    k_dual = _gate_up_dual_kernel(M, GS, dtype)
    k_down = _swiglu_down_kernel(M, GS, dtype)
    if k_dual is None or k_down is None:
        return _stock_mlp(gate_proj, up_proj, down_proj, x)

    n_tg_int = (N_int + 7) // 8
    y_gate, y_up = k_dual(
        inputs=[x_flat,
                gate_proj.weight, gate_proj.scales, gate_proj.biases,
                up_proj.weight,   up_proj.scales,   up_proj.biases,
                K, N_int],
        template=[("T", dtype), ("GS", GS)],
        grid=(32, 2 * n_tg_int, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(M, N_int), (M, N_int)],
        output_dtypes=[dtype, dtype],
    )

    n_tg_hid = (N_hid + 7) // 8
    (y,) = k_down(
        inputs=[y_gate, y_up,
                down_proj.weight, down_proj.scales, down_proj.biases,
                N_int, N_hid],
        template=[("T", dtype), ("GS", GS)],
        grid=(32, 2 * n_tg_hid, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(M, N_hid)],
        output_dtypes=[dtype],
    )

    return y.reshape(*orig_shape[:-1], N_hid)


def _stock_mlp(gate_proj, up_proj, down_proj, x):
    from mlx_lm.models.qwen3_next import swiglu
    return down_proj(swiglu(gate_proj(x), up_proj(x)))


def patch_model_mlp_fused(model: nn.Module) -> None:
    if not mx.metal.is_available():
        return
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP
    except ImportError:
        return

    if getattr(Qwen3NextMLP, "_mlp_fuse_patched", False):
        return

    original_call = Qwen3NextMLP.__call__

    def _patched_call(self, x: mx.array) -> mx.array:
        return fused_mlp_forward(self.gate_proj, self.up_proj, self.down_proj, x)

    Qwen3NextMLP.__call__ = _patched_call
    Qwen3NextMLP._mlp_fuse_patched = True
