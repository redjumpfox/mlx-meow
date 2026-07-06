import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def _median_ms(fn, warmup=5, reps=40):
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def bench_mlp_fused(hidden: int, intermediate: int, group_size: int = 64):
    from mlx_lm.models.mlp_fused import fused_mlp_forward, _is_eligible_q4_linear

    def _qlinear(out_dim, in_dim):
        layer = nn.QuantizedLinear(in_dim, out_dim, bias=False,
                                   group_size=group_size, bits=4)
        pf = 8
        ng = in_dim // group_size
        layer.weight = mx.random.randint(0, 2**32 - 1,
                                         (out_dim, in_dim // pf), dtype=mx.uint32)
        layer.scales = (mx.random.normal((out_dim, ng)) * 0.01).astype(mx.bfloat16)
        layer.biases = mx.random.normal((out_dim, ng)).astype(mx.bfloat16)
        return layer

    gate  = _qlinear(intermediate, hidden)
    up    = _qlinear(intermediate, hidden)
    down  = _qlinear(hidden, intermediate)
    mx.eval(gate.parameters(), up.parameters(), down.parameters())

    from mlx_lm.models.qwen3_next import swiglu

    def stock(x):
        return down(swiglu(gate(x), up(x)))

    def fused(x):
        return fused_mlp_forward(gate, up, down, x)

    print(f"\nMLP fusion benchmark  hidden={hidden} intermediate={intermediate} q4 gs={group_size}")
    print(f"{'M':>4}  {'stock ms':>10}  {'fused ms':>10}  {'speedup':>8}  {'match':>6}")
    print("-" * 48)

    all_ok = True
    for M in range(1, 9):
        x = mx.random.normal((M, hidden)).astype(mx.bfloat16)
        mx.eval(x)

        ref = stock(x)
        out = fused(x)
        mx.eval(ref, out)
        # Tolerance: 2× bfloat16 ULP at the maximum value magnitude.
        # bf16 has 7 mantissa bits → ULP ≈ |max| / 128.
        max_val = float(mx.max(mx.abs(ref.astype(mx.float32))).item())
        tol = max(2.0, 2.0 * max_val / 128)
        max_err = mx.max(mx.abs(ref.astype(mx.float32) - out.astype(mx.float32))).item()
        ok = max_err < tol
        if not ok:
            all_ok = False

        t_stock = _median_ms(lambda: stock(x))
        t_fused = _median_ms(lambda: fused(x))
        speedup = t_stock / t_fused
        marker  = " <--" if M >= 2 and speedup >= 1.10 else ""
        print(f"{M:>4}  {t_stock:>10.3f}  {t_fused:>10.3f}  {speedup:>8.2f}x  {'OK' if ok else 'FAIL':>6}{marker}")

    if all_ok:
        print("All M correctness OK")
    else:
        print("WARNING: Some M values produced wrong results")
    return all_ok


def bench_gdn_tape(B: int, T: int, Hk: int, Hv: int, Dk: int, Dv: int):
    from mlx_lm.models.gdn_tape import _gdn_tape_capture, _gdn_tape_replay
    from mlx_lm.models.gated_delta import gated_delta_kernel

    if not mx.metal.is_available():
        print("Metal not available; GDN tape benchmark skipped.")
        return True

    dtype = mx.bfloat16
    state_dtype = mx.float32

    q     = mx.random.normal((B, T, Hk, Dk)).astype(dtype)
    k     = mx.random.normal((B, T, Hk, Dk)).astype(dtype)
    v     = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
    g     = mx.abs(mx.random.normal((B, T, Hv)).astype(dtype)) * 0.1 + 0.9
    beta  = mx.sigmoid(mx.random.normal((B, T, Hv)).astype(dtype))
    state = mx.zeros((B, Hv, Dv, Dk), dtype=state_dtype)
    mx.eval(q, k, v, g, beta, state)

    print(f"\nGDN tape benchmark  B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}")

    ref_y, ref_state = gated_delta_kernel(q, k, v, g, beta, state)
    mx.eval(ref_y, ref_state)

    result = _gdn_tape_capture(q, k, v, g, beta, state)
    if result is None:
        print("  Capture kernel: unavailable (shapes incompatible)")
        return False
    y_cap, final_state, tape = result
    mx.eval(y_cap, final_state, tape)

    err_y = mx.max(mx.abs(ref_y.astype(mx.float32) - y_cap.astype(mx.float32))).item()
    err_s = mx.max(mx.abs(ref_state.astype(mx.float32) - final_state.astype(mx.float32))).item()
    print(f"  Capture output error: {err_y:.5f}  state error: {err_s:.5f}  "
          f"{'OK' if err_y < 0.1 and err_s < 0.1 else 'WARN'}")

    print(f"  Replay correctness per step:")
    running_state = state
    all_replay_ok = True
    for step in range(1, T + 1):
        _, running_state = gated_delta_kernel(
            q[:, step-1:step], k[:, step-1:step], v[:, step-1:step],
            g[:, step-1:step], beta[:, step-1:step], running_state
        )
        mx.eval(running_state)

        replayed = _gdn_tape_replay(tape, k, g, state, steps=step)
        if replayed is None:
            print(f"    step={step}: replay kernel unavailable")
            all_replay_ok = False
            break
        mx.eval(replayed)
        err = mx.max(mx.abs(running_state.astype(mx.float32) -
                             replayed.astype(mx.float32))).item()
        ok = err < 0.1
        if not ok:
            all_replay_ok = False
        print(f"    step={step}: err={err:.5f}  {'OK' if ok else 'FAIL'}")

    t_stock = _median_ms(lambda: gated_delta_kernel(q, k, v, g, beta, state))
    t_cap   = _median_ms(lambda: _gdn_tape_capture(q, k, v, g, beta, state))
    print(f"  Timing: stock={t_stock:.3f}ms  capture={t_cap:.3f}ms "
          f"overhead={((t_cap/t_stock)-1)*100:.1f}%")

    mid = T // 2
    t_replay = _median_ms(lambda: _gdn_tape_replay(tape, k, g, state, steps=mid))
    print(f"  Replay {mid} steps: {t_replay:.3f}ms")

    return all_replay_ok


def main():
    parser = argparse.ArgumentParser(description="Verify kernel benchmarks")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--hidden",       type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--group-size",   type=int, default=64)
    parser.add_argument("--B",  type=int, default=1,   help="GDN batch size")
    parser.add_argument("--T",  type=int, default=4,   help="GDN sequence length (depth+1)")
    parser.add_argument("--Hk", type=int, default=16,  help="GDN key heads")
    parser.add_argument("--Hv", type=int, default=64,  help="GDN value heads")
    parser.add_argument("--Dk", type=int, default=192, help="GDN key head dim")
    parser.add_argument("--Dv", type=int, default=128, help="GDN value head dim")
    args = parser.parse_args()

    if args.model:
        cfg = Path(args.model).expanduser() / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            args.hidden       = c.get("hidden_size", args.hidden)
            args.intermediate = c.get("intermediate_size", args.intermediate)
            qcfg = c.get("quantization", {})
            args.group_size   = qcfg.get("group_size", args.group_size)
            args.Hk = c.get("linear_num_key_heads",   args.Hk)
            args.Hv = c.get("linear_num_value_heads", args.Hv)
            args.Dk = c.get("linear_key_head_dim",    args.Dk)
            args.Dv = c.get("linear_value_head_dim",  args.Dv)
            print(f"Config loaded from {cfg}")

    if not mx.metal.is_available():
        print("Metal not available.")
        return

    mlp_ok = bench_mlp_fused(args.hidden, args.intermediate, args.group_size)
    gdn_ok = bench_gdn_tape(args.B, args.T, args.Hk, args.Hv, args.Dk, args.Dv)

    print("\nSummary:")
    print(f"  MLP fusion:  {'PASS' if mlp_ok else 'FAIL'}")
    print(f"  GDN tape:    {'PASS' if gdn_ok else 'FAIL'}")
    if mlp_ok:
        print("  → Enable with --mlp-fuse")
    if gdn_ok:
        print("  → Enable with --gdn-tape")


if __name__ == "__main__":
    main()
