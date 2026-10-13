#!/usr/bin/env python3
"""Single SparseFlashAttention call: CPU reference and synchronized host timing.
BSND, noncausal, full lengths; authored harness without recorded performance.
"""
import argparse
import json
import math
import platform
import re
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cann-version', required=True, help='Declared installed formal version, e.g. 9.0.0')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--s1', type=int, default=1)
    parser.add_argument('--s2', type=int, default=8192)
    parser.add_argument('--heads', type=int, default=128)
    parser.add_argument('--topk', type=int, default=2048, help='Number of sparse blocks per Query')
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--dtype', choices=['fp16', 'bf16'], default='fp16')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--atol', type=float, default=0.003)
    parser.add_argument('--rtol', type=float, default=0.03)
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.cann_version) or tuple(map(int, args.cann_version.split('.'))) < (9, 0, 0):
        parser.error('--cann-version must identify a formal CANN version >= 9.0.0')
    if min(args.batch, args.s1, args.s2, args.heads, args.topk, args.repeat) <= 0 or args.warmup < 0:
        parser.error('Sizes/repeat must be positive and warmup nonnegative')
    if args.heads not in [1, 2, 4, 8, 16, 32, 64, 128]:
        parser.error('heads must be a supported power of 2 in [1,128]')
    if args.block_size not in [1, 2, 4, 8, 16, 32, 64, 128] or args.s2 % args.block_size or args.topk > args.s2 // args.block_size:
        parser.error('block-size must be a supported power of 2, divide S2, and leave at least topk blocks')
    import torch
    import torch_npu
    if not torch.npu.is_available() or not hasattr(torch_npu, 'npu_sparse_flash_attention'):
        raise RuntimeError('NPU or torch_npu.npu_sparse_flash_attention is unavailable; use pinned aclnn example')
    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    if '910b' not in device_name.lower():
        raise RuntimeError(f'This pinned Host path is verified only for A2/910B; detected {device_name}')
    torch.manual_seed(123)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    q = (torch.randn(args.batch, args.s1, args.heads, 512) * 0.5).to(dtype)
    k = (torch.randn(args.batch, args.s2, 1, 512) * 0.5).to(dtype)
    qr = (torch.randn(args.batch, args.s1, args.heads, 64) * 0.5).to(dtype)
    kr = (torch.randn(args.batch, args.s2, 1, 64) * 0.5).to(dtype)
    indices = torch.stack([torch.randperm(args.s2 // args.block_size)[:args.topk]
                           for _ in range(args.batch * args.s1)]).reshape(args.batch, args.s1, 1, args.topk).int()
    scale = 1.0 / math.sqrt(576)
    expected = torch.empty(args.batch, args.s1, args.heads, 512, dtype=torch.float32)
    for batch in range(args.batch):
        for query in range(args.s1):
            block_indices = indices[batch, query, 0].long()
            positions = (block_indices[:, None] * args.block_size + torch.arange(args.block_size)).reshape(-1)
            selected_k = k[batch, positions, 0].float()
            selected_kr = kr[batch, positions, 0].float()
            logits = (q[batch, query].float() @ selected_k.T + qr[batch, query].float() @ selected_kr.T) * scale
            expected[batch, query] = logits.softmax(dim=-1) @ selected_k
    qn, kn, qrn, krn, idxn = q.npu(), k.npu(), qr.npu(), kr.npu(), indices.npu()

    def invoke():
        return torch_npu.npu_sparse_flash_attention(qn, kn, kn, idxn, scale,
                    query_rope=qrn, key_rope=krn, sparse_block_size=args.block_size,
                    layout_query='BSND', layout_kv='BSND', sparse_mode=0,
                    attention_mode=2, return_softmax_lse=False)

    with torch.inference_mode():
        result = invoke()
        torch.npu.synchronize()
        actual = result[0].cpu().float()
        error = (actual - expected).abs()
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
        for _ in range(args.warmup):
            invoke()
        torch.npu.synchronize()
        times_us = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            invoke()
            torch.npu.synchronize()
            times_us.append((time.perf_counter() - start) * 1e6)
    ordered = sorted(times_us)
    report = {'operator': 'SparseFlashAttention', 'upstream_commit': 'afe72144f9f2ac8441929035795db88a111b30c5',
              'cann_version_declared': args.cann_version, 'version_verified_by_harness': False,
              'loaded_kernel_commit_verified_by_harness': False, 'torch_version': torch.__version__,
              'torch_npu_version': getattr(torch_npu, '__version__', 'unknown'), 'device_name': device_name,
              'host_platform': platform.platform(), 'parameters': vars(args), 'layout': 'BSND', 'sparse_mode': 0,
              'max_absolute_error': float(error.max()), 'mean_absolute_error': float(error.mean()),
              'precision_check': 'passed',
              'timing_scope': 'host single-operator call plus NPU synchronize; includes dispatch overhead',
              'p50_us': statistics.median(times_us), 'p90_us': ordered[max(0, int(len(ordered) * 0.9 + 0.999) - 1)],
              'min_us': min(times_us), 'samples_us': times_us}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
