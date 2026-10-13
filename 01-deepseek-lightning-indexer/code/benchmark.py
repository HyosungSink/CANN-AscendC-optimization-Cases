#!/usr/bin/env python3
"""Single LightningIndexer call: CPU score check and synchronized host timing.
This authored harness is not an upstream kernel and contains no measured results.
"""
import argparse
import json
import platform
import re
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cann-version', required=True, help='Declared installed formal version, e.g. 9.0.0')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--s1', type=int, default=1)
    parser.add_argument('--s2', type=int, default=8192)
    parser.add_argument('--heads', type=int, default=64)
    parser.add_argument('--topk', type=int, default=2048)
    parser.add_argument('--dtype', choices=['fp16', 'bf16'], default='bf16')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--atol', type=float, default=0.02)
    parser.add_argument('--rtol', type=float, default=0.001)
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.cann_version) or tuple(map(int, args.cann_version.split('.'))) < (9, 0, 0):
        parser.error('--cann-version must identify a formal CANN version >= 9.0.0')
    if min(args.batch, args.s1, args.s2, args.heads, args.topk, args.repeat) <= 0 or args.warmup < 0:
        parser.error('Sizes/repeat must be positive and warmup nonnegative')
    if args.heads > 64 or args.topk > min(args.s2, 8192) or (args.topk > 2048 and args.topk % 1024):
        parser.error('This harness requires heads<=64 and topk<=min(S2,8192), with a 1024 multiple above 2048')
    import torch
    import torch_npu
    if not torch.npu.is_available() or not hasattr(torch_npu, 'npu_lightning_indexer'):
        raise RuntimeError('NPU or torch_npu.npu_lightning_indexer is unavailable; use pinned aclnn example')
    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    normalized_name = device_name.lower().replace('_', '')
    if '910b' not in normalized_name and '91093' not in normalized_name:
        raise RuntimeError(f'This harness is scoped to A2/A3; detected {device_name}')
    torch.manual_seed(123)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    q = (torch.randn(args.batch, args.s1, args.heads, 128) * 0.2).to(dtype)
    k = (torch.randn(args.batch, args.s2, 1, 128) * 0.2).to(dtype)
    w = torch.rand(args.batch, args.s1, args.heads).to(dtype)
    score = torch.einsum('bshd,btd->bsht', q.float(), k[:, :, 0].float()).relu()
    score = (score * w.float().unsqueeze(-1)).sum(dim=2)
    expected_scores = score.topk(args.topk, dim=-1).values
    qn, kn, wn = q.npu(), k.npu(), w.npu()

    def invoke():
        return torch_npu.npu_lightning_indexer(qn, kn, wn, layout_query='BSND', layout_key='BSND',
                                               sparse_count=args.topk, sparse_mode=0, return_value=False)

    with torch.inference_mode():
        result = invoke()
        torch.npu.synchronize()
        indices = result[0].cpu().long().squeeze(2)
        if tuple(indices.shape) != (args.batch, args.s1, args.topk):
            raise AssertionError(f'Unexpected index shape: {tuple(indices.shape)}')
        if bool(((indices < 0) | (indices >= args.s2)).any()):
            raise AssertionError('Output contains invalid indices for the complete unmasked input')
        ordered_indices = indices.sort(dim=-1).values
        if args.topk > 1 and bool((ordered_indices[..., 1:] == ordered_indices[..., :-1]).any()):
            raise AssertionError('Output contains duplicate indices')
        actual_scores = score.gather(-1, indices).sort(dim=-1, descending=True).values
        error = (actual_scores - expected_scores).abs()
        torch.testing.assert_close(actual_scores, expected_scores, atol=args.atol, rtol=args.rtol)
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
    report = {'operator': 'LightningIndexer', 'upstream_commit': 'afe72144f9f2ac8441929035795db88a111b30c5',
              'cann_version_declared': args.cann_version, 'version_verified_by_harness': False,
              'loaded_kernel_commit_verified_by_harness': False, 'torch_version': torch.__version__,
              'torch_npu_version': getattr(torch_npu, '__version__', 'unknown'), 'device_name': device_name,
              'host_platform': platform.platform(), 'parameters': vars(args), 'layout': 'BSND', 'sparse_mode': 0,
              'max_selected_score_error': float(error.max()), 'precision_check': 'passed',
              'timing_scope': 'host single-operator call plus NPU synchronize; includes dispatch overhead',
              'p50_us': statistics.median(times_us), 'p90_us': ordered[max(0, int(len(ordered) * 0.9 + 0.999) - 1)],
              'min_us': min(times_us), 'samples_us': times_us}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
