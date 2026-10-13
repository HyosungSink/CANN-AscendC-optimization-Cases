"""NPU-only correctness and event timing for the real full-page mapping rewrite.

Run after preparing Triton-Ascend v3.2.1 / CANN 9.0.0 on Atlas A2/A3.
No package installation, model download, or distributed model execution.
"""
import argparse
import importlib.metadata
import json
import platform
import random
import statistics
from pathlib import Path

from part2_reference import full_page_reference
from device_guard import require_a2_a3


def make_case(batch, max_extend, page_size, seed):
    rng = random.Random(seed)
    pre = [rng.randrange(0, 81) for _ in range(batch)]
    extend = [rng.randrange(max(1, max_extend // 2), max_extend + 1) for _ in range(batch)]
    # Include a short, masked partial tile and a zero-work sequence.
    if batch > 1:
        pre[0] = 0
        extend[0] = min(max_extend, page_size * 3 + 7)
        extend[1] = 0
    seq = [p + e for p, e in zip(pre, extend)]
    count = sum((s + page_size - 1) // page_size - (p + page_size - 1) // page_size
                for p, s in zip(pre, seq))
    pages = list(range(1000, 1000 + count))
    rng.shuffle(pages)
    return pre, seq, pages


def event_bench(torch, fn, warmups, repeats):
    for _ in range(warmups):
        fn()
    torch.npu.synchronize()
    samples_us = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    return {
        'median_us': statistics.median(samples_us),
        'mean_us': statistics.mean(samples_us),
        'min_us': min(samples_us),
        'max_us': max(samples_us),
        'samples_us': samples_us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--max-extend', type=int, nargs='+', default=[128, 1025, 4096])
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.batch <= 0 or args.page_size <= 0 or args.repeats <= 0 or args.warmups < 0:
        parser.error('Require positive batch/page-size/repeats and non-negative warmups')
    if any(x <= 0 for x in args.max_extend) or args.tile <= 0 or args.tile & (args.tile - 1):
        parser.error('max-extend must be positive; tile must be a positive power of two')
    import torch
    import torch_npu  # noqa: F401
    import triton
    from part2_kernels import single_tile, loop_tiles, loop_tiles_unpadded
    if not torch.npu.is_available():
        raise SystemExit('Atlas A2/A3 NPU environment required')
    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    try:
        require_a2_a3(device_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    versions = {}
    for package in ('triton-ascend', 'torch', 'torch-npu'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unknown; verify source checkout and build separately'
    results = {
        'scope': 'alloc_extend_kernel Part 2 full-page mapping only',
        'expected_upstream_commit': '2badfc89e70a9b7a5e88463a116c2feddce4b101',
        'device_name': device_name, 'platform': platform.platform(),
        'package_versions': versions, 'arguments': {k: str(v) if isinstance(v, Path) else v
                                                  for k, v in vars(args).items()},
        'timing': 'NPU Event interval; compilation/input creation/CPU golden excluded; warm reused buffers',
        'cases': [],
    }
    for max_extend in args.max_extend:
        pre, seq, pages = make_case(args.batch, max_extend, args.page_size, args.seed)
        expected = full_page_reference(pre, seq, pages, args.page_size)
        device = f'npu:{args.device}'
        pre_t = torch.tensor(pre, dtype=torch.int32, device=device)
        seq_t = torch.tensor(seq, dtype=torch.int32, device=device)
        page_t = torch.tensor(pages, dtype=torch.int32, device=device)
        max_block = triton.next_power_of_2(max_extend)
        common = {'BS_UPPER': triton.next_power_of_2(args.batch),
                  'PAGE_SIZE': args.page_size, 'MAX_EXTEND': max_block}
        variants = [('single_tile', single_tile, False, {}),
                    ('loop_no_multibuffer', loop_tiles, False, {'TILE': args.tile}),
                    ('loop_multibuffer', loop_tiles, True, {'TILE': args.tile}),
                    ('loop_multibuffer_unpadded', loop_tiles_unpadded, True, {'TILE': args.tile})]
        case = {'max_extend': max_extend, 'compiled_max_extend': max_block,
                'pre_lens': pre, 'seq_lens': seq, 'free_pages': pages, 'variants': []}
        for name, kernel, multibuffer, extra in variants:
            out = torch.full((len(expected),), -1, dtype=torch.int32, device=device)
            def run(kernel=kernel, out=out, multibuffer=multibuffer, extra=extra):
                kernel[(args.batch,)](pre_t, seq_t, page_t, out,
                                       **common, **extra, multibuffer=multibuffer)
            run()
            torch.npu.synchronize()
            # Compare ALL output slots, including unwritten head/tail sentinels.
            if out.cpu().tolist() != expected:
                raise AssertionError(f'Golden mismatch: {max_extend=} {name=}')
            timing = event_bench(torch, run, args.warmups, args.repeats)
            row = {'name': name, 'golden': 'PASS', 'multibuffer': multibuffer, **timing}
            case['variants'].append(row)
            print(f'{max_extend=} {name}: golden=PASS median={timing["median_us"]:.3f} us')
        results['cases'].append(case)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
