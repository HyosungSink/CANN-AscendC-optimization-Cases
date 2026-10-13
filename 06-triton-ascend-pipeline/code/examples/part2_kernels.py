# Derived from alloc_extend_kernel Part 2 in Triton-Ascend v3.2.1:
# https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/docs/zh/migration_guide/performance_guidelines.md
# Upstream MIT license: ../upstream/LICENSE. The original document is unmodified
# in ../upstream/docs/zh/migration_guide/performance_guidelines.md.
# Adaptations: add explicit zero padding for prefix sums; expose three kernel
# bodies and a complete host benchmark. Only the Part 2 full-page mapping is
# implemented. This does not allocate partial head/tail pages or update KV state.
import triton
import triton.language as tl


@triton.jit
def _locations(pre_lens_ptr, seq_lens_ptr, pid, BS_UPPER: tl.constexpr, PAGE_SIZE: tl.constexpr):
    ids = tl.arange(0, BS_UPPER)
    # These values participate in reductions: masked lanes must remain zero.
    seq_lens = tl.load(seq_lens_ptr + ids, mask=ids <= pid, other=0)
    pre_lens = tl.load(pre_lens_ptr + ids, mask=ids <= pid, other=0)
    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    output_start = tl.sum(seq_lens - pre_lens) - (seq_len - pre_len)
    new_pages = tl.cdiv(seq_lens, PAGE_SIZE) - tl.cdiv(pre_lens, PAGE_SIZE)
    own_pages = tl.cdiv(seq_len, PAGE_SIZE) - tl.cdiv(pre_len, PAGE_SIZE)
    page_start = tl.sum(new_pages) - own_pages
    full_tokens = (seq_len // PAGE_SIZE - tl.cdiv(pre_len, PAGE_SIZE)) * PAGE_SIZE
    return output_start, page_start, full_tokens


@triton.jit
def single_tile(pre_ptr, seq_ptr, page_ptr, out_ptr,
                BS_UPPER: tl.constexpr, PAGE_SIZE: tl.constexpr,
                MAX_EXTEND: tl.constexpr):
    pid = tl.program_id(0)
    out_start, page_start, full_tokens = _locations(pre_ptr, seq_ptr, pid, BS_UPPER, PAGE_SIZE)
    offsets = tl.arange(0, MAX_EXTEND)
    mask = offsets < full_tokens
    pages = tl.load(page_ptr + page_start + offsets // PAGE_SIZE, mask=mask)
    tl.store(out_ptr + out_start + offsets,
             pages * PAGE_SIZE + offsets % PAGE_SIZE, mask=mask)


@triton.jit
def loop_tiles(pre_ptr, seq_ptr, page_ptr, out_ptr,
               BS_UPPER: tl.constexpr, PAGE_SIZE: tl.constexpr,
               MAX_EXTEND: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    out_start, page_start, full_tokens = _locations(pre_ptr, seq_ptr, pid, BS_UPPER, PAGE_SIZE)
    block = tl.arange(0, TILE)
    for iteration in range(tl.cdiv(MAX_EXTEND, TILE)):
        offsets = block + iteration * TILE
        mask = offsets < full_tokens
        pages = tl.load(page_ptr + page_start + offsets // PAGE_SIZE, mask=mask)
        tl.store(out_ptr + out_start + offsets,
                 pages * PAGE_SIZE + offsets % PAGE_SIZE, mask=mask)


@triton.jit
def loop_tiles_unpadded(pre_ptr, seq_ptr, page_ptr, out_ptr,
                        BS_UPPER: tl.constexpr, PAGE_SIZE: tl.constexpr,
                        MAX_EXTEND: tl.constexpr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    out_start, page_start, full_tokens = _locations(pre_ptr, seq_ptr, pid, BS_UPPER, PAGE_SIZE)
    block = tl.arange(0, TILE)
    for iteration in range(tl.cdiv(MAX_EXTEND, TILE)):
        offsets = block + iteration * TILE
        mask = offsets < full_tokens
        # Masked lanes do not contribute to a reduction and have the identical
        # store mask. Their padding values cannot affect an observable output.
        pages = tl.load(page_ptr + page_start + offsets // PAGE_SIZE,
                        mask=mask, care_padding=False)
        tl.store(out_ptr + out_start + offsets,
                 pages * PAGE_SIZE + offsets % PAGE_SIZE, mask=mask)
