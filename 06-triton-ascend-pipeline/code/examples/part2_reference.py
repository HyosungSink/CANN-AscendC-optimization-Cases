"""CPU golden for the full-page mapping fragment, not a complete KV allocator.

Based on the alloc_extend_kernel Part 2 mapping in Triton-Ascend v3.2.1.
Upstream MIT license is retained in ../upstream/LICENSE.
This scalar reference walks sequences and page allocations rather than modeling
the Triton prefix reduction or vector tile implementation.
"""


def full_page_reference(pre_lens, seq_lens, free_pages, page_size):
    if page_size <= 0 or len(pre_lens) != len(seq_lens):
        raise ValueError("Invalid page size or sequence arrays")
    if any(pre < 0 or seq < pre for pre, seq in zip(pre_lens, seq_lens)):
        raise ValueError("Require 0 <= pre_len <= seq_len")
    required_pages = sum(
        (seq + page_size - 1) // page_size - (pre + page_size - 1) // page_size
        for pre, seq in zip(pre_lens, seq_lens)
    )
    if len(free_pages) < required_pages:
        raise ValueError("Insufficient free pages")
    output = [-1] * sum(seq - pre for pre, seq in zip(pre_lens, seq_lens))
    out_cursor = 0
    page_cursor = 0
    for pre, seq in zip(pre_lens, seq_lens):
        first_full_page = (pre + page_size - 1) // page_size
        last_full_page_exclusive = seq // page_size
        write_cursor = out_cursor
        for page_number in range(first_full_page, last_full_page_exclusive):
            physical_page = free_pages[page_cursor + page_number - first_full_page]
            for offset_in_page in range(page_size):
                output[write_cursor] = physical_page * page_size + offset_in_page
                write_cursor += 1
        page_cursor += (seq + page_size - 1) // page_size - first_full_page
        out_cursor += seq - pre
    return output
