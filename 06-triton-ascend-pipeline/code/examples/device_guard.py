"""Strict A2/A3 name guard, independent of torch/NPU runtime imports."""
import re


_A2_A3_PATTERNS = (
    re.compile(r"ASCEND910B(?:[0-9][A-Z0-9-]*)?"),
    re.compile(r"ASCEND910_93[0-9]{2}"),
    re.compile(r"ASCEND910C(?:[0-9][A-Z0-9-]*)?"),
)


def require_a2_a3(device_name):
    """Accept known 910B/910_93xx/910C naming families; reject all others.

    Bare Ascend910 is deliberately ambiguous and does not establish A2/A3.
    Unknown names require checking the real board and its upstream support
    evidence before extending this allowlist; there is no bypass option.
    """
    canonical = str(device_name).strip().upper()
    if not any(pattern.fullmatch(canonical) for pattern in _A2_A3_PATTERNS):
        raise ValueError(
            f"Unconfirmed A2/A3 device name: {device_name!r}. "
            "Only Ascend910B variants, Ascend910_93xx and Ascend910C variants "
            "are accepted. Verify the actual board with npu-smi before "
            "extending the allowlist; this benchmark has no override."
        )

