"""``pack_manifest``'s tensor payloads take a rank-0 tensor (a bf16 scalar used to refuse on write and on read: a byte
view of a rank-0 tensor refuses), and every ranked payload is byte for byte what it was before the fix."""
import hashlib

import pytest

torch = pytest.importorskip("torch")
from experts4bit_qlora.engines.pack_manifest import tensor_from_payload_bytes, tensor_payload_bytes  # noqa: E402


def _fixture(dtype):
    """A deterministic ranked (3 x 5) tensor from an arithmetic sequence (no RNG, the same on every platform);
    integer values in [-51, 51], non-negative for uint8."""
    base = (torch.arange(15, dtype=torch.float32).reshape(3, 5) - 7) * 0.7321
    if dtype.is_floating_point:
        return base.to(dtype)
    ints = (base * 10).round()
    return (ints.abs() if dtype == torch.uint8 else ints).to(dtype)


#: sha256 of tensor_payload_bytes(_fixture(dtype)), computed with the code before the fix (main at 2026-10-10)
PINNED = {
    torch.bfloat16: "0fbec7e6129b6c707d94c09a3d906136380d7a84ca5d17c57e346c8619385a6e",
    torch.float32: "04d8ef103445271c14162e724fba0e3c9a5e438b6e7c34601dcfef259a6e201f",
    torch.float16: "623eada3ef8367f2bb288e25b4d2dde26b0b51d9f8cf9399a7df8fccfeb5e57d",
    torch.int8: "e64a93944b5b49bdfb912bfcfcea8c3417cd60bbb27620cb808bd693f2a8cbb3",
    torch.uint8: "1da2fd38bab2995c53d0701e5608d9b8781cf7daf7cb0df34f768cb112da1f8c",
    torch.int32: "256ea19f95285abd1441f41351b3103fc768f64eba6e7eca11104c160e3e4516",
}


@pytest.mark.parametrize("dtype", list(PINNED))
def test_ranked_payloads_are_unchanged_and_round_trip(dtype):
    t = _fixture(dtype)
    data = tensor_payload_bytes(t)
    assert hashlib.sha256(data).hexdigest() == PINNED[dtype]
    back = tensor_from_payload_bytes(data)
    assert back.dtype == t.dtype and back.shape == t.shape and torch.equal(back, t)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16, torch.int8, torch.uint8, torch.int32])
def test_a_rank0_payload_writes_and_reads_bit_for_bit(dtype):
    t = torch.tensor(5, dtype=dtype) if not dtype.is_floating_point else torch.tensor(-2.5, dtype=dtype)
    data = tensor_payload_bytes(t)
    raw = t.reshape(1).view(torch.uint8).numpy().tobytes()           # the scalar's own bytes, read independently
    assert data.endswith(raw) and b'"shape":[]' in data
    back = tensor_from_payload_bytes(data)
    assert back.dtype == dtype and back.shape == () and torch.equal(back, t)
