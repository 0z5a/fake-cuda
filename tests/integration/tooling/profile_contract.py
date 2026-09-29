"""Compare a captured capability profile with every exposed no-GPU device."""
import argparse
import ctypes as ct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", required=True)
configuration = parser.add_mutually_exclusive_group(required=True)
configuration.add_argument("--profile")
configuration.add_argument("--system")
parser.add_argument("--devices", type=int, required=True)
args = parser.parse_args()
assert not list(Path("/dev").glob("nvidia*"))
def fields(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines()
                if line and not line.startswith("#"))

system = fields(Path(args.system)) if args.system else None
driver = ct.CDLL(args.library)
assert driver.cuInit(0) == 0
count = ct.c_int()
assert driver.cuDeviceGetCount(ct.byref(count)) == 0 and count.value == args.devices
uuids: set[bytes] = set()
for ordinal in range(count.value):
    path = Path(args.system).parent / system[f"device.{ordinal}"] if system is not None else Path(args.profile)
    profile = fields(path)
    name = ct.create_string_buffer(256)
    memory = ct.c_size_t()
    uuid = ct.create_string_buffer(16)
    assert driver.cuDeviceGetName(name, len(name), ordinal) == 0
    assert name.value.decode() == profile["name"]
    assert driver.cuDeviceTotalMem_v2(ct.byref(memory), ordinal) == 0
    assert memory.value == int(profile["memory_bytes"])
    assert driver.cuDeviceGetUuid_v2(uuid, ordinal) == 0
    assert uuid.raw not in uuids
    uuids.add(uuid.raw)
    for key, expected in profile.items():
        if key.startswith("attribute."):
            value = ct.c_int()
            assert driver.cuDeviceGetAttribute(ct.byref(value), int(key[10:]), ordinal) == 0
            assert value.value == int(expected), (ordinal, key, value.value, expected)
if system is not None:
    driver.cuCtxCreate_v2.argtypes = [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int]
    driver.cuCtxSetCurrent.argtypes = [ct.c_void_p]
    driver.cuCtxEnablePeerAccess.argtypes = [ct.c_void_p, ct.c_uint]
    driver.cuCtxDisablePeerAccess.argtypes = [ct.c_void_p]
    driver.cuCtxDestroy_v2.argtypes = [ct.c_void_p]
    contexts = [ct.c_void_p() for _ in range(count.value)]
    for ordinal, context in enumerate(contexts):
        assert driver.cuCtxCreate_v2(ct.byref(context), 0, ordinal) == 0
    for source in range(count.value):
        assert driver.cuCtxSetCurrent(contexts[source]) == 0
        for peer in range(count.value):
            access = ct.c_int()
            expected = 0 if source == peer else int(system[f"peer.{source}.{peer}"])
            assert driver.cuDeviceCanAccessPeer(ct.byref(access), source, peer) == 0
            assert access.value == expected
            if source == peer:
                continue
            result = driver.cuCtxEnablePeerAccess(contexts[peer], 0)
            assert result == (0 if expected else 217), (source, peer, result)
            if expected:
                assert driver.cuCtxDisablePeerAccess(contexts[peer]) == 0
    for context in contexts:
        assert driver.cuCtxDestroy_v2(context) == 0
    print("PASS: complete directed peer matrix and context enable results")
print(f"PASS: {args.devices} virtual devices match captured name, memory and attributes; unique UUIDs")
