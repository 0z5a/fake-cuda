"""Compare a captured capability profile with every exposed no-GPU device."""
import argparse
import ctypes as ct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", required=True)
parser.add_argument("--profile", required=True)
parser.add_argument("--devices", type=int, required=True)
args = parser.parse_args()
assert not list(Path("/dev").glob("nvidia*"))
profile = dict(line.split("=", 1) for line in Path(args.profile).read_text().splitlines()
               if line and not line.startswith("#"))
driver = ct.CDLL(args.library)
assert driver.cuInit(0) == 0
count = ct.c_int()
assert driver.cuDeviceGetCount(ct.byref(count)) == 0 and count.value == args.devices
uuids: set[bytes] = set()
for ordinal in range(count.value):
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
print(f"PASS: {args.devices} virtual devices match captured name, memory and attributes; unique UUIDs")
