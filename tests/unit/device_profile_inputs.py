"""Reject malformed configuration in fresh Driver processes; no timeouts."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

probe, fixture = sys.argv[1:]
environment = {key: value for key, value in os.environ.items()
               if key not in ("FAKE_CUDA_DEVICE_COUNT", "FAKE_CUDA_PROFILE")}

def reject(**settings):
    subprocess.run([probe, "0"], env=environment | settings, check=True)

for count in ("", "0", "-1", "1025", "4x", " 4", "+4", "999999999999999999999999"):
    reject(FAKE_CUDA_DEVICE_COUNT=count)

profile = Path(fixture).read_text()
cases = [
    profile.replace("schema=1", "schema=2"),
    profile.replace("memory_bytes=1048576", "memory_bytes=0"),
    profile.replace("memory_bytes=1048576", "memory_bytes=18446744073709551616"),
    profile + "attribute.1=256\n",
    profile + "attribute.01=256\n",
    profile + "garbage=x\n",
    profile.replace("source=synthetic\n", ""),
    profile + "attribute.1x=1\n",
]
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "invalid.profile"
    for contents in cases:
        path.write_text(contents)
        reject(FAKE_CUDA_PROFILE=str(path))
    reject(FAKE_CUDA_PROFILE=str(path.with_name("missing.profile")))
print("17 invalid configuration cases passed")
