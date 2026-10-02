"""Reject malformed configuration in fresh Driver processes; no timeouts."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

probe, fixture = sys.argv[1:]
environment = {key: value for key, value in os.environ.items()
               if key not in ("FAKE_CUDA_DEVICE_COUNT", "FAKE_CUDA_PROFILE", "FAKE_CUDA_SYSTEM")}

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
    good_profile = Path(directory) / "device profile"
    good_profile.write_text(profile)
    system = Path(directory) / "devices.system"
    valid = ("schema=1\nsource=synthetic\ndevice.0=device profile\ndevice.1=device profile\n"
             "peer.0.1=1\npeer.1.0=0\n")
    malformed = [
        valid.replace("schema=1", "schema=2"),
        valid.replace("source=synthetic\n", ""),
        valid.replace("device.1=", "device.2="),
        valid.replace("device.1=", "device.1024="),
        valid.replace("device.1=device profile\n", ""),
        valid.replace("peer.1.0=0\n", ""),
        valid.replace("peer.1.0=0", "peer.2.0=0"),
        valid.replace("peer.1.0=0", "peer.1.0=2"),
        valid + "peer.0.0=0\n",
        valid + "peer.00.1=1\n",
        valid + "device.00=device profile\n",
        valid.replace("device profile", "missing.profile"),
        valid + "extra=1\n",
    ]
    for contents in malformed:
        system.write_text(contents)
        reject(FAKE_CUDA_SYSTEM=str(system))
    system.write_text(valid)
    reject(FAKE_CUDA_SYSTEM=str(system), FAKE_CUDA_PROFILE=str(good_profile))
    reject(FAKE_CUDA_SYSTEM=str(system), FAKE_CUDA_DEVICE_COUNT="2")
    subprocess.run([probe, "2", "synthetic"], env=environment | {"FAKE_CUDA_SYSTEM": str(system)}, check=True)
    many = "schema=1\nsource=synthetic\n"
    many += "".join(f"device.{i}={good_profile}\n" for i in range(16))
    many += "".join(f"peer.{i}.{j}={int(i < j)}\n" for i in range(16) for j in range(16) if i != j)
    system.write_text(many)
    subprocess.run([probe, "16", "synthetic"], env=environment | {"FAKE_CUDA_SYSTEM": str(system)}, check=True)
print("32 invalid cases, relative paths and a 16-device absolute-path system passed")
