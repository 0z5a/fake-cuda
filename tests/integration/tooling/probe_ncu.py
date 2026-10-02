"""Capture native NCU observations and emit an isolated-kernel replay artifact."""
import argparse
import csv
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def quantity(value: str, unit: str, factors: dict[str, int]) -> int:
    number = Decimal(value.replace(",", "")) * factors[unit]
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"invalid integral measurement: {value} {unit}")
    return int(number)


def dimensions(value: str) -> str:
    match = re.fullmatch(r"\((\d+),\s*(\d+),\s*(\d+)\)", value)
    if not match or not all(int(part) > 0 for part in match.groups()):
        raise ValueError(f"invalid launch dimensions: {value}")
    return ",".join(match.groups())


def observations(path: Path) -> list[list[str]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        units = next(reader)
        records = []
        seen = set()
        for row in reader:
            identity = (row["Process ID"], row["ID"])
            if identity in seen:
                raise ValueError("duplicate NCU observation")
            seen.add(identity)
            symbol = row["Kernel Name"]
            if not symbol or any(char in symbol for char in "\t\r\n"):
                raise ValueError("missing or multiline kernel symbol")
            time = quantity(row["gpu__time_duration.sum"], units["gpu__time_duration.sum"],
                            {"ns": 1, "us": 1000, "ms": 1000000, "s": 1000000000})
            resources = []
            for name in ("launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic"):
                resources.append(quantity(row[name], units[name],
                                          {"byte/block": 1, "Kbyte/block": 1000, "Mbyte/block": 1000000}))
            registers = quantity(row["launch__registers_per_thread"], units["launch__registers_per_thread"],
                                 {"register/thread": 1})
            records.append([str(len(records)+1), str(int(row["Device"])), dimensions(row["Grid Size"]),
                            dimensions(row["Block Size"]), *(str(value) for value in resources),
                            str(registers), str(time), symbol])
    if not records:
        raise ValueError("no profiled kernels")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    images = parser.add_mutually_exclusive_group(required=True)
    images.add_argument("--image", type=Path, help="explicit compiled image/container associated with this run")
    images.add_argument("--capture-cute", action="store_true", help="save one CuTe JIT variant from this capture process")
    parser.add_argument("--conditions", required=True, help="workload identity and, on import, original profiling settings")
    parser.add_argument("--report", type=Path, help="import an existing NCU report instead of launching a command")
    parser.add_argument("--hardware", type=Path, help="original nvidia-smi identity CSV; required for report import")
    parser.add_argument("--kernel", default="regex:.*")
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--count", type=int, default=1, help="observations per launch configuration")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.skip < 0 or args.count < 1 or bool(args.report) == bool(command):
        parser.error("provide a report or a command, nonnegative skip and positive count")
    if args.report and not args.hardware:
        parser.error("report import requires the original --hardware identity snapshot")
    if args.report and args.capture_cute:
        parser.error("CuTe artifact capture requires a command")
    code_hash = digest(args.image) if args.image else None
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    if args.capture_cute:
        compiler = args.output / "compiler"
        compiler.mkdir()
        environment.update(CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(compiler.resolve()),
                           CUTE_DSL_CACHE_DIR=str((compiler / "cache").resolve()))
    else:
        shutil.copyfile(args.image, args.output / "compiled.image")
        if digest(args.output / "compiled.image") != code_hash:
            raise ValueError("compiled image changed while copying")
    report = args.output / "capture.ncu-rep"
    hardware = args.output / "hardware.csv"
    if args.hardware:
        shutil.copyfile(args.hardware, hardware)
    else:
        hardware.write_text(subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pci.bus_id", "--format=csv"], text=True))
    version = subprocess.check_output([args.ncu, "--version"], text=True)
    settings = ["--clock-control", "none", "--cache-control", "all", "--replay-mode", "kernel",
                "--target-processes", "application-only",
                "--filter-mode", "per-launch-config", "--kernel-name-base", "demangled",
                "--kernel-name", args.kernel, "--launch-skip", str(args.skip), "--launch-count", str(args.count),
                "--section", "LaunchStats", "--section", "Occupancy", "--section", "SpeedOfLight"]
    manifest = f"origin={'import' if args.report else 'capture'}\nconditions={args.conditions}\n{version}"
    manifest += f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
    if args.report:
        shutil.copyfile(args.report, report)
    else:
        invocation = [args.ncu, *settings, "--export", str(report), *command]
        manifest += f"command={shlex.join(invocation)}\n"
        with (args.output / "capture.log").open("w") as output:
            subprocess.run(invocation, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True)
    if args.capture_cute:
        cubins = list(compiler.glob("*.cubin"))
        if len(cubins) != 1:
            raise ValueError("capture one CuTe specialization per command; expected one dumped cubin")
        args.image = cubins[0]
        code_hash = digest(args.image)
        shutil.copyfile(args.image, args.output / "compiled.image")
    if digest(args.image) != code_hash:
        raise ValueError("compiled image changed during profiling")
    manifest += f"image={args.image.resolve()}\ncode_sha256={code_hash}\n"
    manifest += f"report_sha256={digest(report)}\nhardware_sha256={digest(hardware)}\n"
    manifest_path = args.output / "capture.txt"
    manifest_path.write_text(manifest)
    metrics = args.output / "metrics.csv"
    with metrics.open("w") as output:
        subprocess.run([args.ncu, "--import", str(report), "--page", "raw", "--csv",
                        "--print-kernel-base", "mangled", "--rename-kernels", "0"], stdout=output, check=True)
    records = observations(metrics)
    if args.capture_cute and len(records) != 1:
        raise ValueError("CuTe capture requires one isolated kernel observation per artifact")
    code_kind = "captured-cubin" if args.capture_cute else "declared-image"
    headers = ["schema=1", "scope=isolated_kernel_observations", f"source=ncu-report-sha256:{digest(report)}",
               f"code={code_kind}-sha256:{code_hash}", f"hardware=sha256:{digest(hardware)}",
               f"conditions=sha256:{digest(manifest_path)}"]
    artifact = "\n".join(headers) + "\n" + "".join("observation=" + "\t".join(row) + "\n" for row in records)
    (args.output / "kernels.probe").write_text(artifact)
    print(f"Captured {len(records)} isolated kernel observations: {args.output / 'kernels.probe'}")


if __name__ == "__main__":
    main()
