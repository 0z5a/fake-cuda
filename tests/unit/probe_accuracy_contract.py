"""Independent-report scoring, coverage and leakage rejection."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

replay, evaluator = sys.argv[1:]
sys.path.insert(0, str(Path(evaluator).parent))
from evaluate_probe import Sample, freeze, metrics

score = metrics([(2000, 2000), (2000, 4000)])
assert score["mape"] == 25 and score["p95"] == 50 and score["maximum"] == 50
assert abs(score["wape"] - 100/3) < 1e-9
assert score["bias"] == -25 and score["within10"] == 50
model = freeze([Sample("a", ("known",), 1000), Sample("b", ("known",), 3000)])
assert model == {("known",): 2000} and ("unseen",) not in model
for pairs in [[], [(1, 0)], [(float("nan"), 1)], [(-1, 1)]]:
    try:
        metrics(pairs)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid accuracy inputs accepted")

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)

    def artifact(name: str, duration: int, grid: int = 1) -> Path:
        folder = root/name
        folder.mkdir()
        headers = ["schema=1", "scope=isolated_kernel_observations", "code=captured-cubin-sha256:" + "0"*64]
        for key, file, content, prefix in [
            ("source", "capture.ncu-rep", name, "ncu-report-sha256:"),
            ("hardware", "hardware.csv", "same GPU", "sha256:"),
            ("conditions", "capture.txt", "same settings", "sha256:")]:
            (folder/file).write_text(content)
            headers.append(f"{key}={prefix}{hashlib.sha256(content.encode()).hexdigest()}")
        path = folder/"kernels.probe"
        path.write_text("\n".join(headers) + f"\nobservation=1\t0\t{grid},1,1\t128,1,1\t0\t0\t32\t{duration}\tsymbol\n")
        return path

    a, b = artifact("train-a", 1000), artifact("train-b", 3000)
    c, d, e = artifact("test-a", 2000), artifact("test-b", 4000), artifact("test-unseen", 5000, 2)

    def evaluate(training: list[Path], validation: list[Path]) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, evaluator, "--replay", replay, "--conditions", "synthetic contract",
                               "--calibration", *map(str, training), "--validation", *map(str, validation)],
                              text=True, capture_output=True)

    result = evaluate([a, b], [c, d, e])
    assert result.returncode == 0, result.stderr
    assert "| Frozen probe / covered | 2/3 | 25.000 | 50.000 | 50.000 | 33.333 | -25.000 | 50.000 |" in result.stdout
    assert "test-unseen/1 | unsupported" in result.stdout
    for training, validation in [([a], [a]), ([a, a], [c]), ([a], [c, c])]:
        result = evaluate(training, validation)
        assert result.returncode != 0 and not result.stdout
    (c.parent/"capture.ncu-rep").write_text("tampered")
    result = evaluate([a], [c])
    assert result.returncode != 0 and not result.stdout
print("PASS: independent probe scoring, unknown coverage, report separation and digest verification")
