"""Check condition isolation and unsupported coverage using retained native input."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("rank_campaign.py")),
               "--config-sha256", args.config_sha256, "--bridge", str(args.bridge)]
    reference = json.loads(subprocess.check_output(
        command + ["--evidence", str(args.evidence), "--limit", "1"], text=True))
    assert len(reference["rows"]) == 1 and reference["legacy_calibration"]
    calibration = json.loads((args.evidence / "calibration.json").read_text())
    original = json.loads((args.evidence / "validation.json").read_text())[0]

    def observed(case: dict, before: list[str], after: list[str], scale: int = 1) -> dict:
        result = copy.deepcopy(case)
        result.update(background_jobs_before=before, background_jobs_after=after)
        for step in result["steps"]:
            step["duration_ns"] *= scale
        return result

    a, b = ["background-a", "same-peer"], ["background-b", "same-peer"]
    fitted = [observed(case, a, a) for case in calibration]
    fitted += [observed(case, b, b, 2) for case in calibration]
    # Changed-load samples must not poison either frozen model.
    fitted.append(observed(calibration[0], a, b, 1000))
    accepted = [observed(original, a, list(reversed(a))), observed(original, b, b)]
    unsupported = [observed(original, a, b), observed(original, ["unseen"], ["unseen"]),
                   copy.deepcopy(original), observed(original, a, a)]
    del unsupported[-1]["background_jobs_after"]

    with tempfile.TemporaryDirectory(prefix="rank-conditions-") as directory:
        root = Path(directory)
        shutil.copyfile(args.evidence / "scheduler-config.pkl", root / "scheduler-config.pkl")
        (root / "calibration.json").write_text(json.dumps(fitted))
        (root / "validation.json").write_text(json.dumps(accepted + unsupported))
        result = json.loads(subprocess.check_output(command + ["--evidence", directory], text=True))
        assert result["rejected_calibration_runs"] == 1 and result["calibrated_conditions"] == 2
        assert not result["legacy_calibration"]
        assert result["rows"][0]["metrics"] == reference["rows"][0]["metrics"]
        assert result["rows"][0]["errors_percent"] == reference["rows"][0]["errors_percent"]
        # Each interpolated prediction is independently rounded to integer ns.
        assert abs(result["rows"][1]["metrics"]["finish_ns"] -
                   2 * result["rows"][0]["metrics"]["finish_ns"]) <= len(original["steps"])
        assert result["covered_steps"] == 2 * len(original["steps"])
        assert result["total_steps"] == 6 * len(original["steps"])
        assert [case["reason"] for case in result["unsupported"]] == [
            "background changed during workload", "uncalibrated background condition",
            "uncalibrated background condition", "incomplete background observations"]
        (root / "validation.json").write_text(json.dumps(unsupported))
        empty = json.loads(subprocess.check_output(command + ["--evidence", directory], text=True))
        assert empty["covered_steps"] == 0 and not empty["rows"]
        assert empty["step_wape_percent"] is None and empty["step_mape_percent"] is None
        rejected_speed = subprocess.run([sys.executable,
            str(Path(__file__).with_name("benchmark_rank_campaign.py")),
            "--evidence", directory, "--config-sha256", args.config_sha256,
            "--bridge", str(args.bridge), "--limit", "1"], capture_output=True, text=True)
        assert rejected_speed.returncode != 0
        assert "speed comparison requires complete supported workloads" in rejected_speed.stderr
        assert "Speedup" not in rejected_speed.stdout
    print("PASS frozen condition isolation, reordered snapshots, changed/incomplete/unseen/legacy coverage, empty errors unknown, incomplete speed rejected")


if __name__ == "__main__":
    main()
