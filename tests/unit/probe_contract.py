"""NCU unit conversion, artifact replay and malformed-artifact rejection."""
import csv
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

replay, probe = sys.argv[1:]
observations = runpy.run_path(probe)["observations"]
columns = ["ID", "Process ID", "Device", "Kernel Name", "Grid Size", "Block Size",
           "gpu__time_duration.sum", "launch__shared_mem_per_block_static",
           "launch__shared_mem_per_block_dynamic", "launch__registers_per_thread"]
units = ["", "", "", "", "", "", "us", "Kbyte/block", "Kbyte/block", "register/thread"]
sample = ["0", "42", "0", "contract_symbol", "(2, 1, 1)", "(128, 1, 1)", "1.234", "0.512", "49.152", "255"]
headers = "schema=1\nscope=isolated_kernel_observations\n" + "\n".join(
    f"{key}={prefix}{'0'*64}" for key, prefix in [("source", "ncu-report-sha256:"),
    ("code", "declared-image-sha256:"), ("hardware", "sha256:"), ("conditions", "sha256:")]) + "\n"

with tempfile.TemporaryDirectory() as directory:
    metrics, artifact = Path(directory)/"metrics.csv", Path(directory)/"kernels.probe"

    def write_csv(rows, unit_row=units):
        with metrics.open("w", newline="") as output:
            csv.writer(output).writerows([columns, unit_row, *rows])

    write_csv([sample])
    records = observations(metrics)
    assert records[0][4:8] == ["512", "49152", "255", "1234"]
    valid = headers + "observation=" + "\t".join(records[0]) + "\n"
    artifact.write_text(valid)
    result = subprocess.run([replay, str(artifact)], text=True, capture_output=True, check=True)
    assert "| 1 | 10000000 | 1234 | 1234 |" in result.stdout
    assert "host/arrival/transfer/dependency times were not collected" in result.stdout
    artifact.write_text(valid.replace("declared-image-sha256:", "captured-cubin-sha256:"))
    subprocess.run([replay, str(artifact)], capture_output=True, check=True)

    invalid_rows = []
    for column, value in [(6, "NaN"), (6, "-1"), (6, "0.0001"), (4, "(0, 1, 1)"), (3, "bad\nsymbol")]:
        row = sample.copy(); row[column] = value
        invalid_rows.append([row])
    invalid_rows += [[], [sample, sample]]
    for rows in invalid_rows:
        write_csv(rows)
        try:
            observations(metrics)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid NCU measurements accepted")
    invalid_units = units.copy(); invalid_units[6] = "cycles"
    write_csv([sample], invalid_units)
    try:
        observations(metrics)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown duration units accepted")

    invalid_artifacts = [valid.replace("schema=1", "schema=2"), valid.replace("isolated_kernel_observations", "whole_forward"),
                         valid.replace("1234", "-1"), valid.replace("2,1,1", "0,1,1"),
                         valid.replace("declared-image-sha256:", "unverified:"),
                         valid + "observation=" + "\t".join(records[0]) + "\n", headers,
                         "schema=1\n" + valid, valid + "unknown=field\n"]
    for text in invalid_artifacts:
        artifact.write_text(text)
        result = subprocess.run([replay, str(artifact)], text=True, capture_output=True)
        assert result.returncode != 0 and not result.stdout, result
print("PASS: probe units, unknown timing scope, isolated replay and 17 invalid inputs")
