"""Native Dynamo line protocol with causally delivered load release."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

os.environ["DYN_ROUTER_USE_KV_EVENTS"] = "false"
os.environ["DYN_ROUTER_ASSUME_KV_REUSE"] = "false"
# Preserve the protocol fd before moving native tracing from stdout to stderr.
protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.dynamo.native import NativeRouter


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    router = NativeRouter()
    try:
        for endpoint in range(args.workers):
            await router.worker(endpoint + 1)
        for line in sys.stdin:
            command = json.loads(line)
            loads = [row for partition in router.service.loads() for row in partition["loads"]]
            if command["operation"] == "choose":
                worker = await router.book(command["request"], command["tokens"], command["output"])
                response = {"endpoint": worker - 1, "loads": loads}
            elif command["operation"] == "feedback":
                for identity in command["prefills"]:
                    await router.prefill_complete(identity)
                for identity in command["finished"]:
                    await router.release(identity)
                response = {"ok": True}
            elif command["operation"] == "loads":
                response = {"loads": loads}
            else:
                raise ValueError("unknown Dynamo operation")
            print(json.dumps(response), file=protocol, flush=True)
    finally:
        await router.close()
        protocol.close()


if __name__ == "__main__":
    asyncio.run(main())
