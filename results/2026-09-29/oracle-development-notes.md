# Oracle development observations

These are transcribed diagnostic observations, not passing test logs.

- The first real-Driver capture-mode probe included a null mode pointer and exited with signal 11/status 139. Its redirected log was empty. No external signal was sent to the process.
- After removing that input, real `cuThreadExchangeStreamCaptureMode` with enum value 99 returned 0, while the simulator's input-validation test expected 1. Real conformance runs therefore use only documented mode values; malformed-mode checks are simulator-only.
- The next real probe expected capture to remain active after a wrong-thread end; its owner-thread `cuStreamEndCapture` returned 401 instead of 0. A separate diagnostic observed: begin=0, wrong-thread end=908, subsequent status=NONE on both threads, owner end=401. The implementation and final probe now preserve this measured lifecycle.
- A separate real allocation diagnostic observed: global begin=0, synchronous allocation=900, capture status=INVALIDATED, end=901 with null graph. Relaxed allocation passes in the final real/fake probe.
- PyTorch stack-symbolization diagnostics produced `ValueError: stoi`; the underlying capture issue was subsequently located in the installed source's RNG initialization and fixed by the mode-exchange implementation. `fake-torch-stack.log` preserves that diagnostic failure.
