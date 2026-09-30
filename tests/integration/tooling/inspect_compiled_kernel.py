"""Inspect an explicit cubin with the real CUDA Driver; print Markdown evidence."""
import argparse
import ctypes as c
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("symbol")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", default="libcuda.so.1")
    args = parser.parse_args()
    image = args.image.read_bytes()
    if not image.startswith(b"\x7fELF"):
        parser.error("provide a compiled ELF cubin, not PTX or an opaque pointer")
    driver = c.CDLL(args.library)

    def bind(name, parameters):
        function = driver[name]
        function.argtypes = parameters
        function.restype = c.c_int
        return function

    def check(status: int) -> None:
        if status:
            raise RuntimeError(f"CUDA Driver returned {status}")

    init = bind("cuInit", [c.c_uint])
    retain = bind("cuDevicePrimaryCtxRetain", [c.POINTER(c.c_void_p), c.c_int])
    current = bind("cuCtxSetCurrent", [c.c_void_p])
    load = bind("cuModuleLoadData", [c.POINTER(c.c_void_p), c.c_void_p])
    get_function = bind("cuModuleGetFunction", [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p])
    attribute = bind("cuFuncGetAttribute", [c.POINTER(c.c_int), c.c_int, c.c_void_p])
    parameter = bind("cuFuncGetParamInfo", [c.c_void_p, c.c_size_t, c.POINTER(c.c_size_t), c.POINTER(c.c_size_t)])
    unload = bind("cuModuleUnload", [c.c_void_p])
    release = bind("cuDevicePrimaryCtxRelease", [c.c_int])
    check(init(0))
    context, module, function = c.c_void_p(), c.c_void_p(), c.c_void_p()
    check(retain(c.byref(context), args.device))
    check(current(context))
    data = c.create_string_buffer(image)
    check(load(c.byref(module), data))
    check(get_function(c.byref(function), module, args.symbol.encode()))
    print(f"Image SHA-256: `{hashlib.sha256(image).hexdigest()}`\n")
    print(f"Symbol: `{args.symbol}`\n")
    print("| Driver function attribute | Value |\n|---|---:|")
    for label, enum in [("Max threads/block", 0), ("Static shared bytes", 1),
                        ("Constant bytes", 2), ("Local bytes/thread", 3),
                        ("Registers/thread", 4), ("PTX version attribute (raw)", 5),
                        ("Binary version attribute (raw)", 6), ("Max dynamic shared bytes", 8)]:
        value = c.c_int()
        check(attribute(c.byref(value), enum, function))
        print(f"| {label} | {value.value} |")
    print("\n| Parameter | Offset (bytes) | Size (bytes) |\n|---:|---:|---:|")
    index = 0
    while True:
        offset, size = c.c_size_t(), c.c_size_t()
        status = parameter(function, index, c.byref(offset), c.byref(size))
        if status == 1:  # CUDA_ERROR_INVALID_VALUE: parameter index out of range.
            break
        check(status)
        print(f"| {index} | {offset.value} | {size.value} |")
        index += 1
    check(unload(module))
    check(release(args.device))


if __name__ == "__main__":
    main()
